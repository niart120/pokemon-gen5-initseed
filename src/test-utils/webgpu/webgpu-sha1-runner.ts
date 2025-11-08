/// <reference types="@webgpu/types" />

import { WORDS_PER_HASH, WORDS_PER_MESSAGE, type GpuSha1WorkloadConfig } from '@/test-utils/perf/sha1-webgpu-harness';
import { buildGeneratedSha1ShaderSource, buildSha1ShaderSource } from './webgpu-sha1-shaders';

const DEFAULT_WORKGROUP_SIZE = 128;
const BYTES_PER_WORD = Uint32Array.BYTES_PER_ELEMENT;
const BYTES_PER_MESSAGE = WORDS_PER_MESSAGE * BYTES_PER_WORD;
const BYTES_PER_HASH = WORDS_PER_HASH * BYTES_PER_WORD;

export interface WebGpuSha1ProfilingSample {
  totalMs: number;
  uploadMs: number;
  dispatchMs: number;
  readbackMs: number;
  batches: number;
  totalMessages: number;
  maxBatchMessages: number;
  details?: WebGpuSha1BatchDetail[];
}

export interface WebGpuSha1BatchDetail {
  slot: number;
  messageCount: number;
  uploadMs: number;
  dispatchMs: number;
  readbackMs: number;
  totalMs: number;
  submitToGpuDoneMs: number | null;
  gpuDoneToMapMs: number | null;
  queueSubmitMs: number;
  gpuIdleBeforeMs: number | null;
}

type WebGpuSha1ProfilingAccumulator = WebGpuSha1ProfilingSample & { details: WebGpuSha1BatchDetail[] };

interface BufferSet {
  input: GPUBuffer | null;
  inputSize: number;
  output: GPUBuffer | null;
  outputSize: number;
  readback: GPUBuffer | null;
  readbackSize: number;
  bindGroup: GPUBindGroup | null;
}

interface GeneratedBufferSet {
  output: GPUBuffer | null;
  outputSize: number;
  readback: GPUBuffer | null;
  readbackSize: number;
  bindGroup: GPUBindGroup | null;
}

export class WebGpuSha1Runner {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private configBuffer: GPUBuffer | null = null;
  private configData: Uint32Array | null = null;
  private generatedPipeline: GPUComputePipeline | null = null;
  private generatedBindGroupLayout: GPUBindGroupLayout | null = null;
  private generatedConfigBuffer: GPUBuffer | null = null;
  private generatedConfigData: Uint32Array | null = null;
  private generatedBufferSets: GeneratedBufferSet[] = [];
  private generatedConfigSignature: string | null = null;
  private generatedLastUploadedMessageCount: number | null = null;
  private generatedLastUploadedBaseOffset: number | null = null;
  private maxMessagesPerDispatch: number | null = null;
  private maxMessagesOverride: number | null = null;
  private lastProfiling: WebGpuSha1ProfilingSample | null = null;
  private bufferSets: BufferSet[] = [];
  private readonly bufferSetCount = 2;
  private workgroupSize = DEFAULT_WORKGROUP_SIZE;
  private lastGpuDoneAt: number | null = null;
  private lastUploadedConfigCount: number | null = null;

  public async init(): Promise<void> {
    if (this.device) {
      return;
    }

    const gpu = typeof navigator !== 'undefined' ? navigator.gpu : undefined;
    if (!gpu) {
      throw new Error('WebGPU is not available in this environment');
    }

    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error('Failed to acquire WebGPU adapter');
    }

    const device = await adapter.requestDevice();

    this.device = device;
    this.configBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.configData = new Uint32Array(4);
    const generatedConfigArray = new Uint32Array(24);
    this.generatedConfigData = generatedConfigArray;
    this.generatedConfigBuffer = device.createBuffer({
      size: this.alignSize(generatedConfigArray.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.initializeBufferSets();
    this.initializeGeneratedBufferSets();
    await this.recreatePipelines();
  }

  public async compute(messages: Uint32Array): Promise<Uint32Array> {
    if (messages.length === 0) {
      this.lastProfiling = null;
      return new Uint32Array(0);
    }

    await this.init();

    const totalMessages = this.validateMessageBuffer(messages);
    const maxBatchSize = this.getMaxMessagesPerDispatch();
    const results = new Uint32Array(totalMessages * WORDS_PER_HASH);
    const profiling: WebGpuSha1ProfilingAccumulator = {
      totalMs: 0,
      uploadMs: 0,
      dispatchMs: 0,
      readbackMs: 0,
      batches: 0,
      totalMessages: 0,
      maxBatchMessages: 0,
      details: [],
    };

    this.lastGpuDoneAt = null;
    this.lastUploadedConfigCount = null;

    const inFlight: Promise<void>[] = Array.from({ length: this.bufferSetCount }, () => Promise.resolve());
    let processedMessages = 0;
    let batchIndex = 0;
    while (processedMessages < totalMessages) {
      const remaining = totalMessages - processedMessages;
      const batchSize = Math.min(maxBatchSize, remaining);
      const startWord = processedMessages * WORDS_PER_MESSAGE;
      const endWord = startWord + batchSize * WORDS_PER_MESSAGE;
      const batchView = messages.subarray(startWord, endWord);

      const slot = batchIndex % this.bufferSetCount;
      await inFlight[slot];
      const bufferSet = this.ensureBufferSet(slot, batchSize);
      const outStart = processedMessages * WORDS_PER_HASH;
      inFlight[slot] = this.scheduleBatch(batchView, batchSize, results, outStart, profiling, slot, bufferSet);
      processedMessages += batchSize;
      batchIndex += 1;
    }

    await Promise.all(inFlight);
    profiling.totalMessages = processedMessages;
    this.lastProfiling = {
      ...profiling,
      details: profiling.details.map((entry) => ({ ...entry })),
    };
    return results;
  }

  public async computeGenerated(
    workload: GpuSha1WorkloadConfig,
    baseOffset: number,
    messageCount: number
  ): Promise<Uint32Array> {
    if (!Number.isFinite(baseOffset) || baseOffset < 0 || !Number.isInteger(baseOffset)) {
      throw new Error('baseOffset must be a non-negative integer');
    }
    if (!Number.isFinite(messageCount) || messageCount < 0 || !Number.isInteger(messageCount)) {
      throw new Error('messageCount must be a non-negative integer');
    }

    if (messageCount === 0) {
      this.lastProfiling = null;
      return new Uint32Array(0);
    }

    await this.init();

    if (baseOffset + messageCount > workload.totalMessages) {
      throw new Error('Requested range exceeds workload total messages');
    }

    this.configureGeneratedWorkload(workload);

    const totalResults = new Uint32Array(messageCount * WORDS_PER_HASH);
    const maxBatchSize = this.getMaxMessagesPerDispatch();
    const profiling: WebGpuSha1ProfilingAccumulator = {
      totalMs: 0,
      uploadMs: 0,
      dispatchMs: 0,
      readbackMs: 0,
      batches: 0,
      totalMessages: 0,
      maxBatchMessages: 0,
      details: [],
    };

    this.lastGpuDoneAt = null;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;

    const inFlight: Promise<void>[] = Array.from({ length: this.bufferSetCount }, () => Promise.resolve());
    let processedMessages = 0;
    let batchIndex = 0;

    while (processedMessages < messageCount) {
      const remaining = messageCount - processedMessages;
      const batchSize = Math.min(maxBatchSize, remaining);
      const slot = batchIndex % this.bufferSetCount;
      await inFlight[slot];
      const bufferSet = this.ensureGeneratedBufferSet(slot, batchSize);
      const outStart = processedMessages * WORDS_PER_HASH;
      const batchBaseOffset = baseOffset + processedMessages;
      inFlight[slot] = this.scheduleGeneratedBatch(
        batchSize,
        batchBaseOffset,
        totalResults,
        outStart,
        profiling,
        slot,
        bufferSet
      );
      processedMessages += batchSize;
      batchIndex += 1;
    }

    await Promise.all(inFlight);
    profiling.totalMessages = messageCount;
    this.lastProfiling = {
      ...profiling,
      details: profiling.details.map((entry) => ({ ...entry })),
    };
    return totalResults;
  }

  public getLastProfiling(): WebGpuSha1ProfilingSample | null {
    if (!this.lastProfiling) {
      return null;
    }
    const { details, ...rest } = this.lastProfiling;
    return {
      ...rest,
      details: details?.map((entry) => ({ ...entry })),
    };
  }

  public getDispatchMessageCapacity(): number {
    if (!this.device) {
      throw new Error('WebGPU device not initialized');
    }
    return this.getMaxMessagesPerDispatch();
  }

  public setMaxMessagesPerDispatch(limit: number | null | undefined): void {
    if (limit == null) {
      this.maxMessagesOverride = null;
      this.maxMessagesPerDispatch = null;
      return;
    }

    if (!Number.isFinite(limit) || limit <= 0) {
      throw new Error('maxMessagesPerDispatch override must be a positive finite number');
    }

    this.maxMessagesOverride = Math.max(1, Math.floor(limit));
    this.maxMessagesPerDispatch = null;
  }

  public async setWorkgroupSize(size: number): Promise<void> {
    if (!Number.isFinite(size) || size <= 0 || !Number.isInteger(size)) {
      throw new Error('workgroup size must be a positive integer');
    }

    if (this.workgroupSize === size) {
      return;
    }

    this.workgroupSize = size;
    this.maxMessagesPerDispatch = null;

    if (this.device) {
      await this.recreatePipelines();
    }
  }

  public dispose(): void {
    this.configBuffer?.destroy();
    this.configBuffer = null;
    this.configData = null;
    this.generatedConfigBuffer?.destroy();
    this.generatedConfigBuffer = null;
    this.generatedConfigData = null;
    this.releaseBufferSets();
    this.releaseGeneratedBufferSets();
    this.bufferSets = [];
    this.generatedBufferSets = [];
    this.pipeline = null;
    this.generatedPipeline = null;
    this.bindGroupLayout = null;
    this.generatedBindGroupLayout = null;
    this.device = null;
    this.lastUploadedConfigCount = null;
    this.generatedConfigSignature = null;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;
  }

  private getMaxMessagesPerDispatch(): number {
    if (this.maxMessagesOverride !== null) {
      return this.maxMessagesOverride;
    }

    if (this.maxMessagesPerDispatch !== null) {
      return this.maxMessagesPerDispatch;
    }

    const device = this.device!;
    const storageLimit = device.limits.maxStorageBufferBindingSize || BYTES_PER_MESSAGE;
    const maxByInput = Math.max(1, Math.floor(storageLimit / BYTES_PER_MESSAGE));
    const maxByOutput = Math.max(1, Math.floor(storageLimit / BYTES_PER_HASH));

    const supportedWorkgroupSize = this.resolveWorkgroupSizeLimit();
    const maxWorkgroups = Math.max(1, device.limits.maxComputeWorkgroupsPerDimension ?? 65535);
    const workgroupLimit = supportedWorkgroupSize * maxWorkgroups;
    const rawMax = Math.max(1, Math.min(maxByInput, maxByOutput, workgroupLimit));

    // 少し余裕を持たせて境界超過を防ぐ
    const safeMax = rawMax > 1 ? rawMax - 1 : rawMax;
    this.maxMessagesPerDispatch = safeMax;
    return safeMax;
  }

  private resolveWorkgroupSizeLimit(): number {
    const device = this.device!;
    const invocationLimit = device.limits.maxComputeInvocationsPerWorkgroup ?? this.workgroupSize;
    const axisLimit = device.limits.maxComputeWorkgroupSizeX ?? this.workgroupSize;
    if (this.workgroupSize > invocationLimit || this.workgroupSize > axisLimit) {
      throw new Error(
        `workgroup size ${this.workgroupSize} exceeds device limits (max invocations ${invocationLimit}, max x-dimension ${axisLimit}).`
      );
    }
    return this.workgroupSize;
  }

  private scheduleBatch(
    messages: Uint32Array,
    messageCount: number,
    results: Uint32Array,
    resultOffset: number,
    profiling: WebGpuSha1ProfilingAccumulator,
    slot: number,
    bufferSet: BufferSet
  ): Promise<void> {
    const device = this.device!;

    const inputSize = messages.byteLength;
    const outputSize = messageCount * BYTES_PER_HASH;
    const batchStart = performance.now();

    device.pushErrorScope?.('validation');

    device.queue.writeBuffer(bufferSet.input!, 0, messages.buffer, messages.byteOffset, inputSize);
    this.updateDispatchConfig(messageCount);

    const afterUpload = performance.now();

    const bindGroup = this.getOrCreateBindGroup(slot, bufferSet);

    return this.completeBatch({
      pipeline: this.pipeline!,
      bindGroup,
      outputBuffer: bufferSet.output!,
      readbackBuffer: bufferSet.readback!,
      outputSize,
      messageCount,
      results,
      resultOffset,
      profiling,
      slot,
      batchStart,
      afterUpload,
      onValidationError: () => this.releaseBufferSets(),
    });
  }

  private configureGeneratedWorkload(workload: GpuSha1WorkloadConfig): void {
    const signatureParts = [
      workload.startSecondsSince2000,
      workload.rangeSeconds,
      workload.timer0Min,
      workload.timer0Count,
      workload.vcountMin,
      workload.vcountCount,
      workload.macLower,
      workload.data7Swapped,
      workload.keyInputSwapped,
      workload.hardwareType,
      workload.startYear,
      workload.startDayOfYear,
      workload.startSecondOfDay,
      workload.startDayOfWeek,
      workload.nazoSwapped[0],
      workload.nazoSwapped[1],
      workload.nazoSwapped[2],
      workload.nazoSwapped[3],
      workload.nazoSwapped[4],
    ];
    const signature = signatureParts.join('|');
    if (signature === this.generatedConfigSignature) {
      return;
    }

    const data = this.generatedConfigData ?? (this.generatedConfigData = new Uint32Array(24));
    data[0] = 0;
    data[1] = 0;
    data[2] = workload.rangeSeconds >>> 0;
    data[3] = workload.timer0Min >>> 0;
    data[4] = workload.timer0Count >>> 0;
    data[5] = workload.vcountMin >>> 0;
    data[6] = workload.vcountCount >>> 0;
    data[7] = workload.startSecondOfDay >>> 0;
    data[8] = workload.startDayOfWeek >>> 0;
    data[9] = workload.macLower >>> 0;
    data[10] = workload.data7Swapped >>> 0;
    data[11] = workload.keyInputSwapped >>> 0;
    data[12] = workload.hardwareType >>> 0;
    data[13] = workload.nazoSwapped[0] >>> 0;
    data[14] = workload.nazoSwapped[1] >>> 0;
    data[15] = workload.nazoSwapped[2] >>> 0;
    data[16] = workload.nazoSwapped[3] >>> 0;
    data[17] = workload.nazoSwapped[4] >>> 0;
    data[18] = workload.startYear >>> 0;
    data[19] = workload.startDayOfYear >>> 0;
    this.generatedConfigSignature = signature;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;
    if (this.device && this.generatedConfigBuffer) {
      this.device.queue.writeBuffer(
        this.generatedConfigBuffer,
        0,
        data.buffer,
        data.byteOffset,
        data.byteLength
      );
    }
  }

  private scheduleGeneratedBatch(
    messageCount: number,
    baseOffset: number,
    results: Uint32Array,
    resultOffset: number,
    profiling: WebGpuSha1ProfilingAccumulator,
    slot: number,
    bufferSet: GeneratedBufferSet
  ): Promise<void> {
    const outputSize = messageCount * BYTES_PER_HASH;
    const batchStart = performance.now();

    this.device!.pushErrorScope?.('validation');

    this.updateGeneratedDispatchConfig(messageCount, baseOffset);

    const afterUpload = performance.now();

    const bindGroup = this.getOrCreateGeneratedBindGroup(slot, bufferSet);

    return this.completeBatch({
      pipeline: this.generatedPipeline!,
      bindGroup,
      outputBuffer: bufferSet.output!,
      readbackBuffer: bufferSet.readback!,
      outputSize,
      messageCount,
      results,
      resultOffset,
      profiling,
      slot,
      batchStart,
      afterUpload,
      onValidationError: () => this.releaseGeneratedBufferSets(),
    });
  }

  private updateDispatchConfig(messageCount: number): void {
    if (this.lastUploadedConfigCount === messageCount) {
      return;
    }
    const device = this.device!;
    const configData = this.configData ?? (this.configData = new Uint32Array(4));
    configData[0] = messageCount >>> 0;
    configData[1] = 0;
    configData[2] = 0;
    configData[3] = 0;
    device.queue.writeBuffer(this.configBuffer!, 0, configData.buffer, configData.byteOffset, configData.byteLength);
    this.lastUploadedConfigCount = messageCount;
  }

  private updateGeneratedDispatchConfig(messageCount: number, baseOffset: number): void {
    if (
      this.generatedLastUploadedMessageCount === messageCount &&
      this.generatedLastUploadedBaseOffset === baseOffset
    ) {
      return;
    }
    const device = this.device!;
    const data = this.generatedConfigData ?? (this.generatedConfigData = new Uint32Array(24));
    data[0] = messageCount >>> 0;
    data[1] = baseOffset >>> 0;
    device.queue.writeBuffer(this.generatedConfigBuffer!, 0, data.buffer, data.byteOffset, data.byteLength);
    this.generatedLastUploadedMessageCount = messageCount;
    this.generatedLastUploadedBaseOffset = baseOffset;
  }

  private completeBatch(params: {
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    outputBuffer: GPUBuffer;
    readbackBuffer: GPUBuffer;
    outputSize: number;
    messageCount: number;
    results: Uint32Array;
    resultOffset: number;
    profiling: WebGpuSha1ProfilingAccumulator;
    slot: number;
    batchStart: number;
    afterUpload: number;
    onValidationError: () => void;
  }): Promise<void> {
    const device = this.device!;
    const {
      pipeline,
      bindGroup,
      outputBuffer,
      readbackBuffer,
      outputSize,
      messageCount,
      results,
      resultOffset,
      profiling,
      slot,
      batchStart,
      afterUpload,
      onValidationError,
    } = params;

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(messageCount / this.workgroupSize));
    pass.end();

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
    const commandBuffer = commandEncoder.finish();
    const previousGpuDoneAt = this.lastGpuDoneAt;
    device.queue.submit([commandBuffer]);
    const submitTime = performance.now();

    const validationPromise = device.popErrorScope?.() ?? Promise.resolve(null);
    const mapPromise = readbackBuffer.mapAsync(GPUMapMode.READ, 0, outputSize);
    const gpuDonePromise: Promise<number | null> =
      typeof device.queue.onSubmittedWorkDone === 'function'
        ? device.queue
            .onSubmittedWorkDone()
            .then(() => performance.now())
            .catch(() => null)
        : Promise.resolve(null);

    return (async () => {
      await mapPromise;
      const mapResolvedAt = performance.now();
      const gpuDoneAt = await gpuDonePromise;
      const validationError = await validationPromise;
      if (validationError) {
        onValidationError();
        throw validationError;
      }

      const mappedRange = readbackBuffer.getMappedRange(0, outputSize);
      const mappedView = new Uint32Array(mappedRange);
      results.set(mappedView, resultOffset);
      const afterReadback = performance.now();
      readbackBuffer.unmap();

      const uploadMs = afterUpload - batchStart;
      const dispatchMs = mapResolvedAt - afterUpload;
      const readbackMs = afterReadback - mapResolvedAt;
      const totalMs = afterReadback - batchStart;
      const submitToGpuDoneMs = gpuDoneAt === null ? null : Math.max(0, gpuDoneAt - afterUpload);
      const gpuDoneToMapMs = gpuDoneAt === null ? null : Math.max(0, mapResolvedAt - gpuDoneAt);
      const queueSubmitMs = submitTime - batchStart;
      const effectiveGpuDoneAt = gpuDoneAt ?? mapResolvedAt;
      const gpuIdleBeforeMs = previousGpuDoneAt === null ? null : Math.max(0, submitTime - previousGpuDoneAt);

      profiling.uploadMs += uploadMs;
      profiling.dispatchMs += dispatchMs;
      profiling.readbackMs += readbackMs;
      profiling.totalMs += totalMs;
      profiling.batches += 1;
      profiling.maxBatchMessages = Math.max(profiling.maxBatchMessages, messageCount);
      profiling.totalMessages += messageCount;
      profiling.details.push({
        slot,
        messageCount,
        uploadMs,
        dispatchMs,
        readbackMs,
        totalMs,
        submitToGpuDoneMs,
        gpuDoneToMapMs,
        queueSubmitMs,
        gpuIdleBeforeMs,
      });

      this.lastGpuDoneAt = effectiveGpuDoneAt;
    })();
  }

  private initializeBufferSets(): void {
    if (this.bufferSets.length === this.bufferSetCount) {
      return;
    }
    this.bufferSets = Array.from({ length: this.bufferSetCount }, () => this.createEmptyBufferSet());
  }

  private ensureBufferSet(slot: number, messageCount: number): BufferSet {
    const device = this.device;
    if (!device) {
      throw new Error('WebGPU device not initialized');
    }

    if (!this.bufferSets[slot]) {
      this.bufferSets[slot] = this.createEmptyBufferSet();
    }

    const set = this.bufferSets[slot];
    const requiredInputSize = this.alignSize(messageCount * BYTES_PER_MESSAGE);
    if (!set.input || set.inputSize < requiredInputSize) {
      set.input?.destroy();
      set.input = device.createBuffer({
        size: requiredInputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      set.inputSize = requiredInputSize;
      set.bindGroup = null;
    }

    const requiredOutputSize = this.alignSize(messageCount * BYTES_PER_HASH);
    if (!set.output || set.outputSize < requiredOutputSize) {
      set.output?.destroy();
      set.output = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      set.outputSize = requiredOutputSize;
      set.bindGroup = null;
    }

    if (!set.readback || set.readbackSize < requiredOutputSize) {
      set.readback?.destroy();
      set.readback = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      set.readbackSize = requiredOutputSize;
    }

    return set;
  }

  private releaseBufferSets(): void {
    for (const set of this.bufferSets) {
      this.clearBufferSet(set);
    }
  }

  private initializeGeneratedBufferSets(): void {
    if (this.generatedBufferSets.length === this.bufferSetCount) {
      return;
    }
    this.generatedBufferSets = Array.from({ length: this.bufferSetCount }, () => this.createEmptyGeneratedBufferSet());
  }

  private ensureGeneratedBufferSet(slot: number, messageCount: number): GeneratedBufferSet {
    const device = this.device;
    if (!device) {
      throw new Error('WebGPU device not initialized');
    }

    if (!this.generatedBufferSets[slot]) {
      this.generatedBufferSets[slot] = this.createEmptyGeneratedBufferSet();
    }

    const set = this.generatedBufferSets[slot];
    const requiredOutputSize = this.alignSize(messageCount * BYTES_PER_HASH);
    if (!set.output || set.outputSize < requiredOutputSize) {
      set.output?.destroy();
      set.output = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      set.outputSize = requiredOutputSize;
      set.bindGroup = null;
    }

    if (!set.readback || set.readbackSize < requiredOutputSize) {
      set.readback?.destroy();
      set.readback = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      set.readbackSize = requiredOutputSize;
    }

    return set;
  }

  private releaseGeneratedBufferSets(): void {
    for (const set of this.generatedBufferSets) {
      this.clearGeneratedBufferSet(set);
    }
  }

  private alignSize(size: number): number {
    const alignment = 256;
    return Math.ceil(size / alignment) * alignment;
  }

  private getOrCreateGeneratedBindGroup(slot: number, set: GeneratedBufferSet): GPUBindGroup {
    if (set.bindGroup) {
      return set.bindGroup;
    }

    const bindGroup = this.device!.createBindGroup({
      layout: this.generatedBindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.generatedConfigBuffer! } },
        { binding: 1, resource: { buffer: set.output! } },
      ],
    });

    set.bindGroup = bindGroup;
    return bindGroup;
  }

  private getOrCreateBindGroup(slot: number, set: BufferSet): GPUBindGroup {
    if (set.bindGroup) {
      return set.bindGroup;
    }

    const bindGroup = this.device!.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: set.input! } },
        { binding: 1, resource: { buffer: set.output! } },
        { binding: 2, resource: { buffer: this.configBuffer! } },
      ],
    });

    set.bindGroup = bindGroup;
    return bindGroup;
  }

  private createEmptyGeneratedBufferSet(): GeneratedBufferSet {
    return {
      output: null,
      outputSize: 0,
      readback: null,
      readbackSize: 0,
      bindGroup: null,
    };
  }

  private createEmptyBufferSet(): BufferSet {
    return {
      input: null,
      inputSize: 0,
      output: null,
      outputSize: 0,
      readback: null,
      readbackSize: 0,
      bindGroup: null,
    };
  }

  private clearGeneratedBufferSet(set: GeneratedBufferSet | undefined): void {
    if (!set) {
      return;
    }

    set.output?.destroy();
    set.output = null;
    set.outputSize = 0;

    set.readback?.destroy();
    set.readback = null;
    set.readbackSize = 0;

    set.bindGroup = null;
  }

  private clearBufferSet(set: BufferSet | undefined): void {
    if (!set) {
      return;
    }

    set.input?.destroy();
    set.input = null;
    set.inputSize = 0;

    set.output?.destroy();
    set.output = null;
    set.outputSize = 0;

    set.readback?.destroy();
    set.readback = null;
    set.readbackSize = 0;

    set.bindGroup = null;
  }

  private async recreatePipelines(): Promise<void> {
    const device = this.device;
    if (!device) {
      return;
    }

  const module = device.createShaderModule({ code: buildSha1ShaderSource(this.workgroupSize) });
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'sha1_main' },
    });

  const generatedModule = device.createShaderModule({ code: buildGeneratedSha1ShaderSource(this.workgroupSize) });
    const generatedPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: generatedModule, entryPoint: 'sha1_generate' },
    });

    this.pipeline = pipeline;
    this.bindGroupLayout = pipeline.getBindGroupLayout(0);
    this.generatedPipeline = generatedPipeline;
    this.generatedBindGroupLayout = generatedPipeline.getBindGroupLayout(0);
    this.maxMessagesPerDispatch = null;
    this.invalidateBindGroups();
    this.invalidateGeneratedBindGroups();
    this.lastUploadedConfigCount = null;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;
  }

  private invalidateBindGroups(): void {
    for (const set of this.bufferSets) {
      if (set) {
        set.bindGroup = null;
      }
    }
  }

  private invalidateGeneratedBindGroups(): void {
    for (const set of this.generatedBufferSets) {
      if (set) {
        set.bindGroup = null;
      }
    }
  }

  private validateMessageBuffer(messages: Uint32Array): number {
    if (messages.length % WORDS_PER_MESSAGE !== 0) {
      throw new Error('Message buffer must contain complete SHA-1 blocks (16 words each)');
    }
    return messages.length / WORDS_PER_MESSAGE;
  }
}
