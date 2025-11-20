import { MATCH_OUTPUT_HEADER_WORDS, MATCH_RECORD_WORDS } from './constants';
import { createWebGpuDeviceContext, type WebGpuDeviceContext } from './device-context';
import type { SeedSearchJobLimits, SeedSearchJobSegment } from './types';
import { createSeedSearchKernel } from './kernel/seed-search-kernel';

const CONFIG_WORD_COUNT = 33;
const BUFFER_ALIGNMENT = 256;
const ZERO_HEADER = new Uint32Array([0]);

export interface SeedSearchEngineObserver {
  onEnsureConfigured?: (payload: {
    limits: SeedSearchJobLimits;
    workgroupSize: number;
    candidateCapacity: number;
    pipelineRecreated: boolean;
    timestampMs: number;
  }) => void;
  onBufferRecreated?: (payload: {
    kind: 'config' | 'match-output' | 'match-readback' | 'target';
    sizeBytes: number;
    timestampMs: number;
  }) => void;
  onDispatchComplete?: (payload: {
    segmentId: string;
    messageCount: number;
    workgroupCount: number;
    workgroupCountX?: number;
    workgroupCountY?: number;
    matchCount: number;
    candidateCapacity: number;
    timings: {
      totalMs: number;
      setupMs: number;
      gpuMs: number;
      readbackMs: number;
    };
    timestampMs: number;
  }) => void;
}

export interface SeedSearchEngineResult {
  words: Uint32Array;
  matchCount: number;
}

export interface SeedSearchEngine {
  ensureConfigured(limits: SeedSearchJobLimits): Promise<void>;
  setTargetSeeds(targetSeeds: Uint32Array): void;
  executeSegment(segment: SeedSearchJobSegment): Promise<SeedSearchEngineResult>;
  dispose(): void;
  getWorkgroupSize(): number;
  getCandidateCapacity(): number;
}

interface EngineState {
  context: WebGpuDeviceContext | null;
  pipeline: GPUComputePipeline | null;
  bindGroupLayout: GPUBindGroupLayout | null;
  configBuffer: GPUBuffer | null;
  configData: Uint32Array | null;
  matchOutputBuffer: GPUBuffer | null;
  readbackBuffer: GPUBuffer | null;
  matchBufferSize: number;
  targetBuffer: GPUBuffer | null;
  targetCapacity: number;
  workgroupSize: number;
  candidateCapacity: number;
  currentLimits: SeedSearchJobLimits | null;
}

export function createSeedSearchEngine(
  observer?: SeedSearchEngineObserver,
  initialContext?: WebGpuDeviceContext
): SeedSearchEngine {
  const state: EngineState = {
    context: initialContext ?? null,
    pipeline: null,
    bindGroupLayout: null,
    configBuffer: null,
    configData: null,
    matchOutputBuffer: null,
    readbackBuffer: null,
    matchBufferSize: 0,
    targetBuffer: null,
    targetCapacity: 0,
    workgroupSize: 0,
    candidateCapacity: 0,
    currentLimits: null,
  };

  const ensureConfigured = async (limits: SeedSearchJobLimits): Promise<void> => {
    if (!state.context) {
      state.context = await createWebGpuDeviceContext();
    }

    const device = state.context.getDevice();
    const resolvedWorkgroupSize = state.context.getSupportedWorkgroupSize(limits.workgroupSize);
    const limitsChanged =
      !state.currentLimits ||
      state.workgroupSize !== resolvedWorkgroupSize ||
      state.candidateCapacity !== limits.candidateCapacityPerDispatch;

    const shouldRecreatePipeline = !state.pipeline || limitsChanged;

    if (shouldRecreatePipeline) {
      const { pipeline, layout } = createSeedSearchKernel(device, resolvedWorkgroupSize);
      state.pipeline = pipeline;
      state.bindGroupLayout = layout;
      recreateConfigBuffer(device);
      recreateMatchBuffers(device, limits.candidateCapacityPerDispatch);
      state.workgroupSize = resolvedWorkgroupSize;
      state.candidateCapacity = limits.candidateCapacityPerDispatch;
      state.currentLimits = limits;
      return;
    }

    if (!state.matchOutputBuffer || !state.readbackBuffer) {
      recreateMatchBuffers(device, limits.candidateCapacityPerDispatch);
    }

    if (!state.configBuffer || !state.configData) {
      recreateConfigBuffer(device);
    }

    state.currentLimits = limits;

    observer?.onEnsureConfigured?.({
      limits,
      workgroupSize: state.workgroupSize,
      candidateCapacity: state.candidateCapacity,
      pipelineRecreated: shouldRecreatePipeline,
      timestampMs: nowMs(),
    });
  };

  const recreateConfigBuffer = (device: GPUDevice): void => {
    const configData = new Uint32Array(CONFIG_WORD_COUNT);
    const size = alignSize(configData.byteLength);
    state.configBuffer?.destroy();
    state.configBuffer = device.createBuffer({
      label: 'seed-search-config',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    state.configData = configData;
    observer?.onBufferRecreated?.({ kind: 'config', sizeBytes: size, timestampMs: nowMs() });
  };

  const recreateMatchBuffers = (device: GPUDevice, candidateCapacity: number): void => {
    const words = MATCH_OUTPUT_HEADER_WORDS + candidateCapacity * MATCH_RECORD_WORDS;
    const bytes = alignSize(words * Uint32Array.BYTES_PER_ELEMENT);
    state.matchOutputBuffer?.destroy();
    state.matchOutputBuffer = device.createBuffer({
      label: 'seed-search-output',
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    state.readbackBuffer?.destroy();
    state.readbackBuffer = device.createBuffer({
      label: 'seed-search-readback',
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    state.matchBufferSize = bytes;
    observer?.onBufferRecreated?.({ kind: 'match-output', sizeBytes: bytes, timestampMs: nowMs() });
    observer?.onBufferRecreated?.({ kind: 'match-readback', sizeBytes: bytes, timestampMs: nowMs() });
  };

  const setTargetSeeds = (targetSeeds: Uint32Array): void => {
    if (!state.context) {
      throw new Error('SeedSearchEngine is not configured yet');
    }

    const device = state.context.getDevice();
    const seedCount = targetSeeds.length;
    const requiredWords = 1 + seedCount;
    const requiredBytes = alignSize(requiredWords * Uint32Array.BYTES_PER_ELEMENT);

    if (!state.targetBuffer || state.targetCapacity < seedCount) {
      state.targetBuffer?.destroy();
      state.targetBuffer = device.createBuffer({
        label: 'seed-search-target-seeds',
        size: requiredBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      state.targetCapacity = seedCount;
      observer?.onBufferRecreated?.({ kind: 'target', sizeBytes: requiredBytes, timestampMs: nowMs() });
    }

    const upload = new Uint32Array(requiredWords);
    upload[0] = seedCount >>> 0;
    for (let i = 0; i < seedCount; i += 1) {
      upload[1 + i] = targetSeeds[i]! >>> 0;
    }

    device.queue.writeBuffer(state.targetBuffer!, 0, upload.buffer, upload.byteOffset, upload.byteLength);
  };

  const executeSegment = async (segment: SeedSearchJobSegment): Promise<SeedSearchEngineResult> => {
    if (!state.context || !state.pipeline || !state.bindGroupLayout) {
      throw new Error('SeedSearchEngine is not ready');
    }

    if (!state.configBuffer || !state.configData || !state.matchOutputBuffer || !state.readbackBuffer) {
      throw new Error('SeedSearchEngine buffers are not ready');
    }

    if (!state.targetBuffer) {
      throw new Error('Target seed buffer is not prepared');
    }

    const device = state.context.getDevice();
    const queue = device.queue;
    const workgroupCountX = Math.max(1, segment.workgroupCountX ?? 1);
    const workgroupCountY = Math.max(1, segment.workgroupCountY ?? 1);
    const totalWorkgroups = workgroupCountX * workgroupCountY;
    const dispatchStart = nowMs();

    queue.writeBuffer(
      state.matchOutputBuffer,
      0,
      ZERO_HEADER.buffer,
      ZERO_HEADER.byteOffset,
      ZERO_HEADER.byteLength
    );

    const configData = state.configData;
    configData.set(segment.configWords);
    configData[0] = segment.messageCount >>> 0;
    configData[22] = workgroupCountX >>> 0;
    configData[23] = state.workgroupSize >>> 0;
    configData[24] = state.candidateCapacity >>> 0;
    configData[32] = workgroupCountY >>> 0;

    queue.writeBuffer(
      state.configBuffer,
      0,
      configData.buffer,
      configData.byteOffset,
      configData.byteLength
    );
    const setupEnd = nowMs();

    const bindGroup = device.createBindGroup({
      label: `seed-search-bind-group-${segment.id}`,
      layout: state.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: state.configBuffer } },
        { binding: 1, resource: { buffer: state.targetBuffer } },
        { binding: 2, resource: { buffer: state.matchOutputBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: `seed-search-encoder-${segment.id}` });
    const pass = encoder.beginComputePass({ label: `seed-search-pass-${segment.id}` });
    pass.setPipeline(state.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    pass.end();

    encoder.copyBufferToBuffer(
      state.matchOutputBuffer,
      0,
      state.readbackBuffer,
      0,
      state.matchBufferSize
    );

    const commandBuffer = encoder.finish();
    queue.submit([commandBuffer]);
    await queue.onSubmittedWorkDone();
    const gpuEnd = nowMs();

    await state.readbackBuffer.mapAsync(GPUMapMode.READ, 0, state.matchBufferSize);
    const mapped = state.readbackBuffer.getMappedRange(0, state.matchBufferSize);
    const words = new Uint32Array(mapped.slice(0));
    state.readbackBuffer.unmap();
    const readbackEnd = nowMs();

    const rawMatchCount = words[0] ?? 0;
    const clampedMatchCount = Math.min(rawMatchCount, state.candidateCapacity);
    const totalWords = Math.min(
      words.length,
      MATCH_OUTPUT_HEADER_WORDS + clampedMatchCount * MATCH_RECORD_WORDS
    );

    const result = {
      words: words.slice(0, totalWords),
      matchCount: clampedMatchCount,
    };

    observer?.onDispatchComplete?.({
      segmentId: segment.id,
      messageCount: segment.messageCount,
      workgroupCount: totalWorkgroups,
      workgroupCountX,
      workgroupCountY,
      matchCount: clampedMatchCount,
      candidateCapacity: state.candidateCapacity,
      timings: {
        totalMs: readbackEnd - dispatchStart,
        setupMs: setupEnd - dispatchStart,
        gpuMs: gpuEnd - setupEnd,
        readbackMs: readbackEnd - gpuEnd,
      },
      timestampMs: readbackEnd,
    });

    return result;
  };

  const dispose = (): void => {
    state.configBuffer?.destroy();
    state.matchOutputBuffer?.destroy();
    state.readbackBuffer?.destroy();
    state.targetBuffer?.destroy();
    state.context = null;
    state.pipeline = null;
    state.bindGroupLayout = null;
    state.configBuffer = null;
    state.configData = null;
    state.matchOutputBuffer = null;
    state.readbackBuffer = null;
    state.targetBuffer = null;
    state.targetCapacity = 0;
    state.currentLimits = null;
  };

  const getWorkgroupSize = (): number => state.workgroupSize;
  const getCandidateCapacity = (): number => state.candidateCapacity;

  return {
    ensureConfigured,
    setTargetSeeds,
    executeSegment,
    dispose,
    getWorkgroupSize,
    getCandidateCapacity,
  };
}

function alignSize(bytes: number): number {
  return Math.ceil(bytes / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
