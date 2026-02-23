import {
  MATCH_OUTPUT_HEADER_WORDS,
  MATCH_RECORD_WORDS,
  createWebGpuDeviceContext,
  type WebGpuDeviceContext,
  type SeedSearchJobLimits,
} from '@/lib/webgpu/utils';
import type { SeedSearchJobSegment } from './types';
import { createSeedSearchKernel } from '@/lib/webgpu/kernel';

const DISPATCH_STATE_WORD_COUNT = 4;
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
    kind: 'config' | 'uniform' | 'match-output' | 'match-readback' | 'target';
    sizeBytes: number;
    timestampMs: number;
  }) => void;
  onDispatchComplete?: (payload: {
    segmentId: string;
    messageCount: number;
    workgroupCount: number;
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

export interface SeedSearchEngineConfigureOptions {
  dispatchSlots?: number;
}

export interface SeedSearchEngine {
  ensureConfigured(limits: SeedSearchJobLimits, options?: SeedSearchEngineConfigureOptions): Promise<void>;
  setTargetSeeds(targetSeeds: Uint32Array): void;
  executeSegment(segment: SeedSearchJobSegment): Promise<SeedSearchEngineResult>;
  dispose(): void;
  getWorkgroupSize(): number;
  getCandidateCapacity(): number;
  getSupportedLimits(): GPUSupportedLimits | null;
}

interface EngineState {
  context: WebGpuDeviceContext | null;
  pipeline: GPUComputePipeline | null;
  bindGroupLayout: GPUBindGroupLayout | null;
  targetBuffer: GPUBuffer | null;
  targetCapacity: number;
  workgroupSize: number;
  candidateCapacity: number;
  currentLimits: SeedSearchJobLimits | null;
  dispatchSlots: DispatchSlot[];
  availableSlots: DispatchSlot[];
  slotWaiters: Array<(slot: DispatchSlot) => void>;
  desiredDispatchSlots: number;
}

interface DispatchSlot {
  id: number;
  dispatchStateBuffer: GPUBuffer;
  dispatchStateData: Uint32Array;
  uniformBuffer: GPUBuffer | null;
  uniformCapacityWords: number;
  matchOutputBuffer: GPUBuffer;
  readbackBuffer: GPUBuffer;
  matchBufferSize: number;
}

export function createSeedSearchEngine(
  observer?: SeedSearchEngineObserver,
  initialContext?: WebGpuDeviceContext
): SeedSearchEngine {
  const state: EngineState = {
    context: initialContext ?? null,
    pipeline: null,
    bindGroupLayout: null,
    targetBuffer: null,
    targetCapacity: 0,
    workgroupSize: 0,
    candidateCapacity: 0,
    currentLimits: null,
    dispatchSlots: [],
    availableSlots: [],
    slotWaiters: [],
    desiredDispatchSlots: 1,
  };

  const ensureConfigured = async (
    limits: SeedSearchJobLimits,
    options?: SeedSearchEngineConfigureOptions
  ): Promise<void> => {
    if (!state.context) {
      state.context = await createWebGpuDeviceContext();
    }

    const device = state.context.getDevice();
    const resolvedWorkgroupSize = state.context.getSupportedWorkgroupSize(limits.workgroupSize);
    const desiredSlots = Math.max(1, options?.dispatchSlots ?? state.desiredDispatchSlots ?? 1);
    const limitsChanged =
      !state.currentLimits ||
      state.workgroupSize !== resolvedWorkgroupSize ||
      state.candidateCapacity !== limits.candidateCapacityPerDispatch;

    const shouldRecreatePipeline = !state.pipeline || limitsChanged;

    if (shouldRecreatePipeline) {
      const { pipeline, layout } = createSeedSearchKernel(device, resolvedWorkgroupSize);
      state.pipeline = pipeline;
      state.bindGroupLayout = layout;
    }

    state.workgroupSize = resolvedWorkgroupSize;
    state.candidateCapacity = limits.candidateCapacityPerDispatch;
    state.currentLimits = limits;
    state.desiredDispatchSlots = desiredSlots;
    syncDispatchSlots(device, desiredSlots, limits.candidateCapacityPerDispatch);

    state.currentLimits = limits;

    observer?.onEnsureConfigured?.({
      limits,
      workgroupSize: state.workgroupSize,
      candidateCapacity: state.candidateCapacity,
      pipelineRecreated: shouldRecreatePipeline,
      timestampMs: nowMs(),
    });
  };

  const syncDispatchSlots = (device: GPUDevice, desiredSlots: number, candidateCapacity: number): void => {
    for (const slot of state.dispatchSlots) {
      ensureSlotMatchBuffers(device, slot, candidateCapacity);
    }

    while (state.dispatchSlots.length < desiredSlots) {
      const slotId = state.dispatchSlots.length;
      const slot = createDispatchSlot(device, slotId, candidateCapacity);
      state.dispatchSlots.push(slot);
    }

    while (state.dispatchSlots.length > desiredSlots) {
      const removed = state.dispatchSlots.pop();
      if (removed) {
        destroyDispatchSlot(removed);
      }
    }

    state.availableSlots = [...state.dispatchSlots];
    state.slotWaiters.length = 0;
  };

  const createDispatchSlot = (
    device: GPUDevice,
    slotId: number,
    candidateCapacity: number
  ): DispatchSlot => {
    const dispatchStateData = new Uint32Array(DISPATCH_STATE_WORD_COUNT);
    const dispatchStateSize = alignSize(dispatchStateData.byteLength);
    const dispatchStateBuffer = device.createBuffer({
      label: `seed-search-dispatch-state-${slotId}`,
      size: dispatchStateSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    observer?.onBufferRecreated?.({ kind: 'config', sizeBytes: dispatchStateSize, timestampMs: nowMs() });

    const { matchOutputBuffer, readbackBuffer, matchBufferSize } = createMatchBuffers(
      device,
      candidateCapacity,
      slotId
    );

    return {
      id: slotId,
      dispatchStateBuffer,
      dispatchStateData,
      uniformBuffer: null,
      uniformCapacityWords: 0,
      matchOutputBuffer,
      readbackBuffer,
      matchBufferSize,
    };
  };

  const ensureSlotUniformCapacity = (
    device: GPUDevice,
    slot: DispatchSlot,
    words: number
  ): void => {
    const requiredBytes = alignSize(words * Uint32Array.BYTES_PER_ELEMENT);
    if (!slot.uniformBuffer || slot.uniformCapacityWords < words) {
      slot.uniformBuffer?.destroy();
      slot.uniformBuffer = device.createBuffer({
        label: `seed-search-uniform-${slot.id}`,
        size: requiredBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      slot.uniformCapacityWords = words;
      observer?.onBufferRecreated?.({ kind: 'uniform', sizeBytes: requiredBytes, timestampMs: nowMs() });
    }
  };

  const createMatchBuffers = (
    device: GPUDevice,
    candidateCapacity: number,
    slotId: number
  ): { matchOutputBuffer: GPUBuffer; readbackBuffer: GPUBuffer; matchBufferSize: number } => {
    const words = MATCH_OUTPUT_HEADER_WORDS + candidateCapacity * MATCH_RECORD_WORDS;
    const bytes = alignSize(words * Uint32Array.BYTES_PER_ELEMENT);
    const matchOutputBuffer = device.createBuffer({
      label: `seed-search-output-${slotId}`,
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const readbackBuffer = device.createBuffer({
      label: `seed-search-readback-${slotId}`,
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    observer?.onBufferRecreated?.({ kind: 'match-output', sizeBytes: bytes, timestampMs: nowMs() });
    observer?.onBufferRecreated?.({ kind: 'match-readback', sizeBytes: bytes, timestampMs: nowMs() });
    return { matchOutputBuffer, readbackBuffer, matchBufferSize: bytes };
  };

  const ensureSlotMatchBuffers = (
    device: GPUDevice,
    slot: DispatchSlot,
    candidateCapacity: number
  ): void => {
    const words = MATCH_OUTPUT_HEADER_WORDS + candidateCapacity * MATCH_RECORD_WORDS;
    const bytes = alignSize(words * Uint32Array.BYTES_PER_ELEMENT);
    if (slot.matchBufferSize === bytes) {
      return;
    }
    slot.matchOutputBuffer.destroy();
    slot.readbackBuffer.destroy();
    const buffers = createMatchBuffers(device, candidateCapacity, slot.id);
    slot.matchOutputBuffer = buffers.matchOutputBuffer;
    slot.readbackBuffer = buffers.readbackBuffer;
    slot.matchBufferSize = buffers.matchBufferSize;
  };

  const destroyDispatchSlot = (slot: DispatchSlot): void => {
    slot.dispatchStateBuffer.destroy();
    slot.uniformBuffer?.destroy();
    slot.matchOutputBuffer.destroy();
    slot.readbackBuffer.destroy();
  };

  const acquireSlot = (): Promise<DispatchSlot> => {
    if (state.availableSlots.length > 0) {
      return Promise.resolve(state.availableSlots.pop()!);
    }
    return new Promise<DispatchSlot>((resolve) => {
      state.slotWaiters.push(resolve);
    });
  };

  const releaseSlot = (slot: DispatchSlot): void => {
    const waiter = state.slotWaiters.shift();
    if (waiter) {
      waiter(slot);
      return;
    }
    state.availableSlots.push(slot);
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

    if (!state.targetBuffer) {
      throw new Error('Target seed buffer is not prepared');
    }

    if (state.dispatchSlots.length === 0) {
      throw new Error('Dispatch slots are not configured');
    }

    const device = state.context.getDevice();
    const queue = device.queue;
    const workgroupCount = Math.max(1, segment.workgroupCount);
    const totalWorkgroups = workgroupCount;
    const slot = await acquireSlot();

    try {
      const dispatchStart = nowMs();

      queue.writeBuffer(
        slot.matchOutputBuffer,
        0,
        ZERO_HEADER.buffer,
        ZERO_HEADER.byteOffset,
        ZERO_HEADER.byteLength
      );

      const dispatchStateData = slot.dispatchStateData;
      dispatchStateData[0] = segment.messageCount >>> 0;
      dispatchStateData[1] = segment.baseSecondOffset >>> 0;
      dispatchStateData[2] = state.candidateCapacity >>> 0;
      dispatchStateData[3] = 0;

      queue.writeBuffer(
        slot.dispatchStateBuffer,
        0,
        dispatchStateData.buffer,
        dispatchStateData.byteOffset,
        dispatchStateData.byteLength
      );

      const uniformWords = segment.getUniformWords();
      ensureSlotUniformCapacity(device, slot, uniformWords.length);
      queue.writeBuffer(
        slot.uniformBuffer!,
        0,
        uniformWords.buffer,
        uniformWords.byteOffset,
        uniformWords.byteLength
      );
      const setupEnd = nowMs();

      const bindGroup = device.createBindGroup({
        label: `seed-search-bind-group-${segment.id}-slot-${slot.id}`,
        layout: state.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: slot.dispatchStateBuffer } },
          { binding: 1, resource: { buffer: slot.uniformBuffer! } },
          { binding: 2, resource: { buffer: state.targetBuffer } },
          { binding: 3, resource: { buffer: slot.matchOutputBuffer } },
        ],
      });

      const encoder = device.createCommandEncoder({ label: `seed-search-encoder-${segment.id}` });
      const pass = encoder.beginComputePass({ label: `seed-search-pass-${segment.id}` });
      pass.setPipeline(state.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();

      encoder.copyBufferToBuffer(
        slot.matchOutputBuffer,
        0,
        slot.readbackBuffer,
        0,
        slot.matchBufferSize
      );

      const commandBuffer = encoder.finish();
      queue.submit([commandBuffer]);

      await slot.readbackBuffer.mapAsync(GPUMapMode.READ, 0, slot.matchBufferSize);
      const gpuEnd = nowMs();
      const mapped = slot.readbackBuffer.getMappedRange(0, slot.matchBufferSize);
      const words = new Uint32Array(mapped.slice(0));
      slot.readbackBuffer.unmap();
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
    } finally {
      releaseSlot(slot);
    }
  };

  const dispose = (): void => {
    for (const slot of state.dispatchSlots) {
      destroyDispatchSlot(slot);
    }
    state.dispatchSlots = [];
    state.availableSlots = [];
    state.slotWaiters.length = 0;
    state.targetBuffer?.destroy();
    state.context = null;
    state.pipeline = null;
    state.bindGroupLayout = null;
    state.targetBuffer = null;
    state.targetCapacity = 0;
    state.currentLimits = null;
  };

  const getWorkgroupSize = (): number => state.workgroupSize;
  const getCandidateCapacity = (): number => state.candidateCapacity;
  const getSupportedLimits = (): GPUSupportedLimits | null => state.context?.getLimits() ?? null;

  return {
    ensureConfigured,
    setTargetSeeds,
    executeSegment,
    dispose,
    getWorkgroupSize,
    getCandidateCapacity,
    getSupportedLimits,
  };
}

function alignSize(bytes: number): number {
  return Math.ceil(bytes / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
