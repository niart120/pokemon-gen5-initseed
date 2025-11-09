import { DEFAULT_WORKGROUP_SIZE, DOUBLE_BUFFER_SET_COUNT, MATCH_OUTPUT_HEADER_BYTES, MATCH_RECORD_BYTES } from '../constants';

interface BufferSlot {
  output: GPUBuffer | null;
  readback: GPUBuffer | null;
  candidate: GPUBuffer | null;
  groupCounts: GPUBuffer | null;
  groupOffsets: GPUBuffer | null;
  matchCount: GPUBuffer | null;
  outputSize: number;
  readbackSize: number;
  candidateSize: number;
  groupCountSize: number;
  groupOffsetSize: number;
  matchCountSize: number;
}

export interface BufferPoolOptions {
  slots?: number;
  workgroupSize?: number;
}

export interface BufferSlotHandle {
  output: GPUBuffer;
  readback: GPUBuffer;
  candidate: GPUBuffer;
  groupCounts: GPUBuffer;
  groupOffsets: GPUBuffer;
  matchCount: GPUBuffer;
  outputSize: number;
  candidateCapacity: number;
  groupCount: number;
  maxRecords: number;
}

export interface WebGpuBufferPool {
  readonly slotCount: number;
  acquire(slotIndex: number, messageCount: number): BufferSlotHandle;
  dispose(): void;
}

export function createWebGpuBufferPool(
  device: GPUDevice,
  options?: BufferPoolOptions
): WebGpuBufferPool {
  const slotCount = options?.slots ?? DOUBLE_BUFFER_SET_COUNT;
  const workgroupSize = options?.workgroupSize ?? DEFAULT_WORKGROUP_SIZE;
  if (slotCount <= 0) {
    throw new Error('buffer pool must have at least one slot');
  }

  const slots: BufferSlot[] = Array.from({ length: slotCount }, () => ({
    output: null,
    readback: null,
    candidate: null,
    groupCounts: null,
    groupOffsets: null,
    matchCount: null,
    outputSize: 0,
    readbackSize: 0,
    candidateSize: 0,
    groupCountSize: 0,
    groupOffsetSize: 0,
    matchCountSize: 0,
  }));

  const alignSize = (bytes: number): number => {
    const alignment = 256;
    return Math.ceil(bytes / alignment) * alignment;
  };

  const acquire = (slotIndex: number, messageCount: number): BufferSlotHandle => {
    if (slotIndex < 0 || slotIndex >= slots.length) {
      throw new Error(`buffer slot ${slotIndex} is out of range`);
    }

    if (!Number.isFinite(messageCount) || messageCount <= 0) {
      throw new Error('messageCount must be a positive integer');
    }

    const slot = slots[slotIndex];
    const requiredRecords = messageCount;
    const requiredBytes = alignSize(MATCH_OUTPUT_HEADER_BYTES + requiredRecords * MATCH_RECORD_BYTES);
    const groupCount = Math.max(1, Math.ceil(messageCount / workgroupSize));
    const candidateCapacity = groupCount * workgroupSize;
    const candidateBytes = alignSize(candidateCapacity * MATCH_RECORD_BYTES);
    const groupCountBytes = alignSize(groupCount * Uint32Array.BYTES_PER_ELEMENT);
    const matchHeaderBytes = alignSize(MATCH_OUTPUT_HEADER_BYTES);

    if (!slot.output || requiredBytes > slot.outputSize) {
      slot.output?.destroy();
      slot.output = device.createBuffer({
        label: `gpu-seed-output-${slotIndex}`,
        size: requiredBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      slot.outputSize = requiredBytes;
    }

    if (!slot.readback || requiredBytes > slot.readbackSize) {
      slot.readback?.destroy();
      slot.readback = device.createBuffer({
        label: `gpu-seed-readback-${slotIndex}`,
        size: requiredBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      slot.readbackSize = requiredBytes;
    }

    if (!slot.candidate || candidateBytes > slot.candidateSize) {
      slot.candidate?.destroy();
      slot.candidate = device.createBuffer({
        label: `gpu-seed-candidate-${slotIndex}`,
        size: candidateBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      slot.candidateSize = candidateBytes;
    }

    if (!slot.groupCounts || groupCountBytes > slot.groupCountSize) {
      slot.groupCounts?.destroy();
      slot.groupCounts = device.createBuffer({
        label: `gpu-seed-group-counts-${slotIndex}`,
        size: groupCountBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      slot.groupCountSize = groupCountBytes;
    }

    if (!slot.groupOffsets || groupCountBytes > slot.groupOffsetSize) {
      slot.groupOffsets?.destroy();
      slot.groupOffsets = device.createBuffer({
        label: `gpu-seed-group-offsets-${slotIndex}`,
        size: groupCountBytes,
        usage: GPUBufferUsage.STORAGE,
      });
      slot.groupOffsetSize = groupCountBytes;
    }

    if (!slot.matchCount || matchHeaderBytes > slot.matchCountSize) {
      slot.matchCount?.destroy();
      slot.matchCount = device.createBuffer({
        label: `gpu-seed-match-header-${slotIndex}`,
        size: matchHeaderBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      slot.matchCountSize = matchHeaderBytes;
    }

    return {
      output: slot.output!,
      readback: slot.readback!,
      candidate: slot.candidate!,
      groupCounts: slot.groupCounts!,
      groupOffsets: slot.groupOffsets!,
      matchCount: slot.matchCount!,
      outputSize: slot.outputSize,
      candidateCapacity,
      groupCount,
      maxRecords: requiredRecords,
    };
  };

  const dispose = (): void => {
    for (const slot of slots) {
      slot.output?.destroy();
      slot.readback?.destroy();
      slot.candidate?.destroy();
      slot.groupCounts?.destroy();
      slot.groupOffsets?.destroy();
      slot.matchCount?.destroy();
      slot.output = null;
      slot.readback = null;
      slot.candidate = null;
      slot.groupCounts = null;
      slot.groupOffsets = null;
      slot.matchCount = null;
      slot.outputSize = 0;
      slot.readbackSize = 0;
      slot.candidateSize = 0;
      slot.groupCountSize = 0;
      slot.groupOffsetSize = 0;
      slot.matchCountSize = 0;
    }
  };

  return {
    get slotCount() {
      return slots.length;
    },
    acquire,
    dispose,
  };
}
