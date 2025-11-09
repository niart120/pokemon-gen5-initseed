import { DOUBLE_BUFFER_SET_COUNT, MATCH_OUTPUT_HEADER_BYTES, MATCH_RECORD_BYTES } from '../constants';

interface BufferSlot {
  output: GPUBuffer | null;
  readback: GPUBuffer | null;
  outputSize: number;
  readbackSize: number;
}

export interface BufferPoolOptions {
  slots?: number;
}

export interface BufferSlotHandle {
  output: GPUBuffer;
  readback: GPUBuffer;
  outputSize: number;
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
  if (slotCount <= 0) {
    throw new Error('buffer pool must have at least one slot');
  }

  const slots: BufferSlot[] = Array.from({ length: slotCount }, () => ({
    output: null,
    readback: null,
    outputSize: 0,
    readbackSize: 0,
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

    return {
      output: slot.output!,
      readback: slot.readback!,
      outputSize: slot.outputSize,
      maxRecords: requiredRecords,
    };
  };

  const dispose = (): void => {
    for (const slot of slots) {
      slot.output?.destroy();
      slot.readback?.destroy();
      slot.output = null;
      slot.readback = null;
      slot.outputSize = 0;
      slot.readbackSize = 0;
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
