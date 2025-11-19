import { MATCH_RECORD_BYTES } from './constants';

type NavigatorWithDeviceMemory = Navigator & { deviceMemory?: number };
type PerformanceWithMemory = Performance & {
  memory?: {
    jsHeapSizeLimit?: number;
  };
};

const BYTES_PER_GIB = 1024 * 1024 * 1024;
const BYTES_PER_MIB = 1024 * 1024;
const HOST_SHARE_OF_DEVICE_MEMORY = 0.25;
const HOST_SHARE_OF_HEAP_LIMIT = 0.5;
const FALLBACK_MEMORY_PER_CORE_MIB = 64;

const isPositiveFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

const getDeviceMemoryBytes = (): number | null => {
  if (typeof navigator === 'undefined') {
    return null;
  }
  const withDeviceMemory = navigator as NavigatorWithDeviceMemory;
  if (!isPositiveFiniteNumber(withDeviceMemory.deviceMemory)) {
    return null;
  }
  return Math.floor(withDeviceMemory.deviceMemory * BYTES_PER_GIB);
};

const getHeapLimitBytes = (): number | null => {
  if (typeof performance === 'undefined') {
    return null;
  }
  const perfWithMemory = performance as PerformanceWithMemory;
  if (!isPositiveFiniteNumber(perfWithMemory.memory?.jsHeapSizeLimit)) {
    return null;
  }
  return Math.floor(perfWithMemory.memory!.jsHeapSizeLimit!);
};

const getHardwareConcurrency = (): number => {
  if (typeof navigator === 'undefined' || !isPositiveFiniteNumber(navigator.hardwareConcurrency)) {
    return 4;
  }
  return Math.max(1, Math.floor(navigator.hardwareConcurrency));
};

/**
 * Estimate how much host memory (in bytes) we can safely dedicate to WebGPU readback buffers.
 * Prefer device exposure (navigator.deviceMemory) or heap limits when available, and fall back to
 * a concurrency-based heuristic in other environments.
 */
export const estimateHostMemoryLimitBytes = (): number => {
  const fromDeviceMemory = getDeviceMemoryBytes();
  if (fromDeviceMemory !== null) {
    const derived = Math.floor(fromDeviceMemory * HOST_SHARE_OF_DEVICE_MEMORY);
    return Math.max(MATCH_RECORD_BYTES, derived);
  }

  const fromHeapLimit = getHeapLimitBytes();
  if (fromHeapLimit !== null) {
    const derived = Math.floor(fromHeapLimit * HOST_SHARE_OF_HEAP_LIMIT);
    return Math.max(MATCH_RECORD_BYTES, derived);
  }

  const fallback = getHardwareConcurrency() * FALLBACK_MEMORY_PER_CORE_MIB * BYTES_PER_MIB;
  return Math.max(MATCH_RECORD_BYTES, fallback);
};
