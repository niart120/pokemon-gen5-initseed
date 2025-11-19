export interface WebGpuDeviceOptions {
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: GPUDeviceDescriptor['requiredLimits'];
  label?: string;
}

export interface WebGpuDeviceContext {
  getAdapter(): GPUAdapter;
  getDevice(): GPUDevice;
  getQueue(): GPUQueue;
  getLimits(): GPUSupportedLimits;
  isLost(): boolean;
  waitForLoss(): Promise<GPUDeviceLostInfo>;
  getSupportedWorkgroupSize(targetSize?: number): number;
}

const DEFAULT_DEVICE_OPTIONS: Required<Pick<WebGpuDeviceOptions, 'requiredFeatures'>> = {
  requiredFeatures: [],
};

export function isWebGpuSupported(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined';
}

export async function createWebGpuDeviceContext(options?: WebGpuDeviceOptions): Promise<WebGpuDeviceContext> {
  if (!isWebGpuSupported()) {
    throw new Error('WebGPU is not available in this environment');
  }

  const gpu = navigator.gpu!;
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('Failed to acquire WebGPU adapter');
  }

  const descriptor: GPUDeviceDescriptor = {
    requiredFeatures: options?.requiredFeatures ?? DEFAULT_DEVICE_OPTIONS.requiredFeatures,
    requiredLimits: options?.requiredLimits,
    label: options?.label ?? 'seed-search-device',
  };

  const device = await adapter.requestDevice(descriptor);
  let deviceLost = false;
  const lostPromise = device.lost.then((info) => {
    deviceLost = true;
    console.warn('[webgpu] device lost:', info.message);
    return info;
  });

  return {
    getAdapter: () => adapter,
    getDevice: () => device,
    getQueue: () => device.queue,
    getLimits: () => device.limits,
    isLost: () => deviceLost,
    waitForLoss: () => lostPromise,
    getSupportedWorkgroupSize: (targetSize?: number) => {
      const limits = device.limits;
      const toPositiveLimit = (value: number | undefined): number => {
        if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
          return Number.POSITIVE_INFINITY;
        }
        return Math.floor(value);
      };

      const invocationLimit = toPositiveLimit(limits.maxComputeInvocationsPerWorkgroup);
      const dimensionLimit = toPositiveLimit(limits.maxComputeWorkgroupSizeX);
      const defaultTarget = Math.min(invocationLimit, dimensionLimit);
      const fallbackTarget = Number.isFinite(defaultTarget) && defaultTarget > 0 ? defaultTarget : 1;

      const requestedSize = typeof targetSize === 'number' && Number.isFinite(targetSize) && targetSize > 0
        ? Math.floor(targetSize)
        : fallbackTarget;

      const resolved = Math.min(requestedSize, fallbackTarget);
      if (resolved <= 0) {
        throw new Error('WebGPU workgroup size limits are invalid');
      }
      return resolved;
    },
  };
}
