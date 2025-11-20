import { MATCH_RECORD_WORDS } from './constants';
import type { SeedSearchJobLimits } from './types';

export interface WebGpuDeviceOptions {
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: GPUDeviceDescriptor['requiredLimits'];
  label?: string;
  powerPreference?: GPUPowerPreference;
}

export interface WebGpuCapabilities {
  limits: GPUSupportedLimits;
  features: ReadonlySet<GPUFeatureName>;
}

export type GpuProfileKind = 'unknown' | 'integrated' | 'mobile' | 'discrete';

export type GpuProfileSource = 'unknown' | 'user-agent' | 'adapter-info' | 'fallback';

export interface GpuProfile {
  kind: GpuProfileKind;
  source: GpuProfileSource;
  userAgent?: string;
  isFallbackAdapter: boolean;
  adapterInfo?: AdapterInfoResult;
}

export interface SeedSearchLimitPreferences {
  workgroupSize?: number;
  maxWorkgroupsPerDispatch?: number;
  maxWorkgroupsPerDispatchY?: number;
  maxMessagesPerDispatch?: number;
  candidateCapacityPerDispatch?: number;
}

export interface WebGpuDeviceContext {
  getAdapter(): GPUAdapter;
  getDevice(): GPUDevice;
  getQueue(): GPUQueue;
  getLimits(): GPUSupportedLimits;
  getCapabilities(): WebGpuCapabilities;
  getGpuProfile(): GpuProfile;
  deriveSearchJobLimits(preferences?: SeedSearchLimitPreferences): SeedSearchJobLimits;
  isLost(): boolean;
  waitForLoss(): Promise<GPUDeviceLostInfo>;
  getSupportedWorkgroupSize(targetSize?: number): number;
}

const DEFAULT_DEVICE_OPTIONS: Required<Pick<WebGpuDeviceOptions, 'requiredFeatures' | 'powerPreference'>> = {
  requiredFeatures: [],
  powerPreference: 'high-performance',
};

const DEFAULT_LIMIT_PREFERENCES: SeedSearchLimitPreferences = {
  workgroupSize: 256,
  maxWorkgroupsPerDispatchY: 256,
  candidateCapacityPerDispatch: 4096,
};

const MATCH_RECORD_BYTES = MATCH_RECORD_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const MAX_U32 = 0xffffffff;

export function isWebGpuSupported(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined';
}

export const isWebGpuSeedSearchSupported = isWebGpuSupported;

export async function createWebGpuDeviceContext(options?: WebGpuDeviceOptions): Promise<WebGpuDeviceContext> {
  if (!isWebGpuSupported()) {
    throw new Error('WebGPU is not available in this environment');
  }

  const gpu = navigator.gpu!;
  const adapter = await gpu.requestAdapter({ powerPreference: options?.powerPreference ?? DEFAULT_DEVICE_OPTIONS.powerPreference });
  if (!adapter) {
    throw new Error('Failed to acquire WebGPU adapter');
  }

  const descriptor: GPUDeviceDescriptor = {
    requiredFeatures: options?.requiredFeatures ?? DEFAULT_DEVICE_OPTIONS.requiredFeatures,
    requiredLimits: options?.requiredLimits,
    label: options?.label ?? 'seed-search-device',
  };

  const [device, gpuProfile] = await Promise.all([adapter.requestDevice(descriptor), detectGpuProfile(adapter)]);
  let deviceLost = false;
  const lostPromise = device.lost.then((info) => {
    deviceLost = true;
    console.warn('[webgpu] device lost:', info.message);
    return info;
  });

  const capabilities = extractCapabilities(adapter, device);
  const deviceLimits = device.limits;

  return {
    getAdapter: () => adapter,
    getDevice: () => device,
    getQueue: () => device.queue,
    getLimits: () => deviceLimits,
    getCapabilities: () => capabilities,
    getGpuProfile: () => gpuProfile,
    deriveSearchJobLimits: (preferences) => deriveSearchJobLimits(capabilities.limits, gpuProfile, preferences),
    isLost: () => deviceLost,
    waitForLoss: () => lostPromise,
    getSupportedWorkgroupSize: (targetSize?: number) => {
      return resolveWorkgroupSize(capabilities.limits, targetSize);
    },
  };
}

function extractCapabilities(adapter: GPUAdapter, device: GPUDevice): WebGpuCapabilities {
  const featureSet = new Set<GPUFeatureName>();
  adapter.features.forEach((feature) => featureSet.add(feature as GPUFeatureName));
  return {
    limits: device.limits,
    features: featureSet,
  };
}

function deriveSearchJobLimits(
  limits: GPUSupportedLimits,
  profile: GpuProfile,
  preferences?: SeedSearchLimitPreferences
): SeedSearchJobLimits {
  const mergedPrefs: SeedSearchLimitPreferences = {
    ...DEFAULT_LIMIT_PREFERENCES,
    ...preferences,
  };
  const prefs = applyProfileOverrides(mergedPrefs, profile);

  const workgroupSize = resolveWorkgroupSize(limits, prefs.workgroupSize);
  const maxWorkgroupsLimit = getLimit(limits.maxComputeWorkgroupsPerDimension);
  const requestedWorkgroupLimitX = prefs.maxWorkgroupsPerDispatch ?? maxWorkgroupsLimit;
  const maxWorkgroupsPerDispatch = clampPositive(
    Math.min(requestedWorkgroupLimitX, maxWorkgroupsLimit),
    'maxWorkgroupsPerDispatch'
  );
  const maxWorkgroupsByMessages = Math.max(
    1,
    Math.floor(MAX_U32 / Math.max(1, workgroupSize * maxWorkgroupsPerDispatch))
  );
  const requestedWorkgroupLimitY = prefs.maxWorkgroupsPerDispatchY ?? maxWorkgroupsLimit;
  const maxWorkgroupsPerDispatchY = clampPositive(
    Math.min(requestedWorkgroupLimitY, maxWorkgroupsLimit, maxWorkgroupsByMessages),
    'maxWorkgroupsPerDispatchY'
  );

  const maxWorkgroupsProduct = Math.max(1, maxWorkgroupsPerDispatch * maxWorkgroupsPerDispatchY);
  const maxMessagesByWorkgroups = workgroupSize * maxWorkgroupsProduct;
  const requestedMessagesLimit = prefs.maxMessagesPerDispatch ?? maxMessagesByWorkgroups;
  const maxMessagesPerDispatch = clampPositive(
    Math.min(requestedMessagesLimit, maxMessagesByWorkgroups),
    'maxMessagesPerDispatch'
  );

  const maxCandidateByStorage = Math.max(
    1,
    Math.floor(getLimit(limits.maxStorageBufferBindingSize) / MATCH_RECORD_BYTES)
  );
  const requestedCandidateCapacity = prefs.candidateCapacityPerDispatch ?? maxCandidateByStorage;
  const candidateCapacityPerDispatch = clampPositive(
    Math.min(requestedCandidateCapacity, maxCandidateByStorage),
    'candidateCapacityPerDispatch'
  );

  return {
    workgroupSize,
    maxWorkgroupsPerDispatch,
    maxWorkgroupsPerDispatchY,
    maxMessagesPerDispatch,
    candidateCapacityPerDispatch,
  };
}

function resolveWorkgroupSize(limits: GPUSupportedLimits, targetSize?: number): number {
  const defaultSize = DEFAULT_LIMIT_PREFERENCES.workgroupSize ?? 128;
  const requested =
    typeof targetSize === 'number' && Number.isFinite(targetSize) && targetSize > 0
      ? Math.floor(targetSize)
      : defaultSize;

  const maxByDimension = getLimit(limits.maxComputeWorkgroupSizeX);
  const maxByInvocations = getLimit(limits.maxComputeInvocationsPerWorkgroup);
  const maxAllowed = Math.max(1, Math.min(maxByDimension, maxByInvocations));
  return Math.max(1, Math.min(requested, maxAllowed));
}

function getLimit(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return Number.MAX_SAFE_INTEGER;
  }
  return Math.floor(value);
}

function clampPositive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number`);
  }
  return Math.floor(value);
}

export type AdapterInfoResult = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

async function detectGpuProfile(adapter: GPUAdapter): Promise<GpuProfile> {
  const adapterWithInfo = adapter as GPUAdapter & {
    requestAdapterInfo?: () => Promise<AdapterInfoResult>;
  };

  const userAgent = getUserAgent();
  const adapterInfo = await tryGetAdapterInfo(adapterWithInfo);
  const fallbackAwareAdapter = adapter as GPUAdapter & { isFallbackAdapter?: boolean };
  const isFallbackAdapter = Boolean(fallbackAwareAdapter.isFallbackAdapter);

  if (isMobileUserAgent(userAgent)) {
    return {
      kind: 'mobile',
      source: 'user-agent',
      userAgent,
      adapterInfo,
      isFallbackAdapter,
    };
  }

  const classifiedKind = classifyAdapterInfo(adapterInfo);
  if (classifiedKind) {
    return {
      kind: classifiedKind,
      source: 'adapter-info',
      userAgent,
      adapterInfo,
      isFallbackAdapter,
    };
  }

  if (isFallbackAdapter) {
    return {
      kind: 'integrated',
      source: 'fallback',
      userAgent,
      adapterInfo,
      isFallbackAdapter,
    };
  }

  return {
    kind: 'unknown',
    source: 'unknown',
    userAgent,
    adapterInfo,
    isFallbackAdapter,
  };
}

async function tryGetAdapterInfo(
  adapter: GPUAdapter & { requestAdapterInfo?: () => Promise<AdapterInfoResult> }
): Promise<AdapterInfoResult | undefined> {
  const requestInfo = adapter.requestAdapterInfo;
  if (typeof requestInfo !== 'function') {
    return undefined;
  }
  try {
    return await requestInfo.call(adapter);
  } catch (error) {
    console.warn('[webgpu] requestAdapterInfo failed:', error);
    return undefined;
  }
}

function getUserAgent(): string {
  if (typeof navigator === 'undefined') {
    return '';
  }
  return navigator.userAgent || '';
}

function isMobileUserAgent(userAgent: string): boolean {
  if (!userAgent) {
    return false;
  }
  return /Android|iPhone|iPad|iPod|Mobile|Silk|Kindle|Opera Mini|Opera Mobi/i.test(userAgent);
}

const MOBILE_GPU_KEYWORDS = ['mali', 'adreno', 'powervr', 'apple gpu', 'apple m', 'snapdragon', 'exynos'];
const INTEGRATED_GPU_KEYWORDS = ['intel', 'iris', 'uhd', 'hd graphics', 'radeon graphics', 'apple'];
const DISCRETE_GPU_KEYWORDS = ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'amd', 'radeon rx', 'radeon pro', 'arc'];

function classifyAdapterInfo(info?: AdapterInfoResult): GpuProfileKind | undefined {
  if (!info) {
    return undefined;
  }
  const searchable = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();

  if (!searchable) {
    return undefined;
  }

  if (includesKeyword(searchable, MOBILE_GPU_KEYWORDS)) {
    return 'mobile';
  }

  if (includesKeyword(searchable, INTEGRATED_GPU_KEYWORDS)) {
    return 'integrated';
  }

  if (includesKeyword(searchable, DISCRETE_GPU_KEYWORDS)) {
    return 'discrete';
  }

  return undefined;
}

function includesKeyword(target: string, keywords: string[]): boolean {
  return keywords.some((keyword) => target.includes(keyword));
}

function applyProfileOverrides(
  prefs: SeedSearchLimitPreferences,
  profile: GpuProfile
): SeedSearchLimitPreferences {
  if (profile.kind === 'mobile' || profile.kind === 'integrated') {
    return {
      ...prefs,
      maxWorkgroupsPerDispatchY: 1,
    };
  }
  return prefs;
}
