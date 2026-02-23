/**
 * WebGPUデバイスコンテキスト
 * 
 * GPUデバイスの取得・能力検出・制限値計算を担当する共通ユーティリティ
 */

import { MATCH_RECORD_WORDS } from './constants';
import type {
  SeedSearchJobLimits,
  SeedSearchLimitPreferences,
  WebGpuDeviceContext,
  WebGpuDeviceOptions,
  WebGpuCapabilities,
  GpuProfile,
  GpuProfileKind,
  AdapterInfoResult,
} from './types';

// === 定数 ===

const DEFAULT_DEVICE_OPTIONS: Required<Pick<WebGpuDeviceOptions, 'requiredFeatures' | 'powerPreference'>> = {
  requiredFeatures: [],
  powerPreference: 'high-performance',
};

const DEFAULT_LIMIT_PREFERENCES: SeedSearchLimitPreferences = {
  workgroupSize: 256,
  candidateCapacityPerDispatch: 4096,
};

const MATCH_RECORD_BYTES = MATCH_RECORD_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const MAX_U32 = 0xffffffff;

const DEFAULT_DISPATCH_SLOTS_BY_PROFILE: Record<GpuProfileKind, number> = {
  mobile: 1,
  integrated: 2,
  discrete: 4,
  unknown: 1,
};
const FALLBACK_DISPATCH_SLOTS = 1;
const MAX_AUTOMATIC_DISPATCH_SLOTS = 8;

// === エクスポート関数 ===

/**
 * WebGPUがサポートされているか確認
 */
export function isWebGpuSupported(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined';
}

/**
 * 後方互換性のためのエイリアス
 */
export const isWebGpuSeedSearchSupported = isWebGpuSupported;

/**
 * WebGPUデバイスコンテキストを作成
 */
export async function createWebGpuDeviceContext(options?: WebGpuDeviceOptions): Promise<WebGpuDeviceContext> {
  if (!isWebGpuSupported()) {
    throw new Error('WebGPU is not available in this environment');
  }

  const gpu = navigator.gpu!;
  const adapter = await gpu.requestAdapter({
    powerPreference: options?.powerPreference ?? DEFAULT_DEVICE_OPTIONS.powerPreference,
  });
  if (!adapter) {
    throw new Error('Failed to acquire WebGPU adapter');
  }

  const descriptor: GPUDeviceDescriptor = {
    requiredFeatures: options?.requiredFeatures ?? DEFAULT_DEVICE_OPTIONS.requiredFeatures,
    requiredLimits: options?.requiredLimits,
    label: options?.label ?? 'seed-search-device',
  };

  const [device, gpuProfile] = await Promise.all([
    adapter.requestDevice(descriptor),
    detectGpuProfile(adapter),
  ]);

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

/**
 * デバイスの制限値から検索ジョブの制限値を導出
 * 外部から直接呼び出す場合用
 */
export function deriveSearchJobLimitsFromDevice(
  device: GPUDevice,
  profile?: GpuProfile,
  preferences?: SeedSearchLimitPreferences
): SeedSearchJobLimits {
  const effectiveProfile: GpuProfile = profile ?? {
    kind: 'unknown',
    source: 'unknown',
    isFallbackAdapter: false,
  };
  return deriveSearchJobLimits(device.limits, effectiveProfile, preferences);
}

// === 内部関数 ===

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
  const maxWorkgroupsByMessages = Math.max(1, Math.floor(MAX_U32 / Math.max(1, workgroupSize)));
  const maxWorkgroupsPerDispatch = clampPositive(
    Math.min(requestedWorkgroupLimitX, maxWorkgroupsLimit, maxWorkgroupsByMessages),
    'maxWorkgroupsPerDispatch'
  );

  const maxMessagesByWorkgroups = workgroupSize * maxWorkgroupsPerDispatch;
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
  const maxDispatchesInFlight = resolveMaxDispatchesInFlight(profile, prefs);

  return {
    workgroupSize,
    maxWorkgroupsPerDispatch,
    maxMessagesPerDispatch,
    candidateCapacityPerDispatch,
    maxDispatchesInFlight,
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

function resolveMaxDispatchesInFlight(profile: GpuProfile, prefs: SeedSearchLimitPreferences): number {
  if (typeof prefs.maxDispatchesInFlight === 'number') {
    return clampPositive(Math.min(prefs.maxDispatchesInFlight, MAX_AUTOMATIC_DISPATCH_SLOTS), 'maxDispatchesInFlight');
  }

  const profileDefault = profile.isFallbackAdapter
    ? FALLBACK_DISPATCH_SLOTS
    : DEFAULT_DISPATCH_SLOTS_BY_PROFILE[profile.kind] ?? DEFAULT_DISPATCH_SLOTS_BY_PROFILE.unknown;

  return clampPositive(Math.min(profileDefault, MAX_AUTOMATIC_DISPATCH_SLOTS), 'maxDispatchesInFlight');
}

// === GPUプロファイル検出 ===

async function detectGpuProfile(adapter: GPUAdapter): Promise<GpuProfile> {
  const userAgent = getUserAgent();
  const fallbackAwareAdapter = adapter as GPUAdapter & { isFallbackAdapter?: boolean };
  const isFallbackAdapter = Boolean(fallbackAwareAdapter.isFallbackAdapter);

  const webglInference = detectGpuKindFromWebGl();
  if (webglInference) {
    const adapterInfo: AdapterInfoResult = { description: webglInference.renderer };
    return {
      kind: webglInference.kind,
      source: 'webgl',
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
      adapterInfo: undefined,
      isFallbackAdapter,
    };
  }

  return {
    kind: 'unknown',
    source: 'unknown',
    userAgent,
    adapterInfo: undefined,
    isFallbackAdapter,
  };
}

function getUserAgent(): string {
  if (typeof navigator === 'undefined') {
    return '';
  }
  return navigator.userAgent || '';
}

const MOBILE_GPU_KEYWORDS = ['mali', 'adreno', 'powervr', 'apple gpu', 'apple m', 'snapdragon', 'exynos'];
const DISCRETE_GPU_KEYWORDS = ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'amd', 'radeon rx', 'radeon pro', 'arc'];
const INTEGRATED_GPU_KEYWORDS = ['intel', 'iris', 'uhd', 'hd graphics', 'radeon graphics', 'apple'];

function includesKeyword(target: string, keywords: string[]): boolean {
  return keywords.some((keyword) => target.includes(keyword));
}

function classifyGpuString(searchable?: string): GpuProfileKind | undefined {
  if (!searchable) {
    return undefined;
  }
  const normalized = searchable.toLowerCase();
  if (includesKeyword(normalized, MOBILE_GPU_KEYWORDS)) {
    return 'mobile';
  }
  if (includesKeyword(normalized, DISCRETE_GPU_KEYWORDS)) {
    return 'discrete';
  }
  if (includesKeyword(normalized, INTEGRATED_GPU_KEYWORDS)) {
    return 'integrated';
  }
  return undefined;
}

function detectGpuKindFromWebGl(): { kind: GpuProfileKind; renderer: string } | undefined {
  const rendererDescription = detectWebGlRendererDescription();
  if (!rendererDescription) {
    return undefined;
  }
  const inferredKind = classifyGpuString(rendererDescription);
  if (!inferredKind) {
    return undefined;
  }
  return { kind: inferredKind, renderer: rendererDescription };
}

function detectWebGlRendererDescription(): string | undefined {
  const canvas = createCanvasForDetection();
  if (!canvas) {
    return undefined;
  }
  try {
    const context = getWebGlContext(canvas);
    if (!context) {
      return undefined;
    }
    const debugInfo = context.getExtension('WEBGL_debug_renderer_info');
    if (!debugInfo) {
      return undefined;
    }
    const renderer = context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    const loseContext = context.getExtension('WEBGL_lose_context');
    if (loseContext) {
      loseContext.loseContext();
    }
    return typeof renderer === 'string' ? renderer : undefined;
  } catch (error) {
    console.warn('[webgpu] webgl renderer detection failed:', error);
    return undefined;
  }
}

type CanvasLike = HTMLCanvasElement | OffscreenCanvas;

function createCanvasForDetection(): CanvasLike | undefined {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(1, 1);
  }
  if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return canvas;
  }
  return undefined;
}

type WebGlContext = WebGLRenderingContext | WebGL2RenderingContext;

function getWebGlContext(canvas: CanvasLike): WebGlContext | null {
  const canvasAny = canvas as { getContext?: (contextId: string) => RenderingContext | null };
  const getContext = canvasAny.getContext;
  if (typeof getContext !== 'function') {
    return null;
  }
  const tryGet = (contextId: 'webgl2' | 'webgl'): WebGlContext | null => {
    const ctx = getContext.call(canvasAny, contextId);
    return (ctx as WebGlContext | null) ?? null;
  };
  return tryGet('webgl2') ?? tryGet('webgl');
}

function applyProfileOverrides(
  prefs: SeedSearchLimitPreferences,
  _profile: GpuProfile
): SeedSearchLimitPreferences {
  return prefs;
}
