/**
 * WebGPU共通型定義
 */

/**
 * GPU検索ジョブの制限値
 * デバイス能力から導出される値
 */
export interface SeedSearchJobLimits {
  /** ワークグループサイズ（スレッド数/ワークグループ） */
  workgroupSize: number;
  /** 1ディスパッチあたりの最大ワークグループ数 */
  maxWorkgroupsPerDispatch: number;
  /** 1ディスパッチあたりの最大メッセージ数 */
  maxMessagesPerDispatch: number;
  /** 1ディスパッチあたりの最大候補バッファ容量 */
  candidateCapacityPerDispatch: number;
  /** 同時実行可能なディスパッチ数（パイプライン深度） */
  maxDispatchesInFlight: number;
}

/**
 * GPUプロファイル種別
 */
export type GpuProfileKind = 'unknown' | 'integrated' | 'mobile' | 'discrete';

/**
 * GPUプロファイル検出ソース
 */
export type GpuProfileSource = 'unknown' | 'user-agent' | 'adapter-info' | 'webgl' | 'fallback';

/**
 * アダプタ情報
 */
export interface AdapterInfoResult {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

/**
 * GPUプロファイル
 */
export interface GpuProfile {
  kind: GpuProfileKind;
  source: GpuProfileSource;
  userAgent?: string;
  isFallbackAdapter: boolean;
  adapterInfo?: AdapterInfoResult;
}

/**
 * デバイスオプション
 */
export interface WebGpuDeviceOptions {
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: GPUDeviceDescriptor['requiredLimits'];
  label?: string;
  powerPreference?: GPUPowerPreference;
}

/**
 * デバイス能力
 */
export interface WebGpuCapabilities {
  limits: GPUSupportedLimits;
  features: ReadonlySet<GPUFeatureName>;
}

/**
 * 検索ジョブ制限のプリファレンス
 */
export interface SeedSearchLimitPreferences {
  workgroupSize?: number;
  maxWorkgroupsPerDispatch?: number;
  maxMessagesPerDispatch?: number;
  candidateCapacityPerDispatch?: number;
  maxDispatchesInFlight?: number;
}

/**
 * WebGPUデバイスコンテキスト
 */
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
