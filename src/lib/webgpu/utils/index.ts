/**
 * WebGPU共通ユーティリティ
 */

// 定数
export {
  BUFFER_ALIGNMENT,
  WORDS_PER_MESSAGE,
  WORDS_PER_HASH,
  BYTES_PER_WORD,
  BYTES_PER_MESSAGE,
  BYTES_PER_HASH,
  MATCH_RECORD_WORDS,
  MATCH_RECORD_BYTES,
  MATCH_OUTPUT_HEADER_WORDS,
  MATCH_OUTPUT_HEADER_BYTES,
  DOUBLE_BUFFER_SET_COUNT,
  MT_SEED_SEARCH_PARAMS_WORDS,
  MT_SEED_RESULT_HEADER_WORDS,
  MT_SEED_RESULT_RECORD_WORDS,
  MT_SEED_MAX_RESULTS_PER_DISPATCH,
} from './constants';

// 型定義
export type {
  SeedSearchJobLimits,
  SeedSearchLimitPreferences,
  WebGpuDeviceContext,
  WebGpuDeviceOptions,
  WebGpuCapabilities,
  GpuProfile,
  GpuProfileKind,
  GpuProfileSource,
  AdapterInfoResult,
} from './types';

// デバイスコンテキスト
export {
  isWebGpuSupported,
  isWebGpuSeedSearchSupported,
  createWebGpuDeviceContext,
  deriveSearchJobLimitsFromDevice,
} from './device-context';
