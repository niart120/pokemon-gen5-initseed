/**
 * MT Seed 32bit全探索 型定義
 * Based on: spec/implementation/07-mt-seed-full-search.md
 */

import type { IvSet } from './egg';

// === IVコード ===

/**
 * IVコード（30bit圧縮表現）
 * 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]
 */
export type IvCode = number;

/**
 * ステータス範囲
 */
export interface StatRange {
  min: number; // 0-31
  max: number; // 0-31
}

/**
 * 個体値検索フィルター
 */
export interface IvSearchFilter {
  ivRanges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
  hiddenPowerType?: number; // 0-15
  hiddenPowerPower?: number; // 30-70
}

/**
 * IVコード生成結果
 */
export type IvCodeGenerationResult =
  | { success: true; ivCodes: IvCode[] }
  | { success: false; error: 'TOO_MANY_COMBINATIONS'; estimatedCount: number };

// === ジョブ定義 ===

/**
 * MT Seed検索ジョブ
 */
export interface MtSeedSearchJob {
  /** 検索範囲（閉区間）[start, end] */
  searchRange: {
    start: number; // u32
    end: number; // u32
  };

  /** 検索対象IVコードリスト */
  ivCodes: IvCode[];

  /** MT消費数 */
  mtAdvances: number;

  /** ジョブID（進捗追跡用） */
  jobId: number;
}

/**
 * 単一のマッチ結果
 */
export interface MtSeedMatch {
  /** 発見されたMT Seed */
  mtSeed: number;

  /** 対応するIVコード */
  ivCode: IvCode;

  /** デコードされたIVセット */
  ivSet: IvSet;
}

/**
 * ジョブ単位の検索結果
 * 1ジョブに対して複数のマッチが発生しうるため配列で保持
 */
export interface MtSeedSearchResult {
  /** ジョブID */
  jobId: number;

  /** マッチしたSeed/IVコードのペア配列 */
  matches: MtSeedMatch[];
}

// === ジョブ計画 ===

/**
 * ジョブ計画設定（共通）
 */
export interface JobPlannerConfig {
  /** 全探索範囲 */
  fullRange: { start: number; end: number }; // 通常 [0, 0xFFFFFFFF]

  /** IVコードリスト */
  ivCodes: IvCode[];

  /** MT消費数 */
  mtAdvances: number;
}

/**
 * GPU向けジョブ計画設定
 */
export interface GpuJobPlannerConfig extends JobPlannerConfig {
  /** WebGPUデバイス制約 */
  deviceLimits: {
    maxComputeWorkgroupsPerDimension: number;
    maxStorageBufferBindingSize: number;
  };

  /** ワークグループサイズ */
  workgroupSize: number;
}

/**
 * CPU向けジョブ計画設定
 */
export interface CpuJobPlannerConfig extends JobPlannerConfig {
  /** 使用するWorker数（通常 navigator.hardwareConcurrency） */
  workerCount: number;
}

/**
 * ジョブ計画
 */
export interface JobPlan {
  jobs: MtSeedSearchJob[];
  totalSearchSpace: number;
  estimatedTimeMs: number;
}

// === Worker通信 ===

/**
 * 検索進捗
 */
export interface MtSeedSearchProgress {
  jobId: number;
  processedCount: number;
  totalCount: number;
  elapsedMs: number;
  matchesFound: number;
}

/**
 * 検索完了情報
 */
export interface MtSeedSearchCompletion {
  reason: 'finished' | 'stopped' | 'error';
  totalProcessed: number;
  totalMatches: number;
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type MtSeedSearchErrorCategory =
  | 'VALIDATION'
  | 'WASM_INIT'
  | 'GPU_INIT'
  | 'RUNTIME';

/**
 * Workerリクエスト
 */
export type MtSeedSearchWorkerRequest =
  | { type: 'START'; job: MtSeedSearchJob }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'STOP' };

/**
 * Workerレスポンス
 */
export type MtSeedSearchWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: MtSeedSearchProgress }
  | { type: 'RESULTS'; payload: MtSeedSearchResult }
  | { type: 'COMPLETE'; payload: MtSeedSearchCompletion }
  | { type: 'ERROR'; message: string; category: MtSeedSearchErrorCategory };

// === 定数 ===

/** IVコードの最大件数 */
export const MAX_IV_CODES = 1024;

/** 32bit全探索の範囲 */
export const FULL_SEARCH_RANGE = {
  start: 0,
  end: 0xffffffff,
} as const;

// === ユーティリティ関数 ===

/**
 * IVセットをIVコードにエンコード
 */
export function encodeIvCode(ivs: IvSet): IvCode {
  return (
    (ivs[0] << 25) |
    (ivs[1] << 20) |
    (ivs[2] << 15) |
    (ivs[3] << 10) |
    (ivs[4] << 5) |
    ivs[5]
  );
}

/**
 * IVコードをIVセットにデコード
 */
export function decodeIvCode(code: IvCode): IvSet {
  return [
    (code >> 25) & 0x1f,
    (code >> 20) & 0x1f,
    (code >> 15) & 0x1f,
    (code >> 10) & 0x1f,
    (code >> 5) & 0x1f,
    code & 0x1f,
  ];
}

/**
 * Workerレスポンスの型ガード
 */
export function isMtSeedSearchWorkerResponse(
  data: unknown
): data is MtSeedSearchWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.type === 'string' &&
    ['READY', 'PROGRESS', 'RESULTS', 'COMPLETE', 'ERROR'].includes(obj.type)
  );
}

/**
 * Workerリクエストの型ガード
 */
export function isMtSeedSearchWorkerRequest(
  data: unknown
): data is MtSeedSearchWorkerRequest {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.type === 'string' &&
    ['START', 'PAUSE', 'RESUME', 'STOP'].includes(obj.type)
  );
}
