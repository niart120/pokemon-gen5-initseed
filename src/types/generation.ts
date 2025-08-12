/**
 * Generation feature types (Phase3/4)
 * Ref: docs/GENERATION_PHASE3_4_PLAN.md sections 9-11
 * NOTE: 重複防止のため既存 search / parallel / pokemon-raw の型を参照しつつ独立。
 */

import type { UnresolvedPokemonData } from './pokemon-raw';

// --- Params ---
export interface GenerationParams {
  baseSeed: bigint;        // 初期シード
  offset: bigint;          // 開始advance (MVP: 0 既定)
  maxAdvances: number;     // 列挙上限 (≤ 1_000_000)
  maxResults: number;      // UI保持上限 (≤ 100_000)
  version: 'B' | 'W' | 'B2' | 'W2';
  encounterType: number;   // WASM EncounterType u8 マッピング値 (0,1,2,3,4,5,6,7,10,11,12,13,20)
  tid: number;             // 0-65535
  sid: number;             // 0-65535
  syncEnabled: boolean;
  syncNatureId: number;    // 0-24
  stopAtFirstShiny: boolean;
  stopOnCap: boolean;      // maxResults 到達で終了するか（デフォルト true）
  batchSize: number;       // 1バッチ生成数 (推奨 1000, ≤ 10000)
}

export interface NormalizedGenerationParams extends GenerationParams {
  // 正規化後 (境界補正/デフォルト適用済み)
}

// --- Result 型 ---
// WASM生データ(UnresolvedPokemonData) に generation 文脈上の advance を付与した最小構造。
// これを worker から直接送出し UI/store が保持する。重複構造を避けるため専用RawLikeは用意しない。
export type GenerationResult = UnresolvedPokemonData & { advance: number };

// --- Progress / Results ---
export interface GenerationProgress {
  processedAdvances: number;
  totalAdvances: number;
  resultsCount: number;
  elapsedMs: number;
  /** 生スループット (直近計算) */
  throughput: number; // DEPRECATED: 後方互換 (raw と同値保持)。将来的除去予定。
  /** 新: 生スループット */
  throughputRaw?: number;
  /** 新: EMA 平滑スループット */
  throughputEma?: number;
  /** 推定残り時間 (ms) */
  etaMs: number;
  status: 'idle' | 'running' | 'paused' | 'stopped' | 'completed' | 'error';
}

export interface GenerationResultBatch {
  batchIndex: number;
  batchSize: number;
  results: GenerationResult[]; // plain objects (postMessage structured clone OK / bigint)
  cumulativeResults: number;
}

// 固定進捗通知間隔 (ms) - UI/worker 双方で参照
export const FIXED_PROGRESS_INTERVAL_MS = 250 as const;

export type GenerationCompletion = {
  reason: 'max-advances' | 'max-results' | 'first-shiny' | 'stopped' | 'error';
  processedAdvances: number;
  resultsCount: number;
  elapsedMs: number;
  shinyFound: boolean;
};

export type GenerationErrorCategory = 'VALIDATION' | 'WASM_INIT' | 'RUNTIME' | 'ABORTED';

// --- Completion Reason Labels ---
export const GENERATION_COMPLETION_REASON_LABELS: Record<GenerationCompletion['reason'], string> = {
  'max-advances': '列挙上限到達',
  'max-results': '結果件数上限到達',
  'first-shiny': '最初の色違い発見',
  'stopped': 'ユーザー停止',
  'error': 'エラー終了',
};

export const GENERATION_COMPLETION_REASON_DESCRIPTIONS: Partial<Record<GenerationCompletion['reason'], string>> = {
  'max-advances': '指定した最大advance数に達したため終了しました。',
  'max-results': '結果保持件数が上限に達したため終了しました。',
  'first-shiny': '色違い検出オプションにより終了しました。',
  'stopped': 'ユーザー操作により中断されました。',
  'error': '実行中にエラーが発生しました。',
};

export function getGenerationCompletionLabel(reason: GenerationCompletion['reason']): string {
  return GENERATION_COMPLETION_REASON_LABELS[reason] ?? reason;
}

// --- Worker Messages ---
export type GenerationWorkerRequest =
  | { type: 'START_GENERATION'; params: GenerationParams; requestId?: string }
  | { type: 'PAUSE'; requestId?: string }
  | { type: 'RESUME'; requestId?: string }
  | { type: 'STOP'; requestId?: string; reason?: string };

export type GenerationWorkerResponse =
  | { type: 'READY'; version: '1' }
  | { type: 'PROGRESS'; payload: GenerationProgress }
  | { type: 'RESULT_BATCH'; payload: GenerationResultBatch }
  | { type: 'PAUSED'; message?: string }
  | { type: 'RESUMED' }
  | { type: 'STOPPED'; payload: Omit<GenerationCompletion, 'reason'> & { reason: 'stopped' } }
  | { type: 'COMPLETE'; payload: GenerationCompletion }
  | { type: 'ERROR'; message: string; category: GenerationErrorCategory; fatal: boolean };

// --- Utility Type Guards ---
export function isGenerationWorkerResponse(msg: any): msg is GenerationWorkerResponse {
  return msg && typeof msg === 'object' && typeof msg.type === 'string' && (
    ['READY','PROGRESS','RESULT_BATCH','PAUSED','RESUMED','STOPPED','COMPLETE','ERROR'] as const
  ).includes(msg.type);
}

export function isResultBatch(msg: GenerationWorkerResponse): msg is Extract<GenerationWorkerResponse,{type:'RESULT_BATCH'}> {
  return msg.type === 'RESULT_BATCH';
}

export function isProgress(msg: GenerationWorkerResponse): msg is Extract<GenerationWorkerResponse,{type:'PROGRESS'}> {
  return msg.type === 'PROGRESS';
}

// --- Adapter Helper ---
// rawLikeToUnresolved は重複となるため削除 (必要なら GenerationResult をそのまま利用)

// --- Validation ---
export function validateGenerationParams(p: GenerationParams): string[] {
  const errors: string[] = [];
  // 基本範囲
  if (p.maxAdvances < 1 || p.maxAdvances > 1_000_000) errors.push('maxAdvances out of range');
  if (p.maxResults < 1 || p.maxResults > 100_000) errors.push('maxResults out of range');
  if (p.batchSize < 1 || p.batchSize > 10_000) errors.push('batchSize out of range');
  if (p.syncNatureId < 0 || p.syncNatureId > 24) errors.push('syncNatureId out of range');
  if (p.tid < 0 || p.tid > 65535) errors.push('tid out of range');
  if (p.sid < 0 || p.sid > 65535) errors.push('sid out of range');
  // 追加簡素チェック
  if (p.baseSeed < 0n) errors.push('baseSeed must be non-negative');
  if (p.offset < 0n) errors.push('offset must be non-negative');
  if (p.offset >= BigInt(p.maxAdvances)) errors.push('offset must be < maxAdvances');
  // A方針: batchSize は maxAdvances を超過不可 (テスト仕様維持)
  if (p.batchSize > p.maxAdvances) errors.push('batchSize must be <= maxAdvances');
  const allowedEncounter = new Set([0,1,2,3,4,5,6,7,10,11,12,13,20]);
  if (!allowedEncounter.has(p.encounterType)) errors.push('encounterType invalid');
  if (p.maxResults > p.maxAdvances) errors.push('maxResults should be <= maxAdvances');
  return errors;
}
