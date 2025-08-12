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
  progressIntervalMs: number; // 進捗通知間隔 (ms)
  batchSize: number;       // 1バッチ生成数 (推奨 1000, ≤ 10000)
}

export interface NormalizedGenerationParams extends GenerationParams {
  // 正規化後 (境界補正/デフォルト適用済み)
}

// --- RawLike (worker送出最小構造) ---
export interface GenerationRawLike {
  seed: bigint;
  pid: number;
  nature: number;
  ability_slot: number;
  gender_value: number;
  encounter_slot_value: number;
  encounter_type: number;
  level_rand_value: bigint;
  shiny_type: number;
  sync_applied: boolean;
  advance: number; // offset + index
}

// --- Progress / Results ---
export interface GenerationProgress {
  processedAdvances: number;
  totalAdvances: number;
  resultsCount: number;
  elapsedMs: number;
  throughput: number; // advances/sec
  etaMs: number;
  status: 'idle' | 'running' | 'paused' | 'stopped' | 'completed' | 'error';
}

export interface GenerationResultBatch {
  batchIndex: number;
  batchSize: number;
  results: GenerationRawLike[]; // plain objects (postMessage structured clone OK / bigint)
  cumulativeResults: number;
}

export type GenerationCompletion = {
  reason: 'max-advances' | 'max-results' | 'first-shiny' | 'stopped' | 'error';
  processedAdvances: number;
  resultsCount: number;
  elapsedMs: number;
  shinyFound: boolean;
};

export type GenerationErrorCategory = 'VALIDATION' | 'WASM_INIT' | 'RUNTIME' | 'ABORTED';

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
export function rawLikeToUnresolved(r: GenerationRawLike): UnresolvedPokemonData {
  return {
    seed: r.seed,
    pid: r.pid,
    nature: r.nature,
    sync_applied: r.sync_applied,
    ability_slot: r.ability_slot,
    gender_value: r.gender_value,
    encounter_slot_value: r.encounter_slot_value,
    encounter_type: r.encounter_type,
    level_rand_value: r.level_rand_value,
    shiny_type: r.shiny_type,
  };
}

// --- Validation ---
export function validateGenerationParams(p: GenerationParams): string[] {
  const errors: string[] = [];
  if (p.maxAdvances < 1 || p.maxAdvances > 1_000_000) errors.push('maxAdvances out of range');
  if (p.maxResults < 1 || p.maxResults > 100_000) errors.push('maxResults out of range');
  if (p.batchSize < 1 || p.batchSize > 10_000) errors.push('batchSize out of range');
  if (p.syncNatureId < 0 || p.syncNatureId > 24) errors.push('syncNatureId out of range');
  if (p.tid < 0 || p.tid > 65535) errors.push('tid out of range');
  if (p.sid < 0 || p.sid > 65535) errors.push('sid out of range');
  return errors;
}
