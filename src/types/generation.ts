/**
 * Generation feature types (Phase3/4)
 * Ref: docs/GENERATION_PHASE3_4_PLAN.md sections 9-11
 * NOTE: 重複防止のため既存 search / parallel / pokemon-raw の型を参照しつつ独立。
 */

import type { UnresolvedPokemonData } from './pokemon-raw';
import { DomainGameMode } from '@/types/domain';
import type { Hardware, ROMRegion } from '@/types/rom';
import type {
  ResolvedPokemonData,
  SerializedResolutionContext,
} from '@/types/pokemon-resolved';
import type { KeyName } from '@/lib/utils/key-input';

// --- Params ---
export type SeedSourceMode = 'lcg' | 'boot-timing';

export interface BootTimingDraft {
  timestampIso?: string;
  keyMask: number;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress: readonly [number, number, number, number, number, number];
}

const DEFAULT_BOOT_TIMING_PLACEHOLDER: BootTimingDraft = {
  timestampIso: undefined,
  keyMask: 0,
  timer0Range: { min: 0, max: 0 },
  vcountRange: { min: 0, max: 0 },
  romRegion: 'JPN',
  hardware: 'DS',
  macAddress: [0, 0, 0, 0, 0, 0] as [number, number, number, number, number, number],
};

export interface GenerationParams {
  baseSeed: bigint;        // 初期Seed
  offset: bigint;          // 開始advance (MVP: 0 既定)
  maxAdvances: number;     // 列挙上限 (≤ 1_000_000)
  maxResults: number;      // UI保持上限 (≤ 100_000)
  version: 'B' | 'W' | 'B2' | 'W2';
  encounterType: number;   // DomainEncounterType 値（StaticLegendary=14 は WASM 送信時に StaticSymbol=10 へ正規化）
  tid: number;             // 0-65535
  sid: number;             // 0-65535
  syncEnabled: boolean;
  syncNatureId: number;    // 0-24
  shinyCharm: boolean;     // 光るお守り所持
  isShinyLocked: boolean;  // 選択エンカウントが色違いロック対象か
  stopAtFirstShiny: boolean;
  stopOnCap: boolean;      // maxResults 到達で終了するか（デフォルト true）
  newGame: boolean;
  withSave: boolean;       // newGame 時に既存セーブを利用するか
  memoryLink: boolean;
}

// 16進文字列保持用: store/UI ではこちらを使い、worker開始直前に GenerationParams へ変換
export interface GenerationParamsHex {
  baseSeedHex: string;     // 小文字/大文字混在許容。正規化時に toLowerCase()
  offsetHex: string;
  maxAdvances: number;
  maxResults: number;
  version: 'B' | 'W' | 'B2' | 'W2';
  encounterType: number;
  tid: number;
  sid: number;
  syncEnabled: boolean;
  syncNatureId: number;
  /** 所持している場合 true (後続: 色違い確率計算に利用予定) */
  shinyCharm: boolean;
  /** 選択エンカウントが色違いロック対象か */
  isShinyLocked: boolean;
  stopAtFirstShiny: boolean;
  stopOnCap: boolean;
  /**
   * UI 拡張: Ability 選択モード (Phase2 で syncEnabled との統合制御に使用)
   * 現行 WASM パラメータへは未伝播。syncEnabled との整合は UI 側で維持。
   */
  abilityMode?: 'none' | 'sync' | 'compound';
  /** BW2 Memory Link 状態 */
  memoryLink: boolean;
  /** True when starting a new game flow */
  newGame: boolean;
  /** True when starting with an existing save */
  withSave: boolean;
  seedSourceMode: SeedSourceMode;
  bootTiming: BootTimingDraft;
}

export function hexParamsToGenerationParams(h: GenerationParamsHex): GenerationParams {
  return {
    baseSeed: BigInt('0x' + normalizeHex(h.baseSeedHex)),
    offset: BigInt('0x' + normalizeHex(h.offsetHex)),
    maxAdvances: h.maxAdvances,
    maxResults: h.maxResults,
    version: h.version,
    encounterType: h.encounterType,
    tid: h.tid,
    sid: h.sid,
    syncEnabled: h.syncEnabled,
    syncNatureId: h.syncNatureId,
  shinyCharm: Boolean(h.shinyCharm) && (h.version === 'B2' || h.version === 'W2'),
  isShinyLocked: Boolean(h.isShinyLocked),
    stopAtFirstShiny: h.stopAtFirstShiny,
    stopOnCap: h.stopOnCap,
    newGame: h.newGame,
    withSave: h.withSave,
    memoryLink: h.memoryLink,
  };
}

export function generationParamsToHex(p: GenerationParams): GenerationParamsHex {
  return {
    baseSeedHex: p.baseSeed.toString(16),
    offsetHex: p.offset.toString(16),
    maxAdvances: p.maxAdvances,
    maxResults: p.maxResults,
    version: p.version,
    encounterType: p.encounterType,
    tid: p.tid,
    sid: p.sid,
    syncEnabled: p.syncEnabled,
    syncNatureId: p.syncNatureId,
    shinyCharm: p.shinyCharm,
    isShinyLocked: p.isShinyLocked,
    stopAtFirstShiny: p.stopAtFirstShiny,
    stopOnCap: p.stopOnCap,
    memoryLink: p.memoryLink,
    newGame: p.newGame,
    withSave: p.withSave,
    seedSourceMode: 'lcg',
    bootTiming: { ...DEFAULT_BOOT_TIMING_PLACEHOLDER },
  };
}

function normalizeHex(s: string): string {
  const v = s.trim().replace(/^0x/i,'');
  return v === '' ? '0' : v.toLowerCase();
}

// 正規化後 (境界補正/デフォルト適用済み) – そのままエイリアス
export type NormalizedGenerationParams = GenerationParams;

// --- Result 型 ---
// UnresolvedPokemonData が advance を含むため、そのまま公開APIとして利用する。
export type GenerationResult = UnresolvedPokemonData & {
  seedSourceMode?: SeedSourceMode;
  derivedSeedIndex?: number;
  timer0?: number;
  vcount?: number;
  bootTimestampIso?: string;
  keyInputNames?: KeyName[];
  macAddress?: readonly [number, number, number, number, number, number];
};

export interface GenerationResultsPayload {
  results: GenerationResult[];
  resolved?: ResolvedPokemonData[];
}

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
  | {
      type: 'START_GENERATION';
      params: GenerationParams;
      requestId?: string;
      resolutionContext?: SerializedResolutionContext;
    }
  | { type: 'STOP'; requestId?: string; reason?: string };

export type GenerationWorkerResponse =
  | { type: 'READY'; version: '1' }
  | { type: 'RESULTS'; payload: GenerationResultsPayload }
  | { type: 'COMPLETE'; payload: GenerationCompletion }
  | { type: 'ERROR'; message: string; category: GenerationErrorCategory; fatal: boolean };

// --- Utility Type Guards ---
export function isGenerationWorkerResponse(msg: unknown): msg is GenerationWorkerResponse {
  if (!msg || typeof msg !== 'object') return false;
  const m = msg as { type?: unknown };
  if (typeof m.type !== 'string') return false;
  const allowed: ReadonlySet<GenerationWorkerResponse['type']> = new Set(['READY','RESULTS','COMPLETE','ERROR']);
  return allowed.has(m.type as GenerationWorkerResponse['type']);
}

// --- Adapter Helper ---
// rawLikeToUnresolved は重複となるため削除 (必要なら GenerationResult をそのまま利用)

// --- Validation ---
export function validateGenerationParams(p: GenerationParams): string[] {
  const errors: string[] = [];
  // 基本範囲
  if (p.maxAdvances < 1 || p.maxAdvances > 1_000_000) errors.push('maxAdvances out of range');
  if (p.maxResults < 1 || p.maxResults > 100_000) errors.push('maxResults out of range');
  if (p.syncNatureId < 0 || p.syncNatureId > 24) errors.push('syncNatureId out of range');
  if (p.tid < 0 || p.tid > 65535) errors.push('tid out of range');
  if (p.sid < 0 || p.sid > 65535) errors.push('sid out of range');
  // 追加簡素チェック
  if (p.baseSeed < 0n) errors.push('baseSeed must be non-negative');
  if (p.offset < 0n) errors.push('offset must be non-negative');
  if (p.offset >= BigInt(p.maxAdvances)) errors.push('offset must be < maxAdvances');
  const allowedEncounter = new Set([0,1,2,3,4,5,6,7,10,11,12,13,14,20]);
  if (!allowedEncounter.has(p.encounterType)) errors.push('encounterType invalid');
  if (!p.newGame && !p.withSave) errors.push('withSave must be true when continuing a game');
  if ((p.version === 'B' || p.version === 'W') && p.memoryLink) errors.push('memoryLink is only available in BW2');
  if (!p.withSave && p.memoryLink) errors.push('memoryLink requires a save file');
  try {
    deriveDomainGameMode(p);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    errors.push(message || 'invalid game mode');
  }
  return errors;
}

export function requiresStaticSelection(encounterType: number): boolean {
  const staticTypes = new Set([10, 14, 11, 12, 13]); // StaticSymbol, StaticLegendary, Starter, Fossil, Event
  return staticTypes.has(encounterType);
}

export function deriveDomainGameMode(input: Pick<GenerationParams, 'version' | 'newGame' | 'withSave' | 'memoryLink'>): DomainGameMode {
  const { version, newGame, withSave, memoryLink } = input;
  const isBw1 = version === 'B' || version === 'W';
  if (isBw1) {
    if (memoryLink) {
      throw new Error('BW versions do not support memory link');
    }
    if (!newGame && !withSave) {
      throw new Error('Continue mode requires an existing save');
    }
    if (newGame) {
      return withSave ? DomainGameMode.BwNewGameWithSave : DomainGameMode.BwNewGameNoSave;
    }
    return DomainGameMode.BwContinue;
  }

  if (!newGame) {
    if (!withSave) {
      throw new Error('Continue mode requires an existing save');
    }
    return memoryLink ? DomainGameMode.Bw2ContinueWithMemoryLink : DomainGameMode.Bw2ContinueNoMemoryLink;
  }

  if (!withSave) {
    if (memoryLink) {
      throw new Error('Memory link requires a save file');
    }
    return DomainGameMode.Bw2NewGameNoSave;
  }

  return memoryLink ? DomainGameMode.Bw2NewGameWithMemoryLinkSave : DomainGameMode.Bw2NewGameNoMemoryLinkSave;
}
