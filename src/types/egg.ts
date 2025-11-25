/**
 * Egg generation feature types
 * Based on: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md
 */

import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import type { KeyName } from '@/lib/utils/key-input';

// === 基本型 ===

/**
 * 親個体の役割
 */
export type ParentRole = 'male' | 'female';

/**
 * かわらずのいし設定
 */
export type EverstonePlan =
  | { type: 'none' }
  | { type: 'fixed'; nature: number }; // 0-24

/**
 * 親個体のIVセット
 * 各値は 0-31 または 32 (Unknown)
 */
export type IvSet = [number, number, number, number, number, number]; // HP, Atk, Def, SpA, SpD, Spe

/**
 * Unknown IV sentinel value
 */
export const IV_UNKNOWN = 32;

/**
 * 親個体情報
 */
export interface ParentsIVs {
  male: IvSet;
  female: IvSet;
}

/**
 * 遺伝スロット情報
 */
export interface InheritanceSlot {
  stat: number; // 0-5 (HP, Atk, Def, SpA, SpD, Spe)
  parent: ParentRole;
}

/**
 * 個体値範囲フィルター
 */
export interface StatRange {
  min: number; // 0-32
  max: number; // 0-32
}

/**
 * めざめるパワー情報
 */
export type HiddenPowerInfo =
  | { type: 'known'; hpType: number; power: number } // hpType: 0-15
  | { type: 'unknown' };

/**
 * 性別比設定
 */
export interface GenderRatioConfig {
  threshold: number; // 0-255
  genderless: boolean;
}

/**
 * 生成条件
 */
export interface EggGenerationConditions {
  hasNidoranFlag: boolean;        // ニドラン系/バルビート系
  everstone: EverstonePlan;       // かわらずのいし
  usesDitto: boolean;             // メタモン利用
  femaleParentAbility: 0 | 1 | 2; // 親♀の特性 (0=特性1, 1=特性2, 2=隠れ特性)
  masudaMethod: boolean;          // 国際孵化 (true=reroll 3, false=reroll 0)
  tid: number;                    // 0-65535
  sid: number;                    // 0-65535
  genderRatio: GenderRatioConfig;
}

/**
 * 個体フィルター
 */
export interface EggIndividualFilter {
  ivRanges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
  nature?: number;                // 0-24
  gender?: 'male' | 'female' | 'genderless';
  ability?: 0 | 1 | 2;            // 0=特性1, 1=特性2, 2=夢特性
  shiny?: 0 | 1 | 2;              // 0=通常, 1=正方形色違い, 2=星型色違い
  hiddenPowerType?: number;       // 0-15
  hiddenPowerPower?: number;      // 30-70
}

/**
 * 生成された個体データ
 */
export interface ResolvedEgg {
  ivs: IvSet;
  nature: number;         // 0-24
  gender: 'male' | 'female' | 'genderless';
  ability: 0 | 1 | 2;
  shiny: 0 | 1 | 2;
  pid: number;            // u32
  hiddenPower: HiddenPowerInfo;
}

/**
 * 列挙された個体データ（advance情報付き）
 */
export interface EnumeratedEggData {
  advance: number;        // bigint → number に変換
  lcgSeedHex: string;     // その個体生成時のLCG Seed (hex string)
  egg: ResolvedEgg;
  isStable: boolean;      // NPC消費考慮時の安定性
}

// === パラメータ型 ===

/**
 * GameMode (offset_calculator.rs)
 */
export enum EggGameMode {
  BwNew = 0,
  BwContinue = 1,
  Bw2New = 2,
  Bw2Continue = 3,
}

/**
 * ROMバージョンとnewGameフラグからEggGameModeを導出
 */
export function deriveEggGameMode(romVersion: string, newGame: boolean): EggGameMode {
  const isBw2 = romVersion === 'B2' || romVersion === 'W2';
  if (isBw2) {
    return newGame ? EggGameMode.Bw2New : EggGameMode.Bw2Continue;
  }
  return newGame ? EggGameMode.BwNew : EggGameMode.BwContinue;
}

/**
 * タマゴ生成パラメータ
 */
export interface EggGenerationParams {
  baseSeed: bigint;                      // 初期Seed
  userOffset: bigint;                    // 開始advance (0から開始が基本)
  count: number;                         // 列挙上限 (1-100000)
  conditions: EggGenerationConditions;   // 生成条件
  parents: ParentsIVs;                   // 親個体値
  filter: EggIndividualFilter | null;    // フィルター (null=全件)
  considerNpcConsumption: boolean;       // NPC消費考慮
  gameMode: EggGameMode;                 // GameMode
}

/**
 * UI用16進数パラメータ
 */
export interface EggGenerationParamsHex {
  baseSeedHex: string;
  userOffsetHex: string;
  count: number;
  conditions: EggGenerationConditions;
  parents: ParentsIVs;
  filter: EggIndividualFilter | null;
  filterDisabled: boolean;
  considerNpcConsumption: boolean;
  gameMode: EggGameMode;
  // Boot-Timing 対応
  seedSourceMode: EggSeedSourceMode;
  bootTiming: EggBootTimingDraft;
}

// === Boot-Timing Enumeration 型 ===

/**
 * Seed入力モード
 */
export type EggSeedSourceMode = 'lcg' | 'boot-timing';

/**
 * 起動時間パラメータ (UI入力)
 */
export interface EggBootTimingDraft {
  timestampIso?: string;
  keyMask: number;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress: readonly [number, number, number, number, number, number];
}

/**
 * デフォルト Boot-Timing Draft
 */
export function createDefaultEggBootTimingDraft(): EggBootTimingDraft {
  return {
    timestampIso: undefined,
    keyMask: 0,
    timer0Range: { min: 0, max: 0 },
    vcountRange: { min: 0, max: 0 },
    romVersion: 'B',
    romRegion: 'JPN',
    hardware: 'DS',
    macAddress: [0, 0, 0, 0, 0, 0],
  };
}

/**
 * 導出されたSeedのメタデータ
 */
export interface DerivedEggSeedMetadata {
  readonly seedSourceMode: 'boot-timing';
  readonly derivedSeedIndex: number;
  readonly timer0: number;
  readonly vcount: number;
  readonly keyMask: number;
  readonly keyCode: number;
  readonly bootTimestampIso: string;
  readonly macAddress: readonly [number, number, number, number, number, number];
  readonly seedSourceSeedHex: string;
}

/**
 * 導出されたSeedジョブ
 */
export interface DerivedEggSeedJob {
  params: EggGenerationParams;
  metadata: DerivedEggSeedMetadata;
}

/**
 * Boot-Timing実行時の集計情報
 */
export interface DerivedEggSeedAggregate {
  processedCount: number;
  filteredCount: number;
  elapsedMs: number;
}

/**
 * 起動時間列挙の進捗状態
 */
export interface DerivedEggSeedRunState {
  readonly jobs: DerivedEggSeedJob[];
  readonly cursor: number;
  readonly total: number;
  readonly aggregate: DerivedEggSeedAggregate;
  readonly abortRequested: boolean;
}

/**
 * 列挙された個体データ（boot-timing用拡張）
 */
export interface EnumeratedEggDataWithBootTiming extends EnumeratedEggData {
  // Boot-Timing モード時に付与されるメタデータ
  seedSourceMode?: EggSeedSourceMode;
  derivedSeedIndex?: number;
  seedSourceSeedHex?: string;
  timer0?: number;
  vcount?: number;
  bootTimestampIso?: string;
  keyInputNames?: KeyName[];
  macAddress?: readonly [number, number, number, number, number, number];
}

/**
 * 親IV入力の状態
 */
export interface ParentIvInputState {
  value: number;        // 0-31 の値（チェック時は無効）
  isUnknown: boolean;   // true の場合、実際の値は 32 (Unknown)
}

/**
 * フィルターIV範囲入力の状態
 */
export interface FilterIvRangeInputState {
  min: number;           // 0-31
  max: number;           // 0-31（includeUnknown=true時は32に強制）
  includeUnknown: boolean; // Unknownを含める場合はtrue
}

// === 変換関数 ===

function normalizeHex(hex: string): string {
  const cleaned = hex.toLowerCase().replace(/^0x/, '');
  return cleaned || '0';
}

/**
 * パラメータ変換関数
 */
export function hexParamsToEggParams(h: EggGenerationParamsHex): EggGenerationParams {
  return {
    baseSeed: BigInt('0x' + normalizeHex(h.baseSeedHex)),
    userOffset: BigInt('0x' + normalizeHex(h.userOffsetHex)),
    count: h.count,
    conditions: h.conditions,
    parents: h.parents,
    filter: h.filter,
    considerNpcConsumption: h.considerNpcConsumption,
    gameMode: h.gameMode,
  };
}

/**
 * EggGenerationParams → EggGenerationParamsHex 変換
 */
export function eggParamsToHex(p: EggGenerationParams): EggGenerationParamsHex {
  return {
    baseSeedHex: p.baseSeed.toString(16).toUpperCase(),
    userOffsetHex: p.userOffset.toString(16).toUpperCase(),
    count: p.count,
    conditions: p.conditions,
    parents: p.parents,
    filter: p.filter,
    filterDisabled: false,
    considerNpcConsumption: p.considerNpcConsumption,
    gameMode: p.gameMode,
    seedSourceMode: 'lcg',
    bootTiming: createDefaultEggBootTimingDraft(),
  };
}

/**
 * IvSet変換（親IV入力状態から）
 */
export function parentIvInputsToIvSet(inputs: ParentIvInputState[]): IvSet {
  if (inputs.length !== 6) {
    throw new Error('Invalid IvSet length');
  }
  return inputs.map(input => input.isUnknown ? IV_UNKNOWN : input.value) as IvSet;
}

/**
 * StatRange変換（フィルターIV範囲入力状態から）
 */
export function filterIvRangeInputToStatRange(input: FilterIvRangeInputState): StatRange {
  return {
    min: input.min,
    max: input.includeUnknown ? IV_UNKNOWN : input.max,
  };
}

// === バリデーション ===

const MAX_COUNT = 100000;
const MAX_TID_SID = 65535;

/**
 * パラメータバリデーション
 */
export function validateEggParams(params: EggGenerationParams): string[] {
  const errors: string[] = [];

  if (params.count < 1 || params.count > MAX_COUNT) {
    errors.push(`count must be 1-${MAX_COUNT}`);
  }

  if (params.conditions.tid < 0 || params.conditions.tid > MAX_TID_SID) {
    errors.push(`tid must be 0-${MAX_TID_SID}`);
  }

  if (params.conditions.sid < 0 || params.conditions.sid > MAX_TID_SID) {
    errors.push(`sid must be 0-${MAX_TID_SID}`);
  }

  // IV値検証
  const validateIvSet = (ivs: IvSet, name: string) => {
    ivs.forEach((iv, i) => {
      if (iv < 0 || iv > IV_UNKNOWN) {
        errors.push(`${name}[${i}] must be 0-${IV_UNKNOWN}`);
      }
    });
  };

  validateIvSet(params.parents.male, 'parents.male');
  validateIvSet(params.parents.female, 'parents.female');

  // フィルター検証
  if (params.filter) {
    params.filter.ivRanges.forEach((range, i) => {
      if (range.min < 0 || range.min > IV_UNKNOWN) {
        errors.push(`filter.ivRanges[${i}].min must be 0-${IV_UNKNOWN}`);
      }
      if (range.max < 0 || range.max > IV_UNKNOWN) {
        errors.push(`filter.ivRanges[${i}].max must be 0-${IV_UNKNOWN}`);
      }
      if (range.min > range.max) {
        errors.push(`filter.ivRanges[${i}].min must be <= max`);
      }
    });
  }

  return errors;
}

// === Worker通信型 ===

/**
 * Worker リクエスト
 */
export type EggWorkerRequest =
  | { type: 'START_GENERATION'; params: EggGenerationParams; requestId?: string }
  | { type: 'STOP'; requestId?: string };

/**
 * Worker レスポンス
 */
export type EggWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'RESULTS'; payload: EggResultsPayload }
  | { type: 'COMPLETE'; payload: EggCompletion }
  | { type: 'ERROR'; message: string; category: EggErrorCategory; fatal: boolean };

/**
 * 結果ペイロード
 */
export interface EggResultsPayload {
  results: EnumeratedEggData[];
}

/**
 * 完了情報
 */
export interface EggCompletion {
  reason: 'max-count' | 'stopped' | 'error';
  processedCount: number;    // 実際に処理した個体数
  filteredCount: number;     // フィルター適用後の個体数
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type EggErrorCategory = 'VALIDATION' | 'WASM_INIT' | 'RUNTIME';

/**
 * 型ガード
 */
export function isEggWorkerResponse(data: unknown): data is EggWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return typeof obj.type === 'string' &&
    ['READY', 'RESULTS', 'COMPLETE', 'ERROR'].includes(obj.type);
}

// === デフォルト値 ===

/**
 * デフォルト生成条件
 */
export function createDefaultEggConditions(): EggGenerationConditions {
  return {
    hasNidoranFlag: false,
    everstone: { type: 'none' },
    usesDitto: false,
    femaleParentAbility: 0,
    masudaMethod: false,
    tid: 0,
    sid: 0,
    genderRatio: {
      threshold: 127,
      genderless: false,
    },
  };
}

/**
 * デフォルト親IV
 */
export function createDefaultParentsIVs(): ParentsIVs {
  return {
    male: [31, 31, 31, 31, 31, 31],
    female: [31, 31, 31, 31, 31, 31],
  };
}

/**
 * デフォルトフィルター
 */
export function createDefaultEggFilter(): EggIndividualFilter {
  return {
    ivRanges: [
      { min: 0, max: IV_UNKNOWN },
      { min: 0, max: IV_UNKNOWN },
      { min: 0, max: IV_UNKNOWN },
      { min: 0, max: IV_UNKNOWN },
      { min: 0, max: IV_UNKNOWN },
      { min: 0, max: IV_UNKNOWN },
    ],
  };
}

/**
 * デフォルトパラメータ (HEX)
 */
export function createDefaultEggParamsHex(): EggGenerationParamsHex {
  return {
    baseSeedHex: '0',
    userOffsetHex: '0',
    count: 100,
    conditions: createDefaultEggConditions(),
    parents: createDefaultParentsIVs(),
    filter: null,
    filterDisabled: false,
    considerNpcConsumption: false,
    gameMode: EggGameMode.BwContinue,
    seedSourceMode: 'lcg',
    bootTiming: createDefaultEggBootTimingDraft(),
  };
}
