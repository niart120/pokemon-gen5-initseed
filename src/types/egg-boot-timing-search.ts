/**
 * Egg boot timing search feature types
 * Based on: spec/agent/pr_egg_boot_timing_search/SPECIFICATION.md
 */

import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import type { DailyTimeRange, DateRange, BootCondition } from '@/types/search';
import type {
  EggGenerationConditions,
  ParentsIVs,
  EggIndividualFilter,
  EnumeratedEggData,
} from '@/types/egg';
import { EggGameMode, createDefaultEggFilter } from '@/types/egg';

// Re-export shared types for convenience
export type { DateRange, BootCondition };

// === 検索パラメータ ===

/**
 * 孵化乱数起動時間検索パラメータ
 */
export interface EggBootTimingSearchParams {
  // === 起動時間パラメータ ===

  /** 日付範囲 */
  dateRange: DateRange;

  /**
   * 検索範囲（秒）
   * チャンク分割時にManagerが設定。
   * 指定されている場合、dateRangeからの再計算をスキップする。
   */
  rangeSeconds?: number;

  /** Timer0範囲 */
  timer0Range: {
    min: number; // 0x0000-0xFFFF
    max: number;
  };

  /** VCount範囲 */
  vcountRange: {
    min: number; // 0x00-0xFF
    max: number;
  };

  /** キー入力マスク (ビットマスク) */
  keyInputMask: number;

  /** MACアドレス (6バイト) */
  macAddress: readonly [number, number, number, number, number, number];

  /** ハードウェア */
  hardware: Hardware;

  /** ROMバージョン */
  romVersion: ROMVersion;

  /** ROM地域 */
  romRegion: ROMRegion;

  /** 時刻範囲フィルター（1日の中で検索する時間帯） */
  timeRange: DailyTimeRange;

  // === 孵化条件パラメータ ===

  /** 生成条件 */
  conditions: EggGenerationConditions;

  /** 親個体値 */
  parents: ParentsIVs;

  /** 個体フィルター (null = フィルタなし) */
  filter: EggIndividualFilter | null;

  /** フィルター無効フラグ */
  filterDisabled: boolean;

  /** NPC消費考慮 */
  considerNpcConsumption: boolean;

  /** ゲームモード */
  gameMode: EggGameMode;

  // === 消費範囲 ===

  /** 開始advance */
  userOffset: number;

  /** 検索件数上限 (per seed) */
  advanceCount: number;

  // === 制限 ===

  /** 結果上限数 (全体) */
  maxResults: number;
}

// === 検索結果 ===

/**
 * 孵化乱数起動時間検索結果1件
 */
export interface EggBootTimingSearchResult {
  /** 起動条件 */
  boot: BootCondition;

  /** LCG Seed (16進文字列) */
  lcgSeedHex: string;

  /** 個体情報 */
  egg: EnumeratedEggData;

  /** 安定性フラグ */
  isStable: boolean;
}

/**
 * WASM結果の型定義
 */
export interface WasmEggBootTimingSearchResult {
  year: number;
  month: number;
  date: number;
  hour: number;
  minute: number;
  second: number;
  timer0: number;
  vcount: number;
  keyCode: number;
  lcgSeedHex: string;
  mtSeed: number;
  mtSeedHex: string;
  advance: bigint | number;
  isStable: boolean;
  ivs: number[];
  nature: number;
  gender: number;
  ability: number;
  shiny: number;
  pid: number;
  hpType: number;
  hpPower: number;
  hpKnown: boolean;
}

// === Worker通信型 ===

/**
 * Worker リクエスト
 */
export type EggBootTimingWorkerRequest =
  | {
      type: 'START_SEARCH';
      params: EggBootTimingSearchParams;
      requestId?: string;
    }
  | {
      type: 'STOP';
      requestId?: string;
    };

/**
 * Worker レスポンス
 */
export type EggBootTimingWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: EggBootTimingProgress }
  | { type: 'RESULTS'; payload: EggBootTimingResultsPayload }
  | { type: 'COMPLETE'; payload: EggBootTimingCompletion }
  | {
      type: 'ERROR';
      message: string;
      category: EggBootTimingErrorCategory;
      fatal: boolean;
    };

/**
 * 進捗情報
 */
export interface EggBootTimingProgress {
  /** 処理済み起動条件の組み合わせ数 */
  processedCombinations: number;

  /** 総組み合わせ数 */
  totalCombinations: number;

  /** 見つかった結果数 */
  foundCount: number;

  /** 進捗率 (0-100) */
  progressPercent: number;

  /** 経過時間 (ms) */
  elapsedMs: number;

  /** 推定残り時間 (ms) */
  estimatedRemainingMs: number;
}

/**
 * 結果ペイロード（バッチ送信用）
 */
export interface EggBootTimingResultsPayload {
  results: EggBootTimingSearchResult[];
  batchIndex: number;
}

/**
 * 完了情報
 */
export interface EggBootTimingCompletion {
  /** 完了理由 */
  reason: 'completed' | 'stopped' | 'max-results' | 'error';

  /** 処理した起動条件の組み合わせ数 */
  processedCombinations: number;

  /** 総組み合わせ数 */
  totalCombinations: number;

  /** 見つかった結果数 */
  resultsCount: number;

  /** 経過時間 (ms) */
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type EggBootTimingErrorCategory =
  | 'VALIDATION' // パラメータ検証エラー
  | 'WASM_INIT' // WASM初期化エラー
  | 'RUNTIME' // 実行時エラー
  | 'ABORTED'; // 中断

/**
 * 完了理由ラベル
 */
export const COMPLETION_REASON_LABELS: Record<
  EggBootTimingCompletion['reason'],
  string
> = {
  completed: '検索完了',
  stopped: 'ユーザー停止',
  'max-results': '結果上限到達',
  error: 'エラー終了',
};

// === 型ガード ===

/**
 * 型ガード
 */
export function isEggBootTimingWorkerResponse(
  data: unknown
): data is EggBootTimingWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as { type?: unknown };
  if (typeof obj.type !== 'string') return false;
  return ['READY', 'PROGRESS', 'RESULTS', 'COMPLETE', 'ERROR'].includes(
    obj.type
  );
}

// === バリデーション ===

/**
 * パラメータバリデーション
 */
export function validateEggBootTimingSearchParams(
  params: EggBootTimingSearchParams
): string[] {
  const errors: string[] = [];

  // 日付範囲検証
  const { dateRange } = params;
  const startDate = new Date(dateRange.startYear, dateRange.startMonth - 1, dateRange.startDay);
  const endDate = new Date(dateRange.endYear, dateRange.endMonth - 1, dateRange.endDay);
  if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
    errors.push('日付が無効です');
  }
  if (startDate > endDate) {
    errors.push('開始日は終了日以前である必要があります');
  }

  // Timer0検証
  if (params.timer0Range.min > params.timer0Range.max) {
    errors.push('Timer0の最小値は最大値以下である必要があります');
  }
  if (params.timer0Range.min < 0 || params.timer0Range.max > 0xffff) {
    errors.push('Timer0は0x0000-0xFFFFの範囲である必要があります');
  }

  // VCount検証
  if (params.vcountRange.min > params.vcountRange.max) {
    errors.push('VCountの最小値は最大値以下である必要があります');
  }
  if (params.vcountRange.min < 0 || params.vcountRange.max > 0xff) {
    errors.push('VCountは0x00-0xFFの範囲である必要があります');
  }

  // MACアドレス検証
  if (
    params.macAddress.length !== 6 ||
    params.macAddress.some((b) => b < 0 || b > 255)
  ) {
    errors.push('MACアドレスは6バイトの配列である必要があります');
  }

  // 時刻範囲検証
  const { hour, minute, second } = params.timeRange;
  if (hour.start > hour.end || hour.start < 0 || hour.end > 23) {
    errors.push('時の範囲が無効です');
  }
  if (minute.start > minute.end || minute.start < 0 || minute.end > 59) {
    errors.push('分の範囲が無効です');
  }
  if (second.start > second.end || second.start < 0 || second.end > 59) {
    errors.push('秒の範囲が無効です');
  }

  // 消費範囲検証
  if (params.userOffset < 0 || params.userOffset > Number.MAX_SAFE_INTEGER) {
    errors.push('開始advanceは0以上の整数である必要があります');
  }
  if (params.advanceCount < 1 || params.advanceCount > 1000000) {
    errors.push('検索件数は1-1000000の範囲である必要があります');
  }

  // 結果上限検証
  if (params.maxResults < 1 || params.maxResults > 100000) {
    errors.push('結果上限は1-100000の範囲である必要があります');
  }

  return errors;
}

/**
 * 計算量見積もり
 */
export function estimateSearchCombinations(
  params: EggBootTimingSearchParams
): number {
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;

  // キーコード数を概算（ビットマスクから）
  const keyCodeCount = Math.pow(2, countBits(params.keyInputMask));

  // 時刻範囲内の秒数
  const { hour, minute, second } = params.timeRange;
  const hourCount = hour.end - hour.start + 1;
  const minuteCount = minute.end - minute.start + 1;
  const secondCount = second.end - second.start + 1;
  const allowedSecondsPerDay = hourCount * minuteCount * secondCount;

  // 日数（日付範囲から計算）
  const { dateRange } = params;
  const startDate = new Date(dateRange.startYear, dateRange.startMonth - 1, dateRange.startDay);
  const endDate = new Date(dateRange.endYear, dateRange.endMonth - 1, dateRange.endDay);
  const days = Math.max(1, Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1);

  const effectiveSeconds = allowedSecondsPerDay * days;

  return effectiveSeconds * timer0Count * vcountCount * keyCodeCount;
}

function countBits(n: number): number {
  let count = 0;
  let value = n;
  while (value) {
    count += value & 1;
    value >>= 1;
  }
  return count;
}

// === デフォルト値 ===

/**
 * デフォルトパラメータ生成
 */
export function createDefaultEggBootTimingSearchParams(): EggBootTimingSearchParams {
  // Dynamic imports at runtime to avoid circular dependencies
  const defaultConditions: EggGenerationConditions = {
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

  const defaultParents: ParentsIVs = {
    male: [31, 31, 31, 31, 31, 31],
    female: [31, 31, 31, 31, 31, 31],
  };

  const now = new Date();

  return {
    dateRange: {
      startYear: now.getFullYear(),
      startMonth: now.getMonth() + 1,
      startDay: now.getDate(),
      endYear: now.getFullYear(),
      endMonth: now.getMonth() + 1,
      endDay: now.getDate(),
    },
    timer0Range: { min: 0x0c79, max: 0x0c7b },
    vcountRange: { min: 0x60, max: 0x60 },
    keyInputMask: 0,
    macAddress: [0, 0, 0, 0, 0, 0],
    hardware: 'DS',
    romVersion: 'B',
    romRegion: 'JPN',
    timeRange: {
      hour: { start: 0, end: 23 },
      minute: { start: 0, end: 59 },
      second: { start: 0, end: 59 },
    },
    conditions: defaultConditions,
    parents: defaultParents,
    filter: createDefaultEggFilter(),
    filterDisabled: false,
    considerNpcConsumption: false,
    gameMode: EggGameMode.BwContinue,
    userOffset: 0,
    advanceCount: 10,
    maxResults: 1000,
  };
}
