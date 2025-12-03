/**
 * ID Adjustment search feature types
 *
 * ID調整（表ID/裏IDを持つ初期Seedを検索）機能の型定義
 */

import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import type { DailyTimeRange, DateRange, BootCondition } from '@/types/search';
import type { DomainShinyType, DomainGameMode } from '@/types/domain';

// Re-export shared types for convenience
export type { DateRange, BootCondition };

// === 検索パラメータ ===

/**
 * ID調整検索パラメータ
 * MtSeedBootTimingSearchParams と同様の構造に従う
 */
export interface IdAdjustmentSearchParams {
  // === 起動時間パラメータ（boot-timing-search共通） ===

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

  /** キー入力マスク (ビットマスク) - IdAdjustmentCardから入力 */
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

  // === ID調整固有パラメータ ===

  /** 検索対象の表ID（必須、0〜65535） */
  targetTid: number;

  /** 検索対象の裏ID（任意、0〜65535 または null） */
  targetSid: number | null;

  /** 色違いにしたい個体のPID（任意、0〜0xFFFFFFFF または null） */
  shinyPid: number | null;

  /** ゲームモード */
  gameMode: DomainGameMode;

  // === 制限 ===

  /** 結果上限数 (全体) */
  maxResults: number;
}

// === 検索結果 ===

/**
 * ID調整検索結果1件
 */
export interface IdAdjustmentSearchResult {
  /** 起動条件（boot-timing-search共通） */
  boot: BootCondition;

  /** LCG Seed (16進文字列) */
  lcgSeedHex: string;

  /** 算出された表ID */
  tid: number;

  /** 算出された裏ID */
  sid: number;

  /**
   * 色違いタイプ（shinyPid指定時のみ有効）
   * - 0: Normal（色違いではない）
   * - 1: Square（四角い色違い、最レア）
   * - 2: Star（星形色違い）
   */
  shinyType?: DomainShinyType;
}

/**
 * WASM結果の型定義
 */
export interface WasmIdAdjustmentSearchResult {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  timer0: number;
  vcount: number;
  keyCode: number;
  lcgSeedHex: string;
  lcgSeedHigh: number;
  lcgSeedLow: number;
  tid: number;
  sid: number;
  shinyType: number;
}

// === Worker通信型 ===

/**
 * Worker リクエスト
 */
export type IdAdjustmentWorkerRequest =
  | {
      type: 'START_SEARCH';
      params: IdAdjustmentSearchParams;
      requestId?: string;
    }
  | {
      type: 'PAUSE';
      requestId?: string;
    }
  | {
      type: 'RESUME';
      requestId?: string;
    }
  | {
      type: 'STOP';
      requestId?: string;
    };

/**
 * Worker レスポンス
 */
export type IdAdjustmentWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: IdAdjustmentProgress }
  | { type: 'RESULTS'; payload: IdAdjustmentResultsPayload }
  | { type: 'COMPLETE'; payload: IdAdjustmentCompletion }
  | {
      type: 'ERROR';
      message: string;
      category: IdAdjustmentErrorCategory;
      fatal: boolean;
    };

/**
 * 進捗情報
 */
export interface IdAdjustmentProgress {
  /** 処理済み起動条件の組み合わせ数（完了セグメント数） */
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

  /**
   * 処理済み秒数（検索範囲の秒数単位）
   * 処理速度計算・進捗表示用。
   */
  processedSeconds?: number;
}

/**
 * 結果ペイロード（バッチ送信用）
 */
export interface IdAdjustmentResultsPayload {
  results: IdAdjustmentSearchResult[];
  batchIndex: number;
}

/**
 * 完了情報
 */
export interface IdAdjustmentCompletion {
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
export type IdAdjustmentErrorCategory =
  | 'VALIDATION' // パラメータ検証エラー
  | 'WASM_INIT' // WASM初期化エラー
  | 'RUNTIME' // 実行時エラー
  | 'ABORTED'; // 中断

/**
 * 完了理由ラベル
 */
export const ID_ADJUSTMENT_COMPLETION_REASON_LABELS: Record<
  IdAdjustmentCompletion['reason'],
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
export function isIdAdjustmentWorkerResponse(
  data: unknown
): data is IdAdjustmentWorkerResponse {
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
export function validateIdAdjustmentSearchParams(
  params: IdAdjustmentSearchParams
): string[] {
  const errors: string[] = [];

  // 日付範囲検証
  const { dateRange } = params;
  const startDate = new Date(
    dateRange.startYear,
    dateRange.startMonth - 1,
    dateRange.startDay
  );
  const endDate = new Date(
    dateRange.endYear,
    dateRange.endMonth - 1,
    dateRange.endDay
  );
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

  // 表ID検証
  if (!Number.isInteger(params.targetTid) || params.targetTid < 0 || params.targetTid > 65535) {
    errors.push('表IDは0〜65535の範囲で入力してください');
  }

  // 裏ID検証（任意）
  if (params.targetSid !== null) {
    if (!Number.isInteger(params.targetSid) || params.targetSid < 0 || params.targetSid > 65535) {
      errors.push('裏IDは0〜65535の範囲で入力してください');
    }
  }

  // PID検証（任意）
  if (params.shinyPid !== null) {
    if (!Number.isInteger(params.shinyPid) || params.shinyPid < 0 || params.shinyPid > 0xffffffff) {
      errors.push('PIDは0〜FFFFFFFFの16進数で入力してください');
    }
  }

  // 時刻範囲検証
  const { timeRange } = params;
  if (timeRange.hour.start > timeRange.hour.end) {
    errors.push('時の開始は終了以前を指定してください');
  }
  if (timeRange.minute.start > timeRange.minute.end) {
    errors.push('分の開始は終了以前を指定してください');
  }
  if (timeRange.second.start > timeRange.second.end) {
    errors.push('秒の開始は終了以前を指定してください');
  }

  // GameMode検証（「続きから」モードはID調整不可）
  if (
    params.gameMode === 2 || // BwContinue
    params.gameMode === 6 || // Bw2ContinueWithMemoryLink
    params.gameMode === 7    // Bw2ContinueNoMemoryLink
  ) {
    errors.push('ID調整には「始めから」モードを選択してください');
  }

  // 結果上限検証
  if (!Number.isInteger(params.maxResults) || params.maxResults < 1) {
    errors.push('結果上限数は1以上の整数である必要があります');
  }

  return errors;
}

// === デフォルト値生成 ===

/**
 * デフォルトの検索パラメータを生成
 */
export function createDefaultIdAdjustmentSearchParams(): IdAdjustmentSearchParams {
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
    timer0Range: { min: 0x1000, max: 0x1100 },
    vcountRange: { min: 0x50, max: 0x60 },
    keyInputMask: 0,
    macAddress: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00] as const,
    hardware: 'DS_LITE',
    romVersion: 'W',
    romRegion: 'JPN',
    timeRange: {
      hour: { start: 0, end: 23 },
      minute: { start: 0, end: 59 },
      second: { start: 0, end: 59 },
    },
    targetTid: 0,
    targetSid: null,
    shinyPid: null,
    gameMode: 1, // BwNewGameNoSave
    maxResults: 1000,
  };
}
