/**
 * 型定義: レポ針検索
 */

import type { NumericRange } from './profile';
import type { Hardware, ROMRegion, ROMVersion } from './rom';

export type ReportNeedleSearchMode = 'startup' | 'initial-seed';

export type ReportNeedleSearchStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'completed'
  | 'error';

export interface ReportNeedleSearchDraft {
  mode: ReportNeedleSearchMode;
  needleValue: string;
  timer0Range: NumericRange;
  vcountRange: NumericRange;
  advanceRange: { start: number; end: number };
  startup: {
    timestampIso: string;
    keyMask: number;
  };
  initialSeed: {
    seedHex: string;
  };
}

export interface ReportNeedleSearchResult {
  consumptionIndex: number;
  timer0: number;
  vcount: number;
  initialSeed: string;
  currentSeed: string;
}

// ========================================
// Worker通信用型定義
// ========================================

/**
 * プロファイル情報（Boot-Timingモード用）
 */
export interface ReportNeedleProfileParams {
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress: number[];
}

/**
 * Worker検索パラメータ
 */
export interface ReportNeedleSearchParams {
  mode: ReportNeedleSearchMode;
  needleValue: string;
  timer0Range: NumericRange;
  vcountRange: NumericRange;
  advanceRange: { start: number; end: number };
  /** LCG Seedモード用 */
  seedHex?: string;
  /** Boot-Timingモード用 */
  timestampIso?: string;
  keyMask?: number;
  profile?: ReportNeedleProfileParams;
}

/**
 * Worker → UIへのメッセージ
 */
export type ReportNeedleWorkerResponse =
  | { type: 'READY' }
  | { type: 'RESULTS'; results: ReportNeedleSearchResult[] }
  | { type: 'COMPLETE'; totalFound: number }
  | { type: 'ERROR'; message: string }
  | { type: 'PAUSED' }
  | { type: 'RESUMED' };

/**
 * UI → Workerへのメッセージ
 */
export type ReportNeedleWorkerRequest =
  | { type: 'START'; params: ReportNeedleSearchParams }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'STOP' };
