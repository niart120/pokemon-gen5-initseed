/**
 * 型定義: レポ針検索
 */

import type { NumericRange } from './profile';

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
}
