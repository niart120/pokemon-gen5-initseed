/**
 * Search related configs and result types
 */

import type { ROMVersion, ROMRegion, Hardware } from './rom';
import type { KeyName } from '@/lib/utils/key-input';

export interface TimeFieldRange {
  start: number;
  end: number;
}

export interface DailyTimeRange {
  hour: TimeFieldRange;
  minute: TimeFieldRange;
  second: TimeFieldRange;
}

/**
 * 日付範囲（Boot Timing検索共通）
 */
export interface DateRange {
  startYear: number;
  startMonth: number;
  startDay: number;
  endYear: number;
  endMonth: number;
  endDay: number;
}

/**
 * 起動条件情報（Boot Timing検索結果共通）
 */
export interface BootCondition {
  /** 起動日時 */
  datetime: Date;

  /** Timer0値 */
  timer0: number;

  /** VCount値 */
  vcount: number;

  /** キーコード (XOR 0x2FFF後) */
  keyCode: number;

  /** キー入力名リスト */
  keyInputNames: KeyName[];

  /** MACアドレス */
  macAddress: readonly [number, number, number, number, number, number];
}

export interface SearchConditions {
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  
  timer0VCountConfig: {
    useAutoConfiguration: boolean;
    timer0Range: {
      min: number;
      max: number;
    };
    vcountRange: {
      min: number;
      max: number;
    };
  };
  timeRange: DailyTimeRange;
  
  dateRange: {
    startYear: number;
    endYear: number;
    startMonth: number;
    endMonth: number;
    startDay: number;
    endDay: number;
    startHour: number;
    endHour: number;
    startMinute: number;
    endMinute: number;
    startSecond: number;
    endSecond: number;
  };
  
  keyInput: number;
  macAddress: number[];
}

export interface InitialSeedResult {
  seed: number;
  datetime: Date;
  timer0: number;
  vcount: number;
  keyCode: number | null;
  keyInputNames?: KeyName[];
  conditions: SearchConditions;
  message: number[];
  sha1Hash: string;
  lcgSeed: bigint;
  isMatch: boolean;
}

// NOTE: Export等で使用する軽量結果。InitialSeedResult との整合性のためフィールド名を `datetime` に統一。
export interface SearchResult {
  seed: number;
  datetime: Date;
  timer0: number;
  vcount: number;
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress?: number[];
  keyInput?: number;
  keyCode?: number | null;
  message?: number[];
  hash?: string;
}

export interface TargetSeedList {
  seeds: number[];
}

export interface SeedInputFormat {
  rawInput: string;
  validSeeds: number[];
  errors: {
    line: number;
    value: string;
    error: string;
  }[];
}

export interface SearchProgress {
  isRunning: boolean;
  currentStep: number;
  totalSteps: number;
  currentDateTime: Date | null;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  matchesFound: number;
  canPause: boolean;
  isPaused: boolean;
}

export interface SearchPreset {
  id: string;
  name: string;
  description?: string;
  conditions: SearchConditions;
  createdAt: Date;
  lastUsed?: Date;
}
