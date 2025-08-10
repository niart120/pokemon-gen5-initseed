/**
 * Search related configs and result types
 */

import type { ROMVersion, ROMRegion, Hardware } from './rom';

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
  conditions: SearchConditions;
  message: number[];
  sha1Hash: string;
  isMatch: boolean;
}

export interface SearchResult {
  seed: number;
  dateTime: Date;
  timer0: number;
  vcount: number;
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress?: number[];
  keyInput?: number;
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
