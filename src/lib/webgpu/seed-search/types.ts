import type { SearchConditions, InitialSeedResult } from '@/types/search';
import type { WorkerProgressMessage } from '@/types/callbacks';
import type { SearchTimePlan } from '@/lib/search/time/time-plan';

export type WebGpuRunnerProgress = WorkerProgressMessage;

export interface WebGpuRunnerCallbacks {
  onProgress: (progress: WebGpuRunnerProgress) => void;
  onResult: (result: InitialSeedResult) => void;
  onComplete: (message: string) => void;
  onError: (error: string, errorCode?: string) => void;
  onPaused: () => void;
  onResumed: () => void;
  onStopped: (message: string, finalProgress: WebGpuRunnerProgress) => void;
}

export interface SeedSearchJobSegment {
  id: string;
  keyCode: number;
  timer0: number;
  vcount: number;
  messageCount: number;
  baseSecondOffset: number;
  globalMessageOffset: number;
  workgroupCount: number;
  uniformWords: Uint32Array;
}

export interface SeedSearchJobSummary {
  totalMessages: number;
  totalSegments: number;
  targetSeedCount: number;
  rangeSeconds: number;
}

export interface SeedSearchJobLimits {
  workgroupSize: number;
  maxWorkgroupsPerDispatch: number;
  candidateCapacityPerDispatch: number;
  maxMessagesPerDispatch: number;
}

export interface SeedSearchJobOptions {
  limits: SeedSearchJobLimits;
}

export interface SeedSearchJob {
  segments: SeedSearchJobSegment[];
  targetSeeds: Uint32Array;
  timePlan: SearchTimePlan;
  summary: SeedSearchJobSummary;
  limits: SeedSearchJobLimits;
  conditions: SearchConditions;
}
