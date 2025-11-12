import type { SearchConditions, InitialSeedResult } from '@/types/search';
import type { WorkerProgressMessage } from '@/types/callbacks';

export interface GpuSha1WorkloadConfig {
  startSecondsSince2000: number;
  rangeSeconds: number;
  timer0Min: number;
  timer0Max: number;
  timer0Count: number;
  vcountMin: number;
  vcountMax: number;
  vcountCount: number;
  totalMessages: number;
  hardwareType: number;
  macLower: number;
  data7Swapped: number;
  keyInputSwapped: number;
  nazoSwapped: Uint32Array;
  startYear: number;
  startDayOfYear: number;
  startSecondOfDay: number;
  startDayOfWeek: number;
}

export interface WebGpuSegment {
  index: number;
  baseOffset: number;
  timer0Min: number;
  timer0Max: number;
  timer0Count: number;
  vcount: number;
  rangeSeconds: number;
  totalMessages: number;
  keyCode: number;
  config: GpuSha1WorkloadConfig;
}

export interface WebGpuSearchContext {
  conditions: SearchConditions;
  startDate: Date;
  startTimestampMs: number;
  rangeSeconds: number;
  totalMessages: number;
  segments: WebGpuSegment[];
}

export interface WebGpuDispatchPlan {
  baseOffset: number;
  messageCount: number;
}

export interface WebGpuBatchPlan {
  maxMessagesPerDispatch: number;
  dispatches: WebGpuDispatchPlan[];
}

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

export interface WebGpuRunRequest {
  context: WebGpuSearchContext;
  targetSeeds: number[];
  callbacks: WebGpuRunnerCallbacks;
  signal?: AbortSignal;
}

export type WebGpuRunnerSpanKind =
  | 'planner.computePlan'
  | 'dispatch'
  | 'dispatch.submit'
  | 'dispatch.submit.encode'
  | 'dispatch.submit.wait'
  | 'dispatch.mapMatchCount'
  | 'dispatch.copyResults'
  | 'dispatch.copyResults.encode'
  | 'dispatch.copyResults.wait'
  | 'dispatch.mapResults'
  | 'dispatch.processMatches';

export interface WebGpuRunnerSpanContext {
  kind: WebGpuRunnerSpanKind;
  metadata: Record<string, unknown>;
}

export interface WebGpuRunnerInstrumentation {
  trace<T>(context: WebGpuRunnerSpanContext, operation: () => Promise<T>): Promise<T>;
}
