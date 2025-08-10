/**
 * Shared callback and progress types for search workers and managers
 */

// Stopwatch-like timer state used both in workers and managers
export interface TimerState {
  cumulativeRunTime: number; // ms
  segmentStartTime: number;  // epoch ms
  isPaused: boolean;
}

// Progress payload coming from a Worker (uses ISO string for date)
export interface WorkerProgressMessage {
  currentStep: number;
  totalSteps: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  matchesFound: number;
  currentDateTime?: string;
}

// Progress object delivered to UI callbacks on main thread (Date object)
export interface SingleWorkerProgress {
  currentStep: number;
  totalSteps: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  matchesFound: number;
  currentDateTime?: Date;
}

// Callbacks for single-worker search flow
export interface SingleWorkerSearchCallbacks<ResultType> {
  onProgress: (progress: SingleWorkerProgress) => void;
  onResult: (result: ResultType) => void;
  onComplete: (message: string) => void;
  onError: (error: string) => void;
  onPaused: () => void;
  onResumed: () => void;
  onStopped: () => void;
}

// Callbacks for multi-worker (aggregated) progress
export interface MultiWorkerSearchCallbacks<AggregatedProgressType, ResultType> {
  onProgress: (progress: AggregatedProgressType) => void;
  onResult: (result: ResultType) => void;
  onComplete: (message: string) => void;
  onError: (error: string) => void;
  onPaused: () => void;
  onResumed: () => void;
  onStopped: () => void;
}
