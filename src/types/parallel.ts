/**
 * Parallel worker related types
 */

export interface WorkerChunk {
  workerId: number;
  startDateTime: Date;
  endDateTime: Date;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  estimatedOperations: number;
}

export interface ParallelSearchSettings {
  maxWorkers: number;
  chunkStrategy: 'time-based' | 'hybrid' | 'auto';
}

export interface AggregatedProgress {
  totalCurrentStep: number;
  totalSteps: number;
  totalElapsedTime: number;
  totalEstimatedTimeRemaining: number;
  totalMatchesFound: number;
  activeWorkers: number;
  completedWorkers: number;
  workerProgresses: Map<number, WorkerProgress>;
}

export interface WorkerProgress {
  workerId: number;
  currentStep: number;
  totalSteps: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  matchesFound: number;
  currentDateTime?: Date;
  status: 'initializing' | 'running' | 'paused' | 'completed' | 'error';
}

export interface ParallelWorkerRequest {
  type: 'START_SEARCH' | 'PAUSE_SEARCH' | 'RESUME_SEARCH' | 'STOP_SEARCH' | 'PING';
  workerId: number;
  conditions?: import('./search').SearchConditions;
  targetSeeds?: number[];
  chunk?: WorkerChunk;
}

export interface ParallelWorkerResponse {
  type: 'PROGRESS' | 'RESULT' | 'COMPLETE' | 'ERROR' | 'PAUSED' | 'RESUMED' | 'STOPPED' | 'READY' | 'INITIALIZED';
  workerId: number;
  progress?: import('./callbacks').WorkerProgressMessage;
  result?: import('./search').InitialSeedResult & { datetime: string };
  error?: string;
  message?: string;
  chunkProgress?: {
    processed: number;
    total: number;
  };
}
