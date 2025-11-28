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
  /** 全体の進捗パーセント（0-100、秒数ベース） */
  progressPercent?: number;
  /** 全Worker合計の処理済み秒数（処理速度計算用） */
  totalProcessedSeconds?: number;
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
  /** 進捗パーセント（0-100） */
  progressPercent?: number;
  /** 処理済み秒数（処理速度計算用） */
  processedSeconds?: number;
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
