import type { SearchConditions, InitialSeedResult } from './search';
import type { WorkerProgressMessage } from './callbacks';

export interface WorkerRequest {
  type: 'START_SEARCH' | 'PAUSE_SEARCH' | 'RESUME_SEARCH' | 'STOP_SEARCH';
  conditions?: SearchConditions;
  targetSeeds?: number[];
}

export interface WorkerResponse {
  type: 'PROGRESS' | 'RESULT' | 'COMPLETE' | 'ERROR' | 'PAUSED' | 'RESUMED' | 'STOPPED' | 'READY';
  progress?: WorkerProgressMessage;
  result?: InitialSeedResult;
  error?: string;
  errorCode?: string;
  message?: string;
}
