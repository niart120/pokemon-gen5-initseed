/**
 * MT Seed 32bit全探索 検索マネージャー
 *
 * GPU/CPUの選択、Worker管理、結果集約を担当する。
 * 既存のWorkerManagerパターンに従い、複数Workerを並列管理する。
 */
import type {
  MtSeedSearchJob,
  MtSeedSearchResult,
  MtSeedMatch,
  MtSeedSearchProgress,
  MtSeedSearchCompletion,
  MtSeedSearchWorkerRequest,
  MtSeedSearchWorkerResponse,
  IvSearchFilter,
  IvCode,
  CpuJobPlannerConfig,
  GpuJobPlannerConfig,
} from '@/types/mt-seed-search';
import { isMtSeedSearchWorkerResponse, FULL_SEARCH_RANGE } from '@/types/mt-seed-search';
import { generateIvCodes } from './iv-code-generator';
import { planCpuJobs } from './job-planner-cpu';
import { planGpuJobs, RECOMMENDED_WORKGROUP_SIZE } from './job-planner-gpu';
import { isMtSeedSearchGpuAvailable } from './mt-seed-search-gpu-engine';

// === 型定義 ===

export type SearchMode = 'auto' | 'gpu' | 'cpu';

export interface MtSeedSearchManagerConfig {
  /** 検索モード: 'auto' | 'gpu' | 'cpu' */
  mode?: SearchMode;

  /** CPU使用時のWorker数（デフォルト: navigator.hardwareConcurrency） */
  cpuWorkerCount?: number;

  /** 進捗コールバック */
  onProgress?: (progress: AggregatedProgress) => void;

  /** 結果コールバック（ストリーミング） */
  onResult?: (result: MtSeedSearchResult) => void;

  /** 完了コールバック */
  onComplete?: (completion: SearchCompletion) => void;

  /** エラーコールバック */
  onError?: (error: SearchError) => void;
}

export interface SearchParams {
  /** MT消費数 */
  mtAdvances: number;

  /** 検索フィルター */
  filter: IvSearchFilter;
}

export interface AggregatedProgress {
  /** 処理済みSeed数 */
  processedCount: number;

  /** 全体のSeed数 */
  totalCount: number;

  /** 発見数 */
  matchesFound: number;

  /** 経過時間（ミリ秒） */
  elapsedMs: number;

  /** 進捗率（0-100） */
  progressPercent: number;

  /** 推定残り時間（ミリ秒） */
  estimatedRemainingMs: number;

  /** 使用モード */
  mode: 'gpu' | 'cpu';

  /** アクティブWorker数 */
  activeWorkers: number;
}

export interface SearchCompletion {
  /** 完了理由 */
  reason: 'finished' | 'stopped' | 'error';

  /** 処理済みSeed数 */
  totalProcessed: number;

  /** 発見数 */
  totalMatches: number;

  /** 経過時間（ミリ秒） */
  elapsedMs: number;

  /** 使用モード */
  mode: 'gpu' | 'cpu';

  /** 全マッチ結果 */
  matches: MtSeedMatch[];
}

export interface SearchError {
  message: string;
  category: 'VALIDATION' | 'WASM_INIT' | 'GPU_INIT' | 'RUNTIME';
  fatal: boolean;
}

export interface MtSeedSearchManager {
  /**
   * 検索を開始
   */
  start(params: SearchParams): Promise<void>;

  /**
   * 検索を一時停止
   */
  pause(): void;

  /**
   * 検索を再開
   */
  resume(): void;

  /**
   * 検索を停止
   */
  stop(): void;

  /**
   * 検索中かどうか
   */
  isRunning(): boolean;

  /**
   * GPU利用可能かどうか
   */
  isGpuAvailable(): boolean;

  /**
   * リソースを解放
   */
  dispose(): void;
}

// === 内部状態 ===

interface WorkerState {
  worker: Worker;
  jobId: number;
  status: 'idle' | 'running' | 'paused';
}

interface ManagerState {
  workers: WorkerState[];
  jobs: MtSeedSearchJob[];
  pendingJobIndex: number;
  completedJobs: number;
  running: boolean;
  paused: boolean;
  mode: 'gpu' | 'cpu';
  startTime: number;
  processedCounts: Map<number, number>;
  matchesCounts: Map<number, number>;
  allMatches: MtSeedMatch[];
  totalCount: number;
}

// === ファクトリ関数 ===

export function createMtSeedSearchManager(
  config?: MtSeedSearchManagerConfig
): MtSeedSearchManager {
  const mode = config?.mode ?? 'auto';
  const cpuWorkerCount = config?.cpuWorkerCount ?? (navigator.hardwareConcurrency || 4);

  const state: ManagerState = {
    workers: [],
    jobs: [],
    pendingJobIndex: 0,
    completedJobs: 0,
    running: false,
    paused: false,
    mode: 'cpu',
    startTime: 0,
    processedCounts: new Map(),
    matchesCounts: new Map(),
    allMatches: [],
    totalCount: 0,
  };

  const isGpuAvailable = (): boolean => {
    return isMtSeedSearchGpuAvailable();
  };

  const resolveMode = (): 'gpu' | 'cpu' => {
    if (mode === 'gpu') {
      return isGpuAvailable() ? 'gpu' : 'cpu';
    }
    if (mode === 'cpu') {
      return 'cpu';
    }
    // auto: GPUが利用可能ならGPU、そうでなければCPU
    return isGpuAvailable() ? 'gpu' : 'cpu';
  };

  const createWorker = (useGpu: boolean): Worker => {
    if (useGpu) {
      return new Worker(
        new URL('@/workers/mt-seed-search-worker-gpu.ts', import.meta.url),
        { type: 'module' }
      );
    }
    return new Worker(
      new URL('@/workers/mt-seed-search-worker-cpu.ts', import.meta.url),
      { type: 'module' }
    );
  };

  const handleWorkerMessage = (
    workerState: WorkerState,
    event: MessageEvent<unknown>
  ): void => {
    if (!isMtSeedSearchWorkerResponse(event.data)) {
      return;
    }

    const response = event.data as MtSeedSearchWorkerResponse;

    switch (response.type) {
      case 'READY':
        // Workerが準備完了、次のジョブを割り当て
        assignNextJob(workerState);
        break;

      case 'PROGRESS':
        handleProgress(workerState, response.payload);
        break;

      case 'RESULTS':
        handleResults(response.payload);
        break;

      case 'COMPLETE':
        handleComplete(workerState, response.payload);
        break;

      case 'ERROR':
        handleError(response);
        break;
    }
  };

  const assignNextJob = (workerState: WorkerState): void => {
    if (state.paused || !state.running) {
      workerState.status = 'idle';
      return;
    }

    if (state.pendingJobIndex >= state.jobs.length) {
      workerState.status = 'idle';
      checkAllComplete();
      return;
    }

    const job = state.jobs[state.pendingJobIndex];
    state.pendingJobIndex++;

    workerState.jobId = job.jobId;
    workerState.status = 'running';

    const request: MtSeedSearchWorkerRequest = { type: 'START', job };
    workerState.worker.postMessage(request);
  };

  const handleProgress = (
    workerState: WorkerState,
    progress: MtSeedSearchProgress
  ): void => {
    state.processedCounts.set(workerState.jobId, progress.processedCount);
    state.matchesCounts.set(workerState.jobId, progress.matchesFound);

    emitProgress();
  };

  const handleResults = (result: MtSeedSearchResult): void => {
    state.allMatches.push(...result.matches);
    config?.onResult?.(result);
  };

  const handleComplete = (
    workerState: WorkerState,
    _completion: MtSeedSearchCompletion
  ): void => {
    state.completedJobs++;
    workerState.status = 'idle';

    // 次のジョブを割り当て
    assignNextJob(workerState);
  };

  const handleError = (response: { message: string; category: string }): void => {
    const error: SearchError = {
      message: response.message,
      category: response.category as SearchError['category'],
      fatal: true,
    };
    config?.onError?.(error);
  };

  const emitProgress = (): void => {
    let processedCount = 0;
    let matchesFound = 0;

    for (const count of state.processedCounts.values()) {
      processedCount += count;
    }
    for (const count of state.matchesCounts.values()) {
      matchesFound += count;
    }

    const elapsedMs = performance.now() - state.startTime;
    const progressPercent = state.totalCount > 0
      ? (processedCount / state.totalCount) * 100
      : 0;

    const estimatedRemainingMs = processedCount > 0
      ? (elapsedMs / processedCount) * (state.totalCount - processedCount)
      : 0;

    const activeWorkers = state.workers.filter((w) => w.status === 'running').length;

    const progress: AggregatedProgress = {
      processedCount,
      totalCount: state.totalCount,
      matchesFound,
      elapsedMs,
      progressPercent,
      estimatedRemainingMs,
      mode: state.mode,
      activeWorkers,
    };

    config?.onProgress?.(progress);
  };

  const checkAllComplete = (): void => {
    const allIdle = state.workers.every((w) => w.status === 'idle');
    const allJobsAssigned = state.pendingJobIndex >= state.jobs.length;

    if (allIdle && allJobsAssigned && state.running) {
      finishSearch('finished');
    }
  };

  const finishSearch = (reason: 'finished' | 'stopped' | 'error'): void => {
    state.running = false;

    let totalProcessed = 0;
    for (const count of state.processedCounts.values()) {
      totalProcessed += count;
    }

    const completion: SearchCompletion = {
      reason,
      totalProcessed,
      totalMatches: state.allMatches.length,
      elapsedMs: performance.now() - state.startTime,
      mode: state.mode,
      matches: state.allMatches,
    };

    config?.onComplete?.(completion);
  };

  const start = async (params: SearchParams): Promise<void> => {
    if (state.running) {
      return;
    }

    // IVコードを生成
    const ivCodeResult = generateIvCodes(params.filter);
    if (!ivCodeResult.success) {
      const error: SearchError = {
        message: `検索条件が広すぎます。個体値の組み合わせが${ivCodeResult.estimatedCount.toLocaleString()}件あります（上限: 1024件）。条件を絞り込んでください。`,
        category: 'VALIDATION',
        fatal: true,
      };
      config?.onError?.(error);
      return;
    }

    const ivCodes: IvCode[] = ivCodeResult.ivCodes;

    // モードを決定
    state.mode = resolveMode();

    // ジョブを計画
    const jobPlanConfig = {
      fullRange: FULL_SEARCH_RANGE,
      ivCodes,
      mtAdvances: params.mtAdvances,
    };

    if (state.mode === 'gpu') {
      // GPU: 1ジョブ（GPU内部でチャンク分割）
      const gpuConfig: GpuJobPlannerConfig = {
        ...jobPlanConfig,
        deviceLimits: {
          maxComputeWorkgroupsPerDimension: 65535,
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
        },
        workgroupSize: RECOMMENDED_WORKGROUP_SIZE,
      };
      const plan = planGpuJobs(gpuConfig);
      state.jobs = plan.jobs;
    } else {
      // CPU: Worker数分のジョブに分割
      const cpuConfig: CpuJobPlannerConfig = {
        ...jobPlanConfig,
        workerCount: cpuWorkerCount,
      };
      const plan = planCpuJobs(cpuConfig);
      state.jobs = plan.jobs;
    }

    // 状態をリセット
    state.pendingJobIndex = 0;
    state.completedJobs = 0;
    state.running = true;
    state.paused = false;
    state.startTime = performance.now();
    state.processedCounts.clear();
    state.matchesCounts.clear();
    state.allMatches = [];
    state.totalCount = FULL_SEARCH_RANGE.end - FULL_SEARCH_RANGE.start + 1;

    // Workerを作成
    const workerCount = state.mode === 'gpu' ? 1 : cpuWorkerCount;
    for (let i = 0; i < workerCount; i++) {
      const worker = createWorker(state.mode === 'gpu');
      const workerState: WorkerState = {
        worker,
        jobId: -1,
        status: 'idle',
      };

      worker.onmessage = (event) => handleWorkerMessage(workerState, event);
      worker.onerror = (event) => {
        const error: SearchError = {
          message: event.message || 'Worker error',
          category: 'RUNTIME',
          fatal: true,
        };
        config?.onError?.(error);
      };

      state.workers.push(workerState);
    }
  };

  const pause = (): void => {
    if (!state.running || state.paused) {
      return;
    }

    state.paused = true;
    for (const workerState of state.workers) {
      const request: MtSeedSearchWorkerRequest = { type: 'PAUSE' };
      workerState.worker.postMessage(request);
      if (workerState.status === 'running') {
        workerState.status = 'paused';
      }
    }
  };

  const resume = (): void => {
    if (!state.running || !state.paused) {
      return;
    }

    state.paused = false;
    for (const workerState of state.workers) {
      if (workerState.status === 'paused') {
        const request: MtSeedSearchWorkerRequest = { type: 'RESUME' };
        workerState.worker.postMessage(request);
        workerState.status = 'running';
      } else if (workerState.status === 'idle') {
        // 待機中のWorkerに新しいジョブを割り当て
        assignNextJob(workerState);
      }
    }
  };

  const stop = (): void => {
    if (!state.running) {
      return;
    }

    for (const workerState of state.workers) {
      const request: MtSeedSearchWorkerRequest = { type: 'STOP' };
      workerState.worker.postMessage(request);
    }

    finishSearch('stopped');
  };

  const isRunning = (): boolean => state.running;

  const dispose = (): void => {
    stop();
    for (const workerState of state.workers) {
      workerState.worker.terminate();
    }
    state.workers = [];
  };

  return {
    start,
    pause,
    resume,
    stop,
    isRunning,
    isGpuAvailable,
    dispose,
  };
}
