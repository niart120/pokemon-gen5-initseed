/**
 * Egg Boot Timing Multi Worker Manager - 並列Worker版
 * 孵化乱数起動時間検索の並列Worker管理
 */

import type {
  EggBootTimingSearchParams,
  EggBootTimingSearchResult,
  EggBootTimingWorkerResponse,
  EggBootTimingProgress,
} from '@/types/egg-boot-timing-search';
import { isEggBootTimingWorkerResponse } from '@/types/egg-boot-timing-search';
import {
  calculateEggBootTimingChunks,
  calculateBatchSize,
  getDefaultWorkerCount,
  type EggBootTimingWorkerChunk,
} from './boot-timing-chunk-calculator';

/**
 * Worker ごとの進捗状態
 */
interface WorkerProgress {
  workerId: number;
  currentStep: number;
  totalSteps: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  matchesFound: number;
  status: 'initializing' | 'running' | 'paused' | 'completed' | 'error';
}

/**
 * 集約された進捗状態
 */
export interface AggregatedEggBootTimingProgress {
  totalCurrentStep: number;
  totalSteps: number;
  totalElapsedTime: number;
  totalEstimatedTimeRemaining: number;
  totalMatchesFound: number;
  activeWorkers: number;
  completedWorkers: number;
  workerProgresses: Map<number, WorkerProgress>;
}

/**
 * コールバック定義
 */
export interface EggBootTimingMultiWorkerCallbacks {
  onProgress: (progress: AggregatedEggBootTimingProgress) => void;
  onResult: (result: EggBootTimingSearchResult) => void;
  onComplete: (message: string) => void;
  onError: (error: string) => void;
  onPaused?: () => void;
  onResumed?: () => void;
  onStopped?: () => void;
}

/**
 * タイマー状態（一時停止対応）
 */
interface TimerState {
  cumulativeRunTime: number;
  segmentStartTime: number;
  isPaused: boolean;
}

/**
 * 並列 Worker 管理システム
 */
export class EggBootTimingMultiWorkerManager {
  private workers: Map<number, Worker> = new Map();
  private workerProgresses: Map<number, WorkerProgress> = new Map();
  private activeChunks: Map<number, EggBootTimingWorkerChunk> = new Map();
  private results: EggBootTimingSearchResult[] = [];
  private completedWorkers = 0;
  private callbacks: EggBootTimingMultiWorkerCallbacks | null = null;
  private searchRunning = false;
  private progressUpdateTimer: ReturnType<typeof setInterval> | null = null;
  private lastProgressCheck: Map<number, number> = new Map();

  private timerState: TimerState = {
    cumulativeRunTime: 0,
    segmentStartTime: 0,
    isPaused: false,
  };

  constructor(
    private maxWorkers: number = getDefaultWorkerCount()
  ) {}

  /**
   * Worker数設定
   */
  setMaxWorkers(count: number): void {
    if (this.searchRunning) {
      console.warn('Cannot change worker count during active search');
      return;
    }
    const maxHwConcurrency = getDefaultWorkerCount();
    this.maxWorkers = Math.max(1, Math.min(count, maxHwConcurrency));
  }

  getMaxWorkers(): number {
    return this.maxWorkers;
  }

  /**
   * 並列検索開始
   */
  async startParallelSearch(
    params: EggBootTimingSearchParams,
    callbacks: EggBootTimingMultiWorkerCallbacks
  ): Promise<void> {
    if (this.searchRunning) {
      throw new Error('Search is already running');
    }

    this.safeCleanup();
    this.callbacks = callbacks;
    this.searchRunning = true;
    this.startManagerTimer();

    try {
      // チャンク分割
      const chunks = calculateEggBootTimingChunks(params, this.maxWorkers);

      if (chunks.length === 0) {
        throw new Error('No valid chunks created for search');
      }

      // バッチサイズ計算
      const batchSize = calculateBatchSize(params);

      // 各チャンクに対してWorker初期化
      for (const chunk of chunks) {
        await this.initializeWorker(chunk, params, batchSize);
      }

      // 進捗監視開始
      this.startProgressMonitoring();
    } catch (error) {
      console.error('Failed to start parallel search:', error);
      this.cleanup();
      callbacks.onError(
        error instanceof Error ? error.message : 'Unknown error'
      );
    }
  }

  /**
   * Worker初期化
   */
  private async initializeWorker(
    chunk: EggBootTimingWorkerChunk,
    params: EggBootTimingSearchParams,
    _batchSize: number
  ): Promise<void> {
    const worker = new Worker(
      new URL('../../workers/egg-boot-timing-worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<EggBootTimingWorkerResponse>) => {
      this.handleWorkerMessage(chunk.workerId, event.data);
    };

    worker.onerror = (error) => {
      console.error(`Worker ${chunk.workerId} error:`, error);
      this.handleWorkerError(
        chunk.workerId,
        new Error(`Worker error: ${error.message}`)
      );
    };

    this.workers.set(chunk.workerId, worker);
    this.activeChunks.set(chunk.workerId, chunk);

    // Worker進捗初期化
    this.workerProgresses.set(chunk.workerId, {
      workerId: chunk.workerId,
      currentStep: 0,
      totalSteps: chunk.estimatedOperations,
      elapsedTime: 0,
      estimatedTimeRemaining: 0,
      matchesFound: 0,
      status: 'initializing',
    });

    // チャンク用パラメータを構築
    const chunkParams: EggBootTimingSearchParams = {
      ...params,
      startDatetimeIso: chunk.startDatetime.toISOString(),
      rangeSeconds: chunk.rangeSeconds,
    };

    // 検索開始リクエスト
    const request = {
      type: 'START_SEARCH' as const,
      params: chunkParams,
      requestId: `worker-${chunk.workerId}`,
    };

    worker.postMessage(request);
  }

  /**
   * Workerメッセージ処理
   */
  private handleWorkerMessage(
    workerId: number,
    data: unknown
  ): void {
    if (!this.callbacks) return;
    if (!isEggBootTimingWorkerResponse(data)) return;

    const response = data;

    switch (response.type) {
      case 'READY':
        break;

      case 'PROGRESS':
        if (response.payload) {
          this.updateWorkerProgress(workerId, response.payload);
        }
        break;

      case 'RESULTS':
        if (response.payload?.results) {
          for (const result of response.payload.results) {
            this.results.push(result);
            this.callbacks.onResult(result);

            const progress = this.workerProgresses.get(workerId);
            if (progress) {
              progress.matchesFound++;
            }
          }
        }
        break;

      case 'COMPLETE':
        this.handleWorkerCompletion(workerId);
        break;

      case 'ERROR':
        console.error(`Worker ${workerId} error:`, response.message);
        this.handleWorkerError(workerId, new Error(response.message));
        break;
    }
  }

  /**
   * Worker進捗更新
   */
  private updateWorkerProgress(
    workerId: number,
    progressData: EggBootTimingProgress
  ): void {
    const current = this.workerProgresses.get(workerId);
    if (!current) return;

    current.currentStep = progressData.processedCombinations;
    current.totalSteps = progressData.totalCombinations;
    current.elapsedTime = progressData.elapsedMs;
    current.estimatedTimeRemaining = progressData.estimatedRemainingMs;
    current.matchesFound = progressData.foundCount;
    current.status = 'running';

    this.lastProgressCheck.set(workerId, Date.now());
  }

  /**
   * 進捗集約とレポート
   */
  private aggregateAndReportProgress(): void {
    if (!this.searchRunning || !this.callbacks) return;

    const progresses = Array.from(this.workerProgresses.values());
    if (progresses.length === 0) return;

    const totalCurrentStep = progresses.reduce(
      (sum, p) => sum + p.currentStep,
      0
    );
    const totalSteps = progresses.reduce((sum, p) => sum + p.totalSteps, 0);
    const totalElapsedTime = this.getManagerElapsedTime();
    const totalMatchesFound = progresses.reduce(
      (sum, p) => sum + p.matchesFound,
      0
    );

    const activeWorkers = progresses.filter(
      (p) => p.status === 'running' || p.status === 'initializing'
    ).length;

    const completedWorkers = progresses.filter(
      (p) => p.status === 'completed'
    ).length;

    const totalEstimatedTimeRemaining =
      this.calculateAggregatedTimeRemaining(progresses);

    const aggregatedProgress: AggregatedEggBootTimingProgress = {
      totalCurrentStep,
      totalSteps,
      totalElapsedTime,
      totalEstimatedTimeRemaining,
      totalMatchesFound,
      activeWorkers,
      completedWorkers,
      workerProgresses: new Map(this.workerProgresses),
    };

    this.callbacks.onProgress(aggregatedProgress);
  }

  /**
   * 残り時間推定
   */
  private calculateAggregatedTimeRemaining(
    progresses: WorkerProgress[]
  ): number {
    const activeProgresses = progresses.filter(
      (p) => p.status === 'running' && p.currentStep > 0
    );

    if (activeProgresses.length === 0) return 0;

    const remainingTimes = activeProgresses.map((p) => {
      if (p.currentStep === 0) return 0;
      const progressRatio = p.currentStep / p.totalSteps;
      if (progressRatio === 0) return 0;
      const estimatedTotalTime = p.elapsedTime / progressRatio;
      return Math.max(0, estimatedTotalTime - p.elapsedTime);
    });

    return Math.max(...remainingTimes);
  }

  /**
   * Worker完了処理
   */
  private handleWorkerCompletion(workerId: number): void {
    const progress = this.workerProgresses.get(workerId);
    if (progress) {
      progress.status = 'completed';
      progress.currentStep = progress.totalSteps;
    }

    this.completedWorkers++;

    if (this.completedWorkers >= this.workers.size) {
      this.handleAllWorkersCompleted();
    }
  }

  /**
   * 全Worker完了処理
   */
  private handleAllWorkersCompleted(): void {
    const totalElapsed = this.getManagerElapsedTime();
    const totalResults = this.results.length;

    // 最終進捗レポート
    this.aggregateAndReportProgress();

    this.callbacks?.onComplete(
      `Parallel search completed. Found ${totalResults} matches in ${Math.round(totalElapsed / 1000)}s`
    );

    this.minimalCleanup();
  }

  /**
   * Workerエラー処理
   */
  private handleWorkerError(workerId: number, error: Error): void {
    const progress = this.workerProgresses.get(workerId);
    if (progress) {
      progress.status = 'error';
    }

    const worker = this.workers.get(workerId);
    if (worker) {
      worker.terminate();
      this.workers.delete(workerId);
    }

    if (this.workers.size === 0) {
      this.cleanup();
      this.callbacks?.onError(`All workers failed: ${error.message}`);
    }
  }

  /**
   * 進捗監視開始
   */
  private startProgressMonitoring(): void {
    this.progressUpdateTimer = setInterval(() => {
      this.aggregateAndReportProgress();
    }, 500);
  }

  /**
   * 一時停止
   */
  pauseAll(): void {
    this.pauseManagerTimer();
    for (const worker of this.workers.values()) {
      worker.postMessage({ type: 'PAUSE' });
    }
    this.callbacks?.onPaused?.();
  }

  /**
   * 再開
   */
  resumeAll(): void {
    this.resumeManagerTimer();
    for (const worker of this.workers.values()) {
      worker.postMessage({ type: 'RESUME' });
    }
    this.callbacks?.onResumed?.();
  }

  /**
   * 停止
   */
  terminateAll(): void {
    const callbacks = this.callbacks;
    this.cleanup();
    callbacks?.onStopped?.();
  }

  /**
   * 状態取得
   */
  isRunning(): boolean {
    return this.searchRunning;
  }
  getActiveWorkerCount(): number {
    return this.workers.size;
  }
  getResultsCount(): number {
    return this.results.length;
  }

  // --- Timer管理 ---
  private startManagerTimer(): void {
    this.timerState.cumulativeRunTime = 0;
    this.timerState.segmentStartTime = Date.now();
    this.timerState.isPaused = false;
  }

  private pauseManagerTimer(): void {
    if (!this.timerState.isPaused) {
      this.timerState.cumulativeRunTime +=
        Date.now() - this.timerState.segmentStartTime;
      this.timerState.isPaused = true;
    }
  }

  private resumeManagerTimer(): void {
    if (this.timerState.isPaused) {
      this.timerState.segmentStartTime = Date.now();
      this.timerState.isPaused = false;
    }
  }

  private getManagerElapsedTime(): number {
    return this.timerState.isPaused
      ? this.timerState.cumulativeRunTime
      : this.timerState.cumulativeRunTime +
          (Date.now() - this.timerState.segmentStartTime);
  }

  // --- クリーンアップ ---
  private minimalCleanup(): void {
    if (this.progressUpdateTimer) {
      clearInterval(this.progressUpdateTimer);
      this.progressUpdateTimer = null;
    }
    for (const worker of this.workers.values()) {
      worker.terminate();
    }
    this.workers.clear();
    this.callbacks = null;
    this.searchRunning = false;
    this.activeChunks.clear();
    this.lastProgressCheck.clear();
    this.results = [];
  }

  safeCleanup(): void {
    this.minimalCleanup();
    this.completedWorkers = 0;
  }

  private cleanup(): void {
    this.safeCleanup();
    this.workerProgresses.clear();
  }
}
