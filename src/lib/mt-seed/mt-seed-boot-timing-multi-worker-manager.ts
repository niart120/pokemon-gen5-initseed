/**
 * MT Seed Boot Timing Multi Worker Manager - 並列Worker版
 * MT Seed 起動時間検索の並列Worker管理
 */

import type {
  MtSeedBootTimingSearchParams,
  MtSeedBootTimingSearchResult,
  MtSeedBootTimingWorkerResponse,
  MtSeedBootTimingProgress,
} from '@/types/mt-seed-boot-timing-search';
import { isMtSeedBootTimingWorkerResponse } from '@/types/mt-seed-boot-timing-search';
import {
  calculateMtSeedBootTimingChunks,
  getDefaultWorkerCount,
  type MtSeedBootTimingWorkerChunk,
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
  /** 進捗パーセント（0-100） */
  progressPercent: number;
  /** 処理済み秒数（処理速度計算用） */
  processedSeconds: number;
}

/**
 * 集約された進捗状態
 */
export interface AggregatedMtSeedBootTimingProgress {
  totalCurrentStep: number;
  totalSteps: number;
  totalElapsedTime: number;
  totalEstimatedTimeRemaining: number;
  totalMatchesFound: number;
  activeWorkers: number;
  completedWorkers: number;
  workerProgresses: Map<number, WorkerProgress>;
  /** 全体の進捗パーセント（0-100、各Workerの平均） */
  progressPercent: number;
  /** 全Worker合計の処理済み秒数（処理速度計算用） */
  totalProcessedSeconds: number;
}

/**
 * コールバック定義
 */
export interface MtSeedBootTimingMultiWorkerCallbacks {
  onProgress: (progress: AggregatedMtSeedBootTimingProgress) => void;
  onResult: (result: MtSeedBootTimingSearchResult) => void;
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
export class MtSeedBootTimingMultiWorkerManager {
  private workers: Map<number, Worker> = new Map();
  private workerProgresses: Map<number, WorkerProgress> = new Map();
  private activeChunks: Map<number, MtSeedBootTimingWorkerChunk> = new Map();
  private resultsCount = 0; // 結果はストリーミングのみ、配列保持しない
  private completedWorkers = 0;
  private callbacks: MtSeedBootTimingMultiWorkerCallbacks | null = null;
  private searchRunning = false;
  private progressUpdateTimer: ReturnType<typeof setInterval> | null = null;
  private lastProgressCheck: Map<number, number> = new Map();

  private timerState: TimerState = {
    cumulativeRunTime: 0,
    segmentStartTime: 0,
    isPaused: false,
  };

  constructor(private maxWorkers: number = getDefaultWorkerCount()) {}

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
    params: MtSeedBootTimingSearchParams,
    callbacks: MtSeedBootTimingMultiWorkerCallbacks
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
      const chunks = calculateMtSeedBootTimingChunks(params, this.maxWorkers);

      if (chunks.length === 0) {
        throw new Error('No valid chunks created for search');
      }

      // 全Workerを並列初期化（直列だと先に初期化したWorkerが先行してしまう）
      await Promise.all(
        chunks.map((chunk) => this.initializeWorker(chunk, params))
      );

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
    chunk: MtSeedBootTimingWorkerChunk,
    params: MtSeedBootTimingSearchParams
  ): Promise<void> {
    const worker = new Worker(
      new URL('../../workers/mt-seed-boot-timing-worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<MtSeedBootTimingWorkerResponse>) => {
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
      progressPercent: 0,
      processedSeconds: 0,
    });

    // チャンク用パラメータを構築
    const chunkStartDatetime = chunk.startDatetime;
    const chunkEndDatetime = chunk.endDatetime;

    const chunkParams: MtSeedBootTimingSearchParams = {
      ...params,
      dateRange: {
        startYear: chunkStartDatetime.getFullYear(),
        startMonth: chunkStartDatetime.getMonth() + 1,
        startDay: chunkStartDatetime.getDate(),
        endYear: chunkEndDatetime.getFullYear(),
        endMonth: chunkEndDatetime.getMonth() + 1,
        endDay: chunkEndDatetime.getDate(),
      },
      // チャンク分割時はrangeSecondsを明示的に指定（Worker側での再計算を防止）
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
  private handleWorkerMessage(workerId: number, data: unknown): void {
    if (!this.callbacks) return;
    if (!isMtSeedBootTimingWorkerResponse(data)) return;

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
            // 結果は配列保持せずストリーミング（OOM対策）
            this.resultsCount++;
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
    progressData: MtSeedBootTimingProgress
  ): void {
    const current = this.workerProgresses.get(workerId);
    if (!current) return;

    current.currentStep = progressData.processedCombinations;
    current.totalSteps = progressData.totalCombinations;
    current.elapsedTime = progressData.elapsedMs;
    current.estimatedTimeRemaining = progressData.estimatedRemainingMs;
    current.matchesFound = progressData.foundCount;
    current.status = 'running';
    // progressPercentがない場合は0を使用（後方互換性）
    current.progressPercent = progressData.progressPercent ?? 0;
    // processedSecondsがない場合は0（後方互換性）
    current.processedSeconds = progressData.processedSeconds ?? 0;

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
    // 各Workerの進捗パーセントの平均を計算
    const progressPercent = progresses.length > 0
      ? progresses.reduce((sum, p) => sum + p.progressPercent, 0) / progresses.length
      : 0;
    const totalProcessedSeconds = progresses.reduce(
      (sum, p) => sum + p.processedSeconds,
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

    const aggregatedProgress: AggregatedMtSeedBootTimingProgress = {
      totalCurrentStep,
      totalSteps,
      totalElapsedTime,
      totalEstimatedTimeRemaining,
      totalMatchesFound,
      activeWorkers,
      completedWorkers,
      workerProgresses: new Map(this.workerProgresses),
      progressPercent,
      totalProcessedSeconds,
    };

    this.callbacks.onProgress(aggregatedProgress);
  }

  /**
   * 残り時間推定
   * 各Workerから報告された残り時間の最大値を使用
   */
  private calculateAggregatedTimeRemaining(
    progresses: WorkerProgress[]
  ): number {
    const activeProgresses = progresses.filter(
      (p) => p.status === 'running' && p.progressPercent > 0
    );

    if (activeProgresses.length === 0) return 0;

    // 各Workerの推定残り時間の最大値を使用
    const remainingTimes = activeProgresses.map((p) => p.estimatedTimeRemaining);
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
    const totalResults = this.resultsCount;

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
      const callbacks = this.callbacks; // cleanup前にコールバックを保持
      this.cleanup();
      callbacks?.onError(`All workers failed: ${error.message}`);
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
      worker.postMessage({ type: 'STOP' }); // IV workerはPAUSEがないためSTOPを使用
    }
    this.callbacks?.onPaused?.();
  }

  /**
   * 再開
   */
  resumeAll(): void {
    this.resumeManagerTimer();
    // MT Seed workerはRESUMEをサポートしていないため、再開は新規検索が必要
    // 現状では警告のみ
    console.warn(
      'MT Seed boot timing worker does not support resume. Please restart search.'
    );
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
    return this.resultsCount;
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
    this.resultsCount = 0;
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
