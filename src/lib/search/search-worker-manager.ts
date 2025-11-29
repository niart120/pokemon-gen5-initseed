/**
 * Manager for search Web Worker
 * Handles communication between main thread and search worker
 * Extended with parallel search capabilities
 */

import type { SearchConditions, InitialSeedResult } from '../../types/search';
import type { AggregatedProgress } from '../../types/parallel';
import type { WorkerRequest, WorkerResponse } from '@/types/worker';
import {
  IVBootTimingMultiWorkerManager,
  type AggregatedIVBootTimingProgress,
} from '../iv/iv-boot-timing-multi-worker-manager';
import type {
  IVBootTimingSearchParams,
  IVBootTimingSearchResult,
} from '@/types/iv-boot-timing-search';
import type { SingleWorkerSearchCallbacks } from '../../types/callbacks';
import { useAppStore } from '@/store/app-store';
import type { SearchExecutionMode } from '@/store/app-store';
import { getVCountFromTimer0 } from '@/lib/utils/rom-parameter-helpers';

/**
 * Auto設定時にTimer0範囲から対応するVCount範囲を計算
 * @param romVersion ROM version
 * @param romRegion ROM region
 * @param timer0Min Timer0 minimum
 * @param timer0Max Timer0 maximum
 * @returns VCount範囲 (min/max)
 */
function computeVCountRangeFromTimer0(
  romVersion: string,
  romRegion: string,
  timer0Min: number,
  timer0Max: number
): { min: number; max: number } {
  const vcountSet = new Set<number>();

  // Timer0範囲内の全値についてVCountを取得
  for (let timer0 = timer0Min; timer0 <= timer0Max; timer0++) {
    const vcount = getVCountFromTimer0(romVersion, romRegion, timer0);
    if (vcount !== null) {
      vcountSet.add(vcount);
    }
  }

  if (vcountSet.size === 0) {
    // フォールバック: デフォルト値 0x60
    return { min: 0x60, max: 0x60 };
  }

  const vcounts = Array.from(vcountSet);
  return {
    min: Math.min(...vcounts),
    max: Math.max(...vcounts),
  };
}

/**
 * SearchConditions を IVBootTimingSearchParams に変換
 */
function convertToIVBootTimingSearchParams(
  conditions: SearchConditions,
  targetSeeds: number[]
): IVBootTimingSearchParams {
  // MACアドレスを6要素のタプルに正規化
  const macAddress: readonly [number, number, number, number, number, number] = [
    conditions.macAddress[0] ?? 0,
    conditions.macAddress[1] ?? 0,
    conditions.macAddress[2] ?? 0,
    conditions.macAddress[3] ?? 0,
    conditions.macAddress[4] ?? 0,
    conditions.macAddress[5] ?? 0,
  ];

  // Auto設定時はROMパラメータからVCount範囲を計算
  // GPU検索と同様の動作を保証する
  const vcountRange = conditions.timer0VCountConfig.useAutoConfiguration
    ? computeVCountRangeFromTimer0(
        conditions.romVersion,
        conditions.romRegion,
        conditions.timer0VCountConfig.timer0Range.min,
        conditions.timer0VCountConfig.timer0Range.max
      )
    : {
        min: conditions.timer0VCountConfig.vcountRange.min,
        max: conditions.timer0VCountConfig.vcountRange.max,
      };

  return {
    dateRange: {
      startYear: conditions.dateRange.startYear,
      startMonth: conditions.dateRange.startMonth,
      startDay: conditions.dateRange.startDay,
      endYear: conditions.dateRange.endYear,
      endMonth: conditions.dateRange.endMonth,
      endDay: conditions.dateRange.endDay,
    },
    timer0Range: {
      min: conditions.timer0VCountConfig.timer0Range.min,
      max: conditions.timer0VCountConfig.timer0Range.max,
    },
    vcountRange,
    keyInputMask: conditions.keyInput,
    macAddress,
    hardware: conditions.hardware,
    romVersion: conditions.romVersion,
    romRegion: conditions.romRegion,
    timeRange: conditions.timeRange,
    targetSeeds,
    maxResults: 10000, // デフォルト上限
  };
}

export type SearchCallbacks = SingleWorkerSearchCallbacks<InitialSeedResult> & {
  onParallelProgress?: (progress: AggregatedProgress | null) => void;
};

/**
 * IV Boot Timing検索用コールバック
 */
export type IVBootTimingSearchCallbacks = SingleWorkerSearchCallbacks<IVBootTimingSearchResult> & {
  onParallelProgress?: (progress: AggregatedIVBootTimingProgress | null) => void;
};

export class SearchWorkerManager {
  private gpuWorker: Worker | null = null;
  private callbacks: SearchCallbacks | null = null;
  private ivBootTimingManager: IVBootTimingMultiWorkerManager | null = null;
  private activeMode: 'cpu-parallel' | 'gpu' | 'iv-boot-timing' = 'cpu-parallel';
  private lastRequest: { conditions: SearchConditions; targetSeeds: number[] } | null = null;

  constructor() {
  }

  private initializeGpuWorker(): void {
    if (this.gpuWorker) {
      return;
    }

    try {
      this.gpuWorker = new Worker(
        new URL('../../workers/search-worker-webgpu.ts', import.meta.url),
        { type: 'module' }
      );

      this.gpuWorker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        this.processWorkerResponse(event.data);
      };

      this.gpuWorker.onerror = (error) => {
        console.error('WebGPU worker error:', error);
        this.handleGpuFailure('WebGPU worker error');
      };
    } catch (error) {
      console.error('Failed to initialize WebGPU worker:', error);
      this.gpuWorker = null;
    }
  }

  private processWorkerResponse(response: WorkerResponse): void {
    if (!this.callbacks) return;

    switch (response.type) {
      case 'READY':
        break;

      case 'PROGRESS':
        if (response.progress) {
          this.callbacks.onProgress({
            ...response.progress,
            currentDateTime: response.progress.currentDateTime
              ? new Date(response.progress.currentDateTime)
              : undefined
          });
        }
        break;

      case 'RESULT':
        if (response.result) {
          const result: InitialSeedResult = {
            ...response.result,
            datetime: new Date(response.result.datetime)
          };
          this.callbacks.onResult(result);
        }
        break;

      case 'COMPLETE':
        this.callbacks.onComplete(response.message || 'Search completed');
        break;

      case 'ERROR':
        this.handleGpuFailure(response.error || 'WebGPU search failed', response.errorCode);
        break;

      case 'PAUSED':
        this.callbacks.onPaused();
        break;

      case 'RESUMED':
        this.callbacks.onResumed();
        break;

      case 'STOPPED':
        this.callbacks.onStopped();
        break;

      default:
        console.warn('Unknown worker response type:', response);
    }
  }

  private handleGpuFailure(errorMessage: string, errorCode?: string): void {
    console.warn('WebGPU worker reported an error. Falling back to CPU search.', {
      errorMessage,
      errorCode
    });

    if (!this.callbacks) {
      return;
    }

    if (this.gpuWorker) {
      this.gpuWorker.terminate();
      this.gpuWorker = null;
    }

    try {
      const store = useAppStore.getState();
      const fallbackMode: SearchExecutionMode = 'cpu-parallel';
      if (this.isParallelSearchAvailable()) {
        store.setSearchExecutionMode(fallbackMode);
      }
    } catch (storeError) {
      console.warn('Failed to update execution mode after WebGPU failure:', storeError);
    }

    const request = this.lastRequest;
    if (!request) {
      this.callbacks.onError(errorMessage);
      return;
    }

    const fallbackStarted = this.startCpuSearchInternal(request.conditions, request.targetSeeds, this.callbacks);

    if (!fallbackStarted) {
      this.callbacks.onError(errorMessage);
      return;
    }

    console.warn('CPU search fallback activated after WebGPU failure.');
  }

  /**
   * CPU並列検索（IVBootTimingMultiWorkerManager使用）
   */
  private startCpuSearchInternal(
    conditions: SearchConditions,
    targetSeeds: number[],
    callbacks: SearchCallbacks
  ): boolean {
    if (!this.isParallelSearchAvailable()) {
      callbacks.onError('Parallel CPU search is not available in this environment.');
      return false;
    }

    const ivParams = convertToIVBootTimingSearchParams(conditions, targetSeeds);

    // IVBootTimingSearchResult を InitialSeedResult に変換するコールバック
    const ivCallbacks: IVBootTimingSearchCallbacks = {
      onProgress: callbacks.onProgress,
      onResult: (result: IVBootTimingSearchResult) => {
        // IVBootTimingSearchResult を InitialSeedResult に変換
        const converted: InitialSeedResult = {
          seed: result.mtSeed,
          datetime: result.boot.datetime,
          timer0: result.boot.timer0,
          vcount: result.boot.vcount,
          keyCode: result.boot.keyCode,
          keyInputNames: result.boot.keyInputNames,
          conditions,
          message: [], // WASMからは取得していない
          sha1Hash: '', // WASMからは取得していない
          lcgSeed: BigInt('0x' + result.lcgSeedHex),
          isMatch: true,
        };
        callbacks.onResult(converted);
      },
      onComplete: callbacks.onComplete,
      onError: callbacks.onError,
      onPaused: callbacks.onPaused,
      onResumed: callbacks.onResumed,
      onStopped: callbacks.onStopped,
      onParallelProgress: callbacks.onParallelProgress 
        ? (progress) => {
            // AggregatedIVBootTimingProgress を AggregatedProgress に変換
            if (progress) {
              const converted: AggregatedProgress = {
                totalCurrentStep: progress.totalCurrentStep,
                totalSteps: progress.totalSteps,
                totalElapsedTime: progress.totalElapsedTime,
                totalEstimatedTimeRemaining: progress.totalEstimatedTimeRemaining,
                totalMatchesFound: progress.totalMatchesFound,
                activeWorkers: progress.activeWorkers,
                completedWorkers: progress.completedWorkers,
                workerProgresses: progress.workerProgresses,
                progressPercent: progress.progressPercent,
                totalProcessedSeconds: progress.totalProcessedSeconds,
              };
              callbacks.onParallelProgress!(converted);
            } else {
              callbacks.onParallelProgress!(null);
            }
          }
        : undefined,
    };

    return this.startIVBootTimingSearch(ivParams, ivCallbacks);
  }

  private tryStartGpuSearch(
    conditions: SearchConditions,
    targetSeeds: number[]
  ): boolean {
    this.initializeGpuWorker();

    if (!this.gpuWorker) {
      return false;
    }

    const request: WorkerRequest = {
      type: 'START_SEARCH',
      conditions,
      targetSeeds
    };

    try {
      this.gpuWorker.postMessage(request);
      this.activeMode = 'gpu';
      return true;
    } catch (error) {
      console.error('Failed to start WebGPU search worker:', error);
      return false;
    }
  }

  public startSearch(
    conditions: SearchConditions,
    targetSeeds: number[],
    callbacks: SearchCallbacks
  ): boolean {
    this.callbacks = callbacks;

    const normalizedTargetSeeds = this.normalizeTargetSeeds(targetSeeds);
    if (normalizedTargetSeeds.length === 0) {
      const errorMessage = 'Target seed list is empty or invalid. Please configure at least one seed before starting the search.';
      console.error(errorMessage, { targetSeeds });
      this.callbacks.onError(errorMessage);
      return false;
    }

    this.lastRequest = {
      conditions,
      targetSeeds: [...normalizedTargetSeeds],
    };

    // モード判定
    const executionMode = this.getCurrentExecutionMode();

    // WebGPU モード
    if (executionMode === 'gpu') {
      const gpuStarted = this.tryStartGpuSearch(conditions, normalizedTargetSeeds);
      if (gpuStarted) {
        return true;
      }
      console.warn('WebGPU search could not be started. Falling back to CPU mode.');
    }

    // CPU並列検索モード
    return this.startCpuSearchInternal(conditions, normalizedTargetSeeds, callbacks);
  }

  /**
   * 現在の実行モードを取得
   */
  private getCurrentExecutionMode(): SearchExecutionMode {
    try {
      return useAppStore.getState().searchExecutionMode;
    } catch {
      return 'cpu-parallel';
    }
  }

  /**
   * IV Boot Timing検索開始
   * 指定されたMT Seedに対応する起動時間を検索
   */
  public startIVBootTimingSearch(
    params: IVBootTimingSearchParams,
    callbacks: IVBootTimingSearchCallbacks
  ): boolean {
    try {
      if (!this.ivBootTimingManager) {
        this.ivBootTimingManager = new IVBootTimingMultiWorkerManager();
      }

      const currentMaxWorkers = this.getMaxWorkers();
      this.ivBootTimingManager.setMaxWorkers(currentMaxWorkers);

      const ivCallbacks = {
        onProgress: (aggregatedProgress: AggregatedIVBootTimingProgress) => {
          callbacks.onProgress({
            currentStep: aggregatedProgress.totalCurrentStep,
            totalSteps: aggregatedProgress.totalSteps,
            elapsedTime: aggregatedProgress.totalElapsedTime,
            estimatedTimeRemaining: aggregatedProgress.totalEstimatedTimeRemaining,
            matchesFound: aggregatedProgress.totalMatchesFound
          });

          if (callbacks.onParallelProgress) {
            callbacks.onParallelProgress(aggregatedProgress);
          }
        },
        onResult: callbacks.onResult,
        onComplete: callbacks.onComplete,
        onError: (error: string) => {
          if (callbacks.onParallelProgress) {
            callbacks.onParallelProgress(null);
          }
          callbacks.onError(error);
        },
        onPaused: callbacks.onPaused,
        onResumed: callbacks.onResumed,
        onStopped: callbacks.onStopped
      };

      this.ivBootTimingManager.startParallelSearch(params, ivCallbacks);
      this.activeMode = 'iv-boot-timing';
      return true;

    } catch (error) {
      console.error('Failed to start IV boot timing search:', error);
      callbacks.onError('Failed to start IV boot timing search.');
      return false;
    }
  }

  public pauseSearch() {
    if (this.activeMode === 'gpu' && this.gpuWorker) {
      const request: WorkerRequest = { type: 'PAUSE_SEARCH' };
      this.gpuWorker.postMessage(request);
      return;
    }

    if (this.ivBootTimingManager) {
      this.ivBootTimingManager.pauseAll();
    }
  }

  public resumeSearch() {
    if (this.activeMode === 'gpu' && this.gpuWorker) {
      const request: WorkerRequest = { type: 'RESUME_SEARCH' };
      this.gpuWorker.postMessage(request);
      return;
    }

    if (this.ivBootTimingManager) {
      this.ivBootTimingManager.resumeAll();
    }
  }

  public stopSearch() {
    if (this.activeMode === 'gpu' && this.gpuWorker) {
      const request: WorkerRequest = { type: 'STOP_SEARCH' };
      this.gpuWorker.postMessage(request);
      return;
    }

    if (this.ivBootTimingManager) {
      this.ivBootTimingManager.terminateAll();
    }
  }

  /**
   * ワーカー数設定
   */
  public setMaxWorkers(count: number): void {
    if (!this.ivBootTimingManager) {
      this.ivBootTimingManager = new IVBootTimingMultiWorkerManager();
    }
    this.ivBootTimingManager.setMaxWorkers(count);
  }

  /**
   * 現在のワーカー数設定を取得
   */
  public getMaxWorkers(): number {
    if (!this.ivBootTimingManager) {
      return navigator.hardwareConcurrency || 4;
    }
    return this.ivBootTimingManager.getMaxWorkers();
  }

  /**
   * 並列検索の利用可能性確認
   */
  public isParallelSearchAvailable(): boolean {
    return typeof Worker !== 'undefined';
  }

  public terminate() {
    if (this.ivBootTimingManager) {
      this.ivBootTimingManager.terminateAll();
      this.ivBootTimingManager = null;
    }
    
    if (this.gpuWorker) {
      this.gpuWorker.terminate();
      this.gpuWorker = null;
    }

    this.callbacks = null;
    this.activeMode = 'cpu-parallel';
    this.lastRequest = null;
  }

  private normalizeTargetSeeds(seeds: number[] | undefined | null): number[] {
    if (!Array.isArray(seeds)) {
      return [];
    }

    return seeds.filter((seed) => typeof seed === 'number' && Number.isFinite(seed));
  }
}

// Singleton instance
let workerManager: SearchWorkerManager | null = null;

export function getSearchWorkerManager(): SearchWorkerManager {
  if (!workerManager) {
    workerManager = new SearchWorkerManager();
  }
  return workerManager;
}

/**
 * ワーカーマネージャーのシングルトンインスタンスを完全にリセット
 * メモリリーク防止のため、検索完了時に呼び出し推奨
 */
export function resetSearchWorkerManager(): void {
  if (workerManager) {
    workerManager.terminate();
    workerManager = null;
  }
}
