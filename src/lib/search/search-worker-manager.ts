/**
 * Manager for search Web Worker
 * Handles communication between main thread and search worker
 * Extended with parallel search capabilities
 */

import type { SearchConditions, InitialSeedResult } from '../../types/search';
import type { AggregatedProgress } from '../../types/parallel';
import type { WorkerRequest, WorkerResponse } from '../../workers/search-worker';
import { MultiWorkerSearchManager } from './multi-worker-manager';
import type { SingleWorkerSearchCallbacks } from '../../types/callbacks';
import { shouldUseWebGpuSearch } from './search-mode';
import { useAppStore } from '@/store/app-store';
import type { SearchExecutionMode } from '@/store/app-store';

export type SearchCallbacks = SingleWorkerSearchCallbacks<InitialSeedResult> & {
  onParallelProgress?: (progress: AggregatedProgress | null) => void;
};

export class SearchWorkerManager {
  private worker: Worker | null = null;
  private gpuWorker: Worker | null = null;
  private callbacks: SearchCallbacks | null = null;
  private singleWorkerMode: boolean = true;
  private multiWorkerManager: MultiWorkerSearchManager | null = null;
  private activeMode: 'cpu-single' | 'cpu-parallel' | 'gpu' = 'cpu-single';
  private lastRequest: { conditions: SearchConditions; targetSeeds: number[] } | null = null;

  constructor() {
    this.initializeWorker();
  }

  private initializeWorker() {
    try {
      // Create worker with Vite's URL constructor
      this.worker = new Worker(
        new URL('../../workers/search-worker.ts', import.meta.url),
        { type: 'module' }
      );

      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        this.processWorkerResponse(event.data, 'cpu');
      };

      this.worker.onerror = (error) => {
        console.error('Worker error:', error);
        this.callbacks?.onError('Worker error occurred');
      };

    } catch (error) {
      console.error('Failed to initialize search worker:', error);
      this.worker = null;
    }
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
        this.processWorkerResponse(event.data, 'gpu');
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

  private processWorkerResponse(response: WorkerResponse, origin: 'cpu' | 'gpu'): void {
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
        if (origin === 'gpu') {
          this.handleGpuFailure(response.error || 'WebGPU search failed', response.errorCode);
          return;
        }
        this.callbacks.onError(response.error || 'Unknown error');
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
      const fallbackMode: SearchExecutionMode = this.isParallelSearchAvailable() ? 'cpu-parallel' : 'cpu-single';
      store.setSearchExecutionMode(fallbackMode);
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

  private startCpuSearchInternal(
    conditions: SearchConditions,
    targetSeeds: number[],
    callbacks: SearchCallbacks
  ): boolean {
    if (!this.singleWorkerMode) {
      this.activeMode = 'cpu-parallel';
      return this.startParallelSearch(conditions, targetSeeds, callbacks);
    }

    if (!this.worker) {
      this.initializeWorker();
    }

    if (!this.worker) {
      callbacks.onError('Worker not available. Falling back to main thread.');
      return false;
    }

    const request: WorkerRequest = {
      type: 'START_SEARCH',
      conditions,
      targetSeeds
    };

    try {
      this.worker.postMessage(request);
      this.activeMode = 'cpu-single';
      return true;
    } catch (error) {
      console.error('Failed to start CPU search worker:', error);
      callbacks.onError('Failed to start CPU search worker');
      return false;
    }
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
    this.lastRequest = {
      conditions,
      targetSeeds: [...targetSeeds],
    };

    if (shouldUseWebGpuSearch()) {
      const gpuStarted = this.tryStartGpuSearch(conditions, targetSeeds);
      if (gpuStarted) {
        return true;
      }
      console.warn('WebGPU search could not be started. Falling back to CPU mode.');
    }

    return this.startCpuSearchInternal(conditions, targetSeeds, callbacks);
  }

  /**
   * 並列検索開始
   */
  private startParallelSearch(
    conditions: SearchConditions,
    targetSeeds: number[],
    callbacks: SearchCallbacks
  ): boolean {
    try {
      if (!this.multiWorkerManager) {
        this.multiWorkerManager = new MultiWorkerSearchManager();
      }

      // アプリストアから現在のワーカー数設定を取得
      // 注意: ここでは直接importを避けて、公開APIを使用
      const currentMaxWorkers = this.getMaxWorkers();
      this.multiWorkerManager.setMaxWorkers(currentMaxWorkers);

      // 📝 Note: MultiWorkerSearchManager.startParallelSearch()内で
      // safeCleanup()が自動実行されるため、ここでの明示的な呼び出しは不要

      // 並列検索用のコールバック変換
      const parallelCallbacks = {
        onProgress: (aggregatedProgress: AggregatedProgress) => {
          // 既存の進捗フォーマットに変換
          callbacks.onProgress({
            currentStep: aggregatedProgress.totalCurrentStep,
            totalSteps: aggregatedProgress.totalSteps,
            elapsedTime: aggregatedProgress.totalElapsedTime,
            estimatedTimeRemaining: aggregatedProgress.totalEstimatedTimeRemaining,
            matchesFound: aggregatedProgress.totalMatchesFound
          });

          // 並列進捗情報も送信（利用可能な場合）
          if (callbacks.onParallelProgress) {
            callbacks.onParallelProgress(aggregatedProgress);
          }
        },
        onResult: callbacks.onResult,
        onComplete: (message: string) => {
          // 並列進捗は保持（統計表示のため）
          // if (callbacks.onParallelProgress) {
          //   callbacks.onParallelProgress(null);
          // }
          callbacks.onComplete(message);
        },
        onError: (error: string) => {
          // エラー時は進捗をクリア（不正な状態を避けるため）
          if (callbacks.onParallelProgress) {
            callbacks.onParallelProgress(null);
          }
          callbacks.onError(error);
        },
        onPaused: callbacks.onPaused,
        onResumed: callbacks.onResumed,
        onStopped: callbacks.onStopped
      };

      this.multiWorkerManager.startParallelSearch(conditions, targetSeeds, parallelCallbacks);
      this.activeMode = 'cpu-parallel';
      return true;

    } catch (error) {
      console.error('Failed to start parallel search:', error);
      callbacks.onError('Failed to start parallel search. Falling back to single worker mode.');
      
      // フォールバック: 単一Workerモードに切り替え
      this.singleWorkerMode = true;
      return this.startCpuSearchInternal(conditions, targetSeeds, callbacks);
    }
  }

  public pauseSearch() {
    if (this.activeMode === 'gpu' && this.gpuWorker) {
      const request: WorkerRequest = { type: 'PAUSE_SEARCH' };
      this.gpuWorker.postMessage(request);
      return;
    }

    if (!this.singleWorkerMode && this.multiWorkerManager) {
      this.multiWorkerManager.pauseAll();
    } else if (this.worker) {
      const request: WorkerRequest = { type: 'PAUSE_SEARCH' };
      this.worker.postMessage(request);
    }
  }

  public resumeSearch() {
    if (this.activeMode === 'gpu' && this.gpuWorker) {
      const request: WorkerRequest = { type: 'RESUME_SEARCH' };
      this.gpuWorker.postMessage(request);
      return;
    }

    if (!this.singleWorkerMode && this.multiWorkerManager) {
      this.multiWorkerManager.resumeAll();
    } else if (this.worker) {
      const request: WorkerRequest = { type: 'RESUME_SEARCH' };
      this.worker.postMessage(request);
    }
  }

  public stopSearch() {
    if (this.activeMode === 'gpu' && this.gpuWorker) {
      const request: WorkerRequest = { type: 'STOP_SEARCH' };
      this.gpuWorker.postMessage(request);
      return;
    }

    if (!this.singleWorkerMode && this.multiWorkerManager) {
      this.multiWorkerManager.terminateAll();
    } else if (this.worker) {
      const request: WorkerRequest = { type: 'STOP_SEARCH' };
      this.worker.postMessage(request);
    }
  }

  /**
   * 並列検索モードの設定
   */
  public setParallelMode(enabled: boolean): void {
    this.singleWorkerMode = !enabled;
    
    if (enabled && !this.multiWorkerManager) {
      this.multiWorkerManager = new MultiWorkerSearchManager();
    }
  }

  /**
   * ワーカー数設定
   */
  public setMaxWorkers(count: number): void {
    if (!this.multiWorkerManager) {
      this.multiWorkerManager = new MultiWorkerSearchManager();
    }
    this.multiWorkerManager.setMaxWorkers(count);
  }

  /**
   * 現在のワーカー数設定を取得
   */
  public getMaxWorkers(): number {
    if (!this.multiWorkerManager) {
      return navigator.hardwareConcurrency || 4;
    }
    return this.multiWorkerManager.getMaxWorkers();
  }

  /**
   * 並列検索の利用可能性確認
   */
  public isParallelSearchAvailable(): boolean {
    return (navigator.hardwareConcurrency ?? 1) > 1;
  }

  /**
   * 現在のモード取得
   */
  public isParallelMode(): boolean {
    return !this.singleWorkerMode;
  }

  public terminate() {
    if (this.multiWorkerManager) {
      this.multiWorkerManager.terminateAll();
      this.multiWorkerManager = null;
    }
    
    if (this.gpuWorker) {
      this.gpuWorker.terminate();
      this.gpuWorker = null;
    }

    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.callbacks = null;
    this.activeMode = 'cpu-single';
    this.lastRequest = null;
  }

  public isWorkerAvailable(): boolean {
    return this.worker !== null || this.gpuWorker !== null;
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
