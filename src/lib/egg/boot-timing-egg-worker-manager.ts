/**
 * Egg Boot Timing Worker Manager - 単一Worker版
 * 孵化乱数起動時間検索の単一Worker管理
 */

import type {
  EggBootTimingSearchParams,
  EggBootTimingSearchResult,
  EggBootTimingWorkerRequest,
  EggBootTimingWorkerResponse,
  EggBootTimingCompletion,
  EggBootTimingProgress,
} from '@/types/egg-boot-timing-search';
import { isEggBootTimingWorkerResponse } from '@/types/egg-boot-timing-search';

export interface EggBootTimingWorkerCallbacks {
  onReady?: () => void;
  onProgress?: (progress: EggBootTimingProgress) => void;
  onResults?: (results: EggBootTimingSearchResult[]) => void;
  onComplete?: (completion: EggBootTimingCompletion) => void;
  onError?: (error: {
    message: string;
    category: string;
    fatal: boolean;
  }) => void;
}

/**
 * 単一Worker版マネージャ
 */
export class EggBootTimingWorkerManager {
  private worker: Worker | null = null;
  private callbacks: EggBootTimingWorkerCallbacks = {};
  private isRunning = false;

  constructor() {}

  async initialize(
    callbacks: EggBootTimingWorkerCallbacks
  ): Promise<void> {
    this.callbacks = callbacks;

    this.worker = new Worker(
      new URL('../../workers/egg-boot-timing-worker.ts', import.meta.url),
      { type: 'module' }
    );

    this.worker.onmessage = (
      event: MessageEvent<EggBootTimingWorkerResponse>
    ) => {
      this.handleMessage(event.data);
    };

    this.worker.onerror = (error) => {
      this.callbacks.onError?.({
        message: error.message || 'Worker error',
        category: 'RUNTIME',
        fatal: true,
      });
    };

    // Wait for READY message
    await new Promise<void>((resolve) => {
      const originalOnReady = this.callbacks.onReady;
      this.callbacks.onReady = () => {
        originalOnReady?.();
        resolve();
      };
    });
  }

  async startSearch(params: EggBootTimingSearchParams): Promise<void> {
    if (!this.worker) throw new Error('Worker not initialized');
    if (this.isRunning) throw new Error('Search already running');

    this.isRunning = true;

    const request: EggBootTimingWorkerRequest = {
      type: 'START_SEARCH',
      params,
      requestId: crypto.randomUUID(),
    };

    this.worker.postMessage(request);
  }

  stopSearch(): void {
    if (!this.worker || !this.isRunning) return;
    const request: EggBootTimingWorkerRequest = { type: 'STOP' };
    this.worker.postMessage(request);
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.isRunning = false;
  }

  get running(): boolean {
    return this.isRunning;
  }

  private handleMessage(data: unknown): void {
    if (!isEggBootTimingWorkerResponse(data)) {
      return;
    }

    const response = data;

    switch (response.type) {
      case 'READY':
        this.callbacks.onReady?.();
        break;
      case 'PROGRESS':
        if (response.payload) {
          this.callbacks.onProgress?.(response.payload);
        }
        break;
      case 'RESULTS':
        if (response.payload?.results) {
          this.callbacks.onResults?.(response.payload.results);
        }
        break;
      case 'COMPLETE':
        this.isRunning = false;
        this.callbacks.onComplete?.(response.payload);
        break;
      case 'ERROR':
        if (response.fatal) this.isRunning = false;
        this.callbacks.onError?.({
          message: response.message,
          category: response.category,
          fatal: response.fatal,
        });
        break;
    }
  }
}
