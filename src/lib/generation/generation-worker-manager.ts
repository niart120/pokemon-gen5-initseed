// GenerationWorkerManager (Phase3/4 Task6)
// Worker ライフサイクル抽象化 + コールバック配信

import {
  type GenerationParams,
  type GenerationWorkerRequest,
  type GenerationWorkerResponse,
  type GenerationResultsPayload,
  type GenerationCompletion,
  type GenerationErrorCategory,
  validateGenerationParams,
  isGenerationWorkerResponse,
} from '@/types/generation';
import type { SerializedResolutionContext } from '@/types/pokemon-resolved';

type ResultsCb = (payload: GenerationResultsPayload) => void;
type CompleteCb = (c: GenerationCompletion) => void;
type ErrorCb = (msg: string, cat: GenerationErrorCategory, fatal: boolean) => void;

interface CallbackRegistry {
  results: ResultsCb[];
  complete: CompleteCb[];
  error: ErrorCb[];
}

export class GenerationWorkerManager {
  private worker: Worker | null = null;
  private callbacks: CallbackRegistry = { results: [], complete: [], error: [] };
  private running = false;
  private terminated = false;
  private currentRequestId: string | null = null;
  private status: 'idle' | 'running' | 'stopping' = 'idle';

  constructor(
    private readonly createWorker: () => Worker = () =>
      new Worker(new URL('@/workers/generation-worker.ts', import.meta.url), { type: 'module' }),
  ) {}

  // --- Public API ---
  start(params: GenerationParams, options?: { resolutionContext?: SerializedResolutionContext }): Promise<void> {
    if (this.running) {
      throw new Error('generation already running');
    }
    if (this.terminated) {
      // 再利用しない方針 -> 真新しい worker を生成し直す
      this.terminated = false;
    }
    const validation = validateGenerationParams(params);
    if (validation.length) {
      return Promise.reject(new Error(validation.join(', ')));
    }
    this.ensureWorker();
    this.running = true;
    this.status = 'running';
    const rid = this.generateRequestId();
    this.currentRequestId = rid;

    const req: GenerationWorkerRequest = {
      type: 'START_GENERATION',
      params,
      requestId: rid,
      resolutionContext: options?.resolutionContext,
    };
    this.worker!.postMessage(req);
    return Promise.resolve();
  }

  stop(): void {
    if (!this.running) return;
    this.status = 'stopping';
    const requestId = this.currentRequestId || undefined;
    this.worker?.postMessage({ type: 'STOP', requestId } satisfies GenerationWorkerRequest);
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
    }
    this.worker = null;
    this.running = false;
    this.terminated = true;
    this.status = 'idle';
  }

  onResults(cb: ResultsCb) { this.callbacks.results.push(cb); return this; }
  onComplete(cb: CompleteCb) { this.callbacks.complete.push(cb); return this; }
  onError(cb: ErrorCb) { this.callbacks.error.push(cb); return this; }

  getStatus(): 'idle' | 'running' | 'stopping' {
    return this.status;
  }

  isRunning() { return this.running; }

  // --- Internal ---
  private ensureWorker() {
    if (this.worker) return;
    this.worker = this.createWorker();
    this.worker.onmessage = (ev: MessageEvent) => this.handleMessage(ev.data);
    this.worker.onerror = () => {
      this.emitError('Worker error event', 'RUNTIME', true);
      this.terminate();
    };
  }

  private handleMessage(raw: unknown) {
    if (!isGenerationWorkerResponse(raw)) return;
    const msg: GenerationWorkerResponse = raw;
    switch (msg.type) {
      case 'READY':
        break; // noop
      case 'RESULTS':
        this.callbacks.results.forEach(cb => cb(msg.payload));
        break;
      case 'COMPLETE': {
        this.running = false;
        this.status = 'idle';
        this.callbacks.complete.forEach(cb => cb(msg.payload));
        this.terminate();
        break; }
      case 'ERROR':
        this.emitError(msg.message, msg.category, msg.fatal);
        if (msg.fatal) {
          this.running = false;
          this.terminate();
        }
        break;
    }
  }

  private emitError(message: string, category: GenerationErrorCategory, fatal: boolean) {
    this.callbacks.error.forEach(cb => cb(message, category, fatal));
  }

  private generateRequestId(): string {
    return 'gen-' + Math.random().toString(36).slice(2, 10);
  }
}

// TODO: BigInt serialization strategy (result batches) — 実装フェーズで検討
// TODO: READY待機強化 (timeout + 再試行) — 必要になれば追加
