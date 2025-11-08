// GenerationWorkerManager (Phase3/4 Task6)
// Worker ライフサイクル抽象化 + コールバック配信

import {
  type GenerationParams,
  type GenerationProgress,
  type GenerationWorkerRequest,
  type GenerationWorkerResponse,
  type GenerationResultBatch,
  type GenerationCompletion,
  type GenerationErrorCategory,
  validateGenerationParams,
  isGenerationWorkerResponse,
} from '@/types/generation';

type ProgressCb = (p: GenerationProgress) => void;
type BatchCb = (b: GenerationResultBatch) => void;
type CompleteCb = (c: GenerationCompletion) => void;
type ErrorCb = (msg: string, cat: GenerationErrorCategory, fatal: boolean) => void;

interface CallbackRegistry {
  progress: ProgressCb[];
  batch: BatchCb[];
  complete: CompleteCb[];
  stopped: CompleteCb[]; // STOPPED payload は Completion のサブセット扱い
  error: ErrorCb[];
}

export class GenerationWorkerManager {
  private worker: Worker | null = null;
  private callbacks: CallbackRegistry = { progress: [], batch: [], complete: [], stopped: [], error: [] };
  private lastProgress: GenerationProgress | null = null;
  private running = false;
  private paused = false;
  private terminated = false;
  private currentRequestId: string | null = null;

  constructor(
    private readonly createWorker: () => Worker = () =>
      new Worker(new URL('@/workers/generation-worker.ts', import.meta.url), { type: 'module' }),
  ) {}

  // --- Public API ---
  start(params: GenerationParams): Promise<void> {
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
    this.paused = false;
    const rid = this.generateRequestId();
    this.currentRequestId = rid;

    const req: GenerationWorkerRequest = { type: 'START_GENERATION', params, requestId: rid };
    this.worker!.postMessage(req);
    return Promise.resolve();
  }

  pause(): void {
    if (!this.running || this.paused) return;
    const requestId = this.currentRequestId || undefined;
    this.worker?.postMessage({ type: 'PAUSE', requestId } satisfies GenerationWorkerRequest);
  }

  resume(): void {
    if (!this.running || !this.paused) return;
    const requestId = this.currentRequestId || undefined;
    this.worker?.postMessage({ type: 'RESUME', requestId } satisfies GenerationWorkerRequest);
  }

  stop(): void {
    if (!this.running) return;
    const requestId = this.currentRequestId || undefined;
    this.worker?.postMessage({ type: 'STOP', requestId } satisfies GenerationWorkerRequest);
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
    }
    this.worker = null;
    this.running = false;
    this.paused = false;
    this.terminated = true;
  }

  onProgress(cb: ProgressCb) { this.callbacks.progress.push(cb); return this; }
  onResultBatch(cb: BatchCb) { this.callbacks.batch.push(cb); return this; }
  onComplete(cb: CompleteCb) { this.callbacks.complete.push(cb); return this; }
  onStopped(cb: CompleteCb) { this.callbacks.stopped.push(cb); return this; }
  onError(cb: ErrorCb) { this.callbacks.error.push(cb); return this; }

  getStatus(): GenerationProgress['status'] | 'idle' {
    return this.lastProgress?.status ?? 'idle';
  }
  getLastProgress() { return this.lastProgress; }

  isRunning() { return this.running && !this.paused; }
  isPaused() { return this.paused; }

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
      case 'PROGRESS':
        this.lastProgress = msg.payload;
        this.callbacks.progress.forEach(cb => cb(msg.payload));
        this.paused = msg.payload.status === 'paused';
        break;
      case 'RESULT_BATCH':
        this.callbacks.batch.forEach(cb => cb(msg.payload));
        break;
      case 'PAUSED':
        this.paused = true;
        break;
      case 'RESUMED':
        this.paused = false;
        break;
      case 'STOPPED': {
        this.running = false;
        this.paused = false;
        if (this.lastProgress) {
          this.lastProgress = { ...this.lastProgress, status: 'stopped' };
        }
        this.callbacks.stopped.forEach(cb => cb(msg.payload));
        this.terminate();
        break; }
      case 'COMPLETE': {
        this.running = false;
        this.paused = false;
        if (this.lastProgress) {
          this.lastProgress = { ...this.lastProgress, status: 'completed' };
        }
        this.callbacks.complete.forEach(cb => cb(msg.payload));
        this.terminate();
        break; }
      case 'ERROR':
        this.emitError(msg.message, msg.category, msg.fatal);
        if (msg.fatal) {
          this.running = false;
          this.paused = false;
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
