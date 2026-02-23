// EggWorkerManager - Worker ライフサイクル管理とコールバック配信

import {
  type EggGenerationParams,
  type EggWorkerRequest,
  type EggWorkerResponse,
  type EggResultsPayload,
  type EggCompletion,
  type EggErrorCategory,
  validateEggParams,
  isEggWorkerResponse,
} from '@/types/egg';

type ResultsCb = (payload: EggResultsPayload) => void;
type CompleteCb = (c: EggCompletion) => void;
type ErrorCb = (msg: string, cat: EggErrorCategory, fatal: boolean) => void;

interface CallbackRegistry {
  results: ResultsCb[];
  complete: CompleteCb[];
  error: ErrorCb[];
}

export class EggWorkerManager {
  private worker: Worker | null = null;
  private callbacks: CallbackRegistry = { results: [], complete: [], error: [] };
  private running = false;
  private terminated = false;
  private currentRequestId: string | null = null;
  private status: 'idle' | 'running' | 'stopping' = 'idle';

  constructor(
    private readonly createWorker: () => Worker = () =>
      new Worker(new URL('@/workers/egg-worker.ts', import.meta.url), { type: 'module' }),
  ) {}

  // --- Public API ---
  start(params: EggGenerationParams): Promise<void> {
    if (this.running) {
      throw new Error('egg generation already running');
    }
    const validation = validateEggParams(params);
    if (validation.length) {
      return Promise.reject(new Error(validation.join(', ')));
    }
    const needsFreshWorker = this.terminated || this.worker === null;
    this.ensureWorker(needsFreshWorker);
    this.running = true;
    this.status = 'running';
    const rid = this.generateRequestId();
    this.currentRequestId = rid;

    const req: EggWorkerRequest = {
      type: 'START_GENERATION',
      params,
      requestId: rid,
    };
    this.worker!.postMessage(req);
    return Promise.resolve();
  }

  stop(): void {
    if (!this.running) return;
    this.status = 'stopping';
    const requestId = this.currentRequestId || undefined;
    this.worker?.postMessage({ type: 'STOP', requestId } satisfies EggWorkerRequest);
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
    }
    this.worker = null;
    this.running = false;
    this.terminated = true;
    this.status = 'idle';
    this.currentRequestId = null;
  }

  onResults(cb: ResultsCb) { this.callbacks.results.push(cb); return this; }
  onComplete(cb: CompleteCb) { this.callbacks.complete.push(cb); return this; }
  onError(cb: ErrorCb) { this.callbacks.error.push(cb); return this; }

  clearCallbacks(): void {
    this.callbacks = { results: [], complete: [], error: [] };
  }

  getStatus(): 'idle' | 'running' | 'stopping' {
    return this.status;
  }

  isRunning() { return this.running; }

  // --- Internal ---
  private ensureWorker(forceNew = false) {
    if (this.worker && (forceNew || this.terminated)) {
      this.worker.terminate();
      this.worker = null;
    }
    if (this.worker) return;
    this.worker = this.createWorker();
    this.terminated = false;
    this.worker.onmessage = (ev: MessageEvent) => this.handleMessage(ev.data);
    this.worker.onerror = () => {
      this.emitError('Worker error event', 'RUNTIME', true);
      this.terminate();
    };
  }

  private handleMessage(raw: unknown) {
    if (!isEggWorkerResponse(raw)) return;
    const msg: EggWorkerResponse = raw;
    switch (msg.type) {
      case 'READY':
        break; // noop
      case 'RESULTS':
        this.callbacks.results.forEach(cb => cb(msg.payload));
        break;
      case 'COMPLETE': {
        this.running = false;
        this.status = 'idle';
        this.currentRequestId = null;
        const completedWorker = this.worker;
        this.worker = null;
        this.terminated = true;
        this.callbacks.complete.forEach(cb => cb(msg.payload));
        completedWorker?.terminate();
        break;
      }
      case 'ERROR':
        this.emitError(msg.message, msg.category, msg.fatal);
        if (msg.fatal) {
          this.running = false;
          this.terminate();
        }
        break;
    }
  }

  private emitError(message: string, category: EggErrorCategory, fatal: boolean) {
    this.callbacks.error.forEach(cb => cb(message, category, fatal));
  }

  private generateRequestId(): string {
    return 'egg-' + Math.random().toString(36).slice(2, 10);
  }
}
