import { describe, it, expect, beforeAll } from 'vitest';
import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationCompletion } from '@/types/generation';

// --- Mock Worker (Node 環境用) ---
class MockWorker {
  public onmessage: ((ev: MessageEvent)=>void) | null = null;
  public onerror: ((ev: any)=>void) | null = null;
  private interval: any = null;
  private running = false;
  private advances = 0;
  private startTs = 0;
  private params!: GenerationParams;
  constructor(_script: string, _opts?: any) {
    setTimeout(()=> this.dispatch({ type: 'READY', version: '1' }), 0);
  }
  postMessage(msg: any) {
    switch (msg.type) {
      case 'START_GENERATION':
        this.params = msg.params;
        this.running = true; this.advances = 0; this.startTs = Date.now();
        this.startLoop();
        break;
      case 'STOP':
        if (this.running) {
          this.running = false;
          this.dispatchComplete('stopped');
          this.cleanup();
        }
        break;
    }
  }
  terminate() { this.cleanup(); }
  private startLoop() {
    const step = () => {
      if (!this.running) return;
      this.advances += 250;
      this.dispatch({ type: 'RESULTS', payload: { results: [] } });
      if (this.advances >= this.params.maxAdvances) {
        this.running = false;
        this.dispatchComplete('max-advances');
        this.cleanup();
        return;
      }
      this.interval = setTimeout(step, 25);
    };
    step();
  }
  private buildCompletion(reason: GenerationCompletion['reason']): GenerationCompletion {
    const elapsedMs = Date.now() - this.startTs;
    return {
      processedAdvances: Math.min(this.advances, this.params.maxAdvances),
      resultsCount: 0,
      elapsedMs,
      shinyFound: false,
      reason,
    };
  }
  private dispatchComplete(reason: GenerationCompletion['reason']) {
    this.dispatch({ type: 'COMPLETE', payload: this.buildCompletion(reason) });
  }
  private cleanup() { if (this.interval) clearTimeout(this.interval); this.interval = null; }
  private dispatch(data: any) { this.onmessage?.({ data } as any); }
}

beforeAll(() => {
  if (typeof (globalThis as any).Worker === 'undefined') {
    (globalThis as any).Worker = MockWorker as any;
  }
});

function params(overrides: Partial<GenerationParams> = {}): GenerationParams {
  // NOTE: Validationルール: maxResults <= maxAdvances
  // シーケンステストで maxAdvances を小さく上書きするケースがあるため、
  // ここで動的に整合性を確保する。
  const p: GenerationParams = {
    baseSeed: 1n,
    offset: 0n,
    maxAdvances: 3000,
    maxResults: 2000,
    version: 'B',
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
    shinyCharm: false,
    isShinyLocked: false,
    newGame: true,
    withSave: true,
    memoryLink: false,
    ...overrides,
  };
  if (p.maxResults > p.maxAdvances) p.maxResults = p.maxAdvances;
  return p;
}

function waitFor<T>(register: (cb: (v: T)=>unknown) => unknown, predicate: (v: T)=>boolean, timeoutMs = 4000): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
    register((v: T) => { if (predicate(v)) { clearTimeout(timer); resolve(v); } });
  });
}

describe('generation state machine sequences (mocked worker)', () => {
  it('Seq1 start -> complete', async () => {
    const mgr = new GenerationWorkerManager();
    const c = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1000 }));
    await c;
    expect(mgr.getStatus()).toBe('idle');
  });

  it('Seq2 start -> stop results in stopped completion', async () => {
    const mgr = new GenerationWorkerManager();
    const completion = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 5000 }));
    mgr.stop();
    const result = await completion;
    expect(result.reason).toBe('stopped');
    expect(mgr.getStatus()).toBe('idle');
  });

  it('Seq3 restart after stop', async () => {
    const mgr = new GenerationWorkerManager();
    const first = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1200 }));
    mgr.stop();
    await first;
    const second = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 800 }));
    const result = await second;
    expect(result.reason).toBe('max-advances');
    expect(mgr.getStatus()).toBe('idle');
  });

  it('Seq4 ignores stop when idle', () => {
    const mgr = new GenerationWorkerManager();
    mgr.stop();
    expect(mgr.getStatus()).toBe('idle');
  });
});
