import { describe, it, expect, beforeAll } from 'vitest';
import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationCompletion, GenerationProgress } from '@/types/generation';

// --- Mock Worker (Node 環境用) ---
class MockWorker {
  public onmessage: ((ev: MessageEvent)=>void) | null = null;
  public onerror: ((ev: any)=>void) | null = null;
  private interval: any = null;
  private running = false;
  private paused = false;
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
        this.running = true; this.paused = false; this.advances = 0; this.startTs = Date.now();
        this.startLoop();
        break;
      case 'PAUSE':
        if (this.running && !this.paused) { this.paused = true; this.dispatch({ type: 'PAUSED' }); }
        break;
      case 'RESUME':
        if (this.running && this.paused) { this.paused = false; this.dispatch({ type: 'RESUMED' }); }
        break;
      case 'STOP':
        if (this.running) {
          this.cleanup();
          const prog = this.buildProgress('stopped');
          this.dispatch({ type: 'STOPPED', payload: { reason: 'stopped', processedAdvances: prog.processedAdvances, resultsCount: prog.resultsCount, elapsedMs: prog.elapsedMs, shinyFound: false } });
        }
        break;
    }
  }
  terminate() { this.cleanup(); }
  private startLoop() {
    const step = () => {
      if (!this.running) return;
      if (!this.paused) {
        this.advances += 250; // 固定増分
        const prog = this.buildProgress('running');
        this.dispatch({ type: 'PROGRESS', payload: prog });
        if (this.advances >= this.params.maxAdvances) {
          this.running = false;
          const finalProg = this.buildProgress('completed');
            this.dispatch({ type: 'COMPLETE', payload: { reason: 'max-advances', processedAdvances: finalProg.processedAdvances, resultsCount: finalProg.resultsCount, elapsedMs: finalProg.elapsedMs, shinyFound: false } });
          this.cleanup();
          return;
        }
      }
      this.interval = setTimeout(step, 25);
    };
    step();
  }
  private buildProgress(status: GenerationProgress['status']): GenerationProgress {
    const elapsedMs = Date.now() - this.startTs;
    return {
      processedAdvances: Math.min(this.advances, this.params.maxAdvances),
      totalAdvances: this.params.maxAdvances,
      resultsCount: 0,
      elapsedMs,
      throughput: elapsedMs > 0 ? (this.advances / (elapsedMs / 1000)) : 0,
      etaMs: 0,
      status,
    };
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
  // NOTE: Validationルール: maxResults <= maxAdvances, batchSize <= maxAdvances
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
    batchSize: 5000,
    newGame: true,
    withSave: true,
    memoryLink: false,
    ...overrides,
  };
  if (p.maxResults > p.maxAdvances) p.maxResults = p.maxAdvances;
  if (p.batchSize > p.maxAdvances) p.batchSize = p.maxAdvances;
  return p;
}

function waitFor<T>(register: (cb: (v: T)=>unknown) => unknown, predicate: (v: T)=>boolean, timeoutMs = 4000): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
    register((v: T) => { if (predicate(v)) { clearTimeout(timer); resolve(v); } });
  });
}
const waitSomeProgress = (mgr: GenerationWorkerManager) => waitFor<any>(mgr.onProgress.bind(mgr), p => p.processedAdvances > 0);

describe('generation state machine sequences (mocked worker)', () => {
  it('Seq1 start -> complete', async () => {
    const mgr = new GenerationWorkerManager();
    const c = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1000 }));
    await c;
    expect(['completed','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq2 start -> pause -> resume -> complete', async () => {
    const mgr = new GenerationWorkerManager();
    const c = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1200 }));
    await waitSomeProgress(mgr);
    mgr.pause();
    await new Promise(r => setTimeout(r, 80));
    mgr.resume();
    await c;
    expect(['completed','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq3 start -> pause -> resume -> stop', async () => {
    const mgr = new GenerationWorkerManager();
    const s = waitFor<any>(mgr.onStopped.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 2500 }));
    await waitSomeProgress(mgr);
    mgr.pause();
    await new Promise(r => setTimeout(r, 50));
    mgr.resume();
    await new Promise(r => setTimeout(r, 50));
    mgr.stop();
    const stopped = await s;
    expect(stopped.reason).toBe('stopped');
    expect(['stopped','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq4 start -> stop (early)', async () => {
    const mgr = new GenerationWorkerManager();
    const s = waitFor<any>(mgr.onStopped.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 3000 }));
    await waitSomeProgress(mgr);
    mgr.stop();
    await s;
    expect(['stopped','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq5 start -> pause -> stop', async () => {
    const mgr = new GenerationWorkerManager();
    const s = waitFor<any>(mgr.onStopped.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 3000 }));
    await waitSomeProgress(mgr);
    mgr.pause();
    await new Promise(r => setTimeout(r, 40));
    mgr.stop();
    await s;
    expect(['stopped','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq6 start -> pause (double) -> resume (double) -> complete', async () => {
    const mgr = new GenerationWorkerManager();
    const c = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 900 }));
    await waitSomeProgress(mgr);
    mgr.pause(); mgr.pause();
    await new Promise(r => setTimeout(r, 30));
    mgr.resume(); mgr.resume();
    await c;
    expect(['completed','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq7 start -> stop -> start (restart)', async () => {
    const mgr = new GenerationWorkerManager();
    const s1 = waitFor<any>(mgr.onStopped.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 4000 }));
    await waitSomeProgress(mgr); mgr.stop(); await s1;
    const c2 = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 800 }));
    await c2;
    expect(['completed','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq8 start -> pause -> stop -> start', async () => {
    const mgr = new GenerationWorkerManager();
    const s1 = waitFor<any>(mgr.onStopped.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 4000 }));
    await waitSomeProgress(mgr); mgr.pause(); await new Promise(r => setTimeout(r, 40)); mgr.stop(); await s1;
    const c2 = waitFor<GenerationCompletion>(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1000 }));
    await c2;
    expect(['completed','idle'].includes(mgr.getStatus())).toBe(true);
  });
  it('Seq9 invalid ops (resume/pause/stop idle) are no-ops', async () => {
    const mgr = new GenerationWorkerManager();
    mgr.resume(); mgr.pause(); mgr.stop();
    expect(mgr.getStatus()).toBe('idle');
  });
});
