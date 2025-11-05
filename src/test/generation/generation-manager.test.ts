import { describe, it, expect } from 'vitest';
import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams } from '@/types/generation';

const NO_WORKER = typeof Worker === 'undefined';
if (NO_WORKER) {
  describe.skip('GenerationWorkerManager (no Worker env)', () => { it('skipped', () => expect(true).toBe(true)); });
} else {

function params(overrides: Partial<GenerationParams> = {}): GenerationParams {
  return {
    baseSeed: 1n,
    offset: 0n,
    maxAdvances: 1200,
    maxResults: 1000,
    version: 'B',
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
  batchSize: 5000,
  newGame: true,
  noSave: false,
  memoryLink: false,
    ...overrides,
  };
}

function waitForEvent<T>(register: (cb: (v: T)=>void)=>unknown, predicate: (v: T)=>boolean, timeoutMs=2000): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(()=>reject(new Error('timeout')), timeoutMs);
    register((v: T) => { if (predicate(v)) { clearTimeout(timer); resolve(v); } });
  });
}

describe('GenerationWorkerManager', () => {
  it('start -> complete terminates worker', async () => {
    const mgr = new GenerationWorkerManager();
    const complete = waitForEvent(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1000 }));
    await complete;
    expect(mgr.getStatus() === 'completed' || mgr.getStatus() === 'idle').toBe(true);
    await mgr.start(params({ maxAdvances: 1000 }));
  });

  it('pause/resume updates status', async () => {
    const mgr = new GenerationWorkerManager();
    await mgr.start(params({ maxAdvances: 4000 }));
  await waitForEvent<any>(mgr.onProgress.bind(mgr), (p: any) => p.processedAdvances > 0);
    mgr.pause();
    await new Promise(r => setTimeout(r, 120));
    mgr.resume();
    await new Promise(r => setTimeout(r, 120));
    expect(mgr.isRunning()).toBe(true);
    mgr.stop();
  });

  it('double start throws', async () => {
    const mgr = new GenerationWorkerManager();
    await mgr.start(params());
    let threw = false;
    try { await mgr.start(params()); } catch { threw = true; }
    expect(threw).toBe(true);
    mgr.stop();
  });
});
}
