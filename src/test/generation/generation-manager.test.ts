import { describe, it, expect } from 'vitest';
import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationResultsPayload } from '@/types/generation';

const NO_WORKER = typeof Worker === 'undefined';
if (NO_WORKER) {
  describe.skip('GenerationWorkerManager (no Worker env)', () => { it('skipped', () => expect(true).toBe(true)); });
} else {

function params(overrides: Partial<GenerationParams> = {}): GenerationParams {
  return {
    baseSeed: 1n,
    offset: 0n,
    maxAdvances: 1200,
    maxResults: 2000,
    version: 'B',
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: false,
    shinyCharm: false,
    isShinyLocked: false,
    newGame: true,
    withSave: true,
    memoryLink: false,
    ...overrides,
  };
}

function waitForEvent<T>(register: (cb: (v: T)=>void)=>unknown, predicate: (v: T)=>boolean, timeoutMs=5000): Promise<T> {
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
    expect(mgr.getStatus()).toBe('idle');
    const secondComplete = waitForEvent(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1000 }));
    await secondComplete;
  });

  it('emits results callbacks before completion', async () => {
    const mgr = new GenerationWorkerManager();
    const results = waitForEvent<GenerationResultsPayload>(mgr.onResults.bind(mgr), payload => payload.results.length >= 0);
    const completion = waitForEvent(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 1200 }));
    await results;
    await completion;
    expect(mgr.getStatus()).toBe('idle');
  });

  it('double start throws', async () => {
    const mgr = new GenerationWorkerManager();
    await mgr.start(params());
    let threw = false;
    try { await mgr.start(params()); } catch { threw = true; }
    expect(threw).toBe(true);
    mgr.stop();
  });

  it('stop transitions to stopped completion', async () => {
    const mgr = new GenerationWorkerManager();
    const completion = waitForEvent(mgr.onComplete.bind(mgr), () => true);
    await mgr.start(params({ maxAdvances: 8000 }));
    mgr.stop();
    const result = await completion;
    expect(result.reason).toBe('stopped');
    expect(mgr.getStatus()).toBe('idle');
  });
});
}
