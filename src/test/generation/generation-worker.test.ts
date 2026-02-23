import { describe, it, expect } from 'vitest';

if (typeof Worker === 'undefined') {
  describe.skip('generation-worker (no Worker support)', () => {
    it('skipped', () => { expect(true).toBe(true); });
  });
} else {
  function makeParams(maxAdvances = 3000) {
    return {
      baseSeed: 1n,
      offset: 0n,
      maxAdvances,
      maxResults: 2000,
      version: 'B' as const,
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
    };
  }

  function createWorker() {
    return new Worker(new URL('@/workers/generation-worker.ts', import.meta.url), { type: 'module' });
  }

  function waitFor(worker: Worker, predicate: (msg: any)=>boolean, timeoutMs = 5000): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(()=>{ reject(new Error('timeout')); }, timeoutMs);
      worker.addEventListener('message', (ev) => {
        if (predicate(ev.data)) { clearTimeout(timer); resolve(ev.data); }
      });
    });
  }

  describe('generation-worker simplified protocol', () => {
    it('emits RESULTS then COMPLETE', async () => {
      const w = createWorker();
      await waitFor(w, m => m.type === 'READY');
      const params = { ...makeParams(200), maxResults: 50 };
      w.postMessage({ type: 'START_GENERATION', params });
      const resultsMsg = await waitFor(w, m => m.type === 'RESULTS');
      expect(Array.isArray(resultsMsg.payload.results)).toBe(true);
      const complete = await waitFor(w, m => m.type === 'COMPLETE');
      expect(complete.payload.reason).toBe('max-advances');
      expect(complete.payload.resultsCount).toBe(resultsMsg.payload.results.length);
      w.terminate();
    });

    it('STOP request sets completion reason to stopped', async () => {
      const w = createWorker();
      await waitFor(w, m => m.type === 'READY');
      w.postMessage({ type: 'START_GENERATION', params: makeParams(10_000) });
      // issue stop almost immediately
      w.postMessage({ type: 'STOP' });
      const complete = await waitFor(w, m => m.type === 'COMPLETE');
      expect(complete.payload.reason).toBe('stopped');
      w.terminate();
    });
  });
}
