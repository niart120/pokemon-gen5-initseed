import { describe, it, expect } from 'vitest';

// Node 環境では Worker 未定義のためスキップ（ブラウザ環境E2Eで補完）
if (typeof Worker === 'undefined') {
  describe.skip('generation-worker skeleton (no Worker in env)', () => {
    it('skipped', () => { expect(true).toBe(true); });
  });
} else {
  // 実行可能環境でのみ以下を定義

function makeParams(maxAdvances = 3000) {
  return {
    baseSeed: 1n,
    offset: 0n,
    maxAdvances,
    maxResults: 1000,
    version: 'B' as const,
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
    progressIntervalMs: 50,
    batchSize: 1000,
  };
}

function createWorker() {
  return new Worker(new URL('@/workers/generation-worker.ts', import.meta.url), { type: 'module' });
}

function waitFor(worker: Worker, predicate: (msg: any)=>boolean, timeoutMs = 2000): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(()=>{ reject(new Error('timeout')); }, timeoutMs);
    worker.addEventListener('message', (ev) => {
      if (predicate(ev.data)) { clearTimeout(timer); resolve(ev.data); }
    });
  });
}

describe('generation-worker skeleton', () => {
  it('RUN → COMPLETE', async () => {
    const w = createWorker();
    await waitFor(w, m => m.type === 'READY');
    w.postMessage({ type: 'START_GENERATION', params: makeParams(1000) });
    const complete = await waitFor(w, m => m.type === 'COMPLETE');
    expect(complete.payload.reason).toBe('max-advances');
    w.terminate();
  });

  it('PAUSE/RESUME progression', async () => {
    const w = createWorker();
    await waitFor(w, m => m.type === 'READY');
    w.postMessage({ type: 'START_GENERATION', params: makeParams(5000) });
    let firstProgress: any;
    firstProgress = await waitFor(w, m => m.type === 'PROGRESS' && m.payload.processedAdvances > 0);
    w.postMessage({ type: 'PAUSE' });
    await waitFor(w, m => m.type === 'PAUSED');
    const pausedValue = firstProgress.payload.processedAdvances;
    await new Promise(r => setTimeout(r, 120));
    let increased = false;
    w.addEventListener('message', ev => { if (ev.data.type === 'PROGRESS' && ev.data.payload.processedAdvances > pausedValue) increased = true; });
    await new Promise(r => setTimeout(r, 120));
    expect(increased).toBe(false);
    w.postMessage({ type: 'RESUME' });
    await waitFor(w, m => m.type === 'RESUMED');
    const afterResume = await waitFor(w, m => m.type === 'PROGRESS' && m.payload.processedAdvances > pausedValue);
    expect(afterResume.payload.processedAdvances).toBeGreaterThan(pausedValue);
    w.terminate();
  });

  it('STOP produces STOPPED not COMPLETE', async () => {
    const w = createWorker();
    await waitFor(w, m => m.type === 'READY');
    w.postMessage({ type: 'START_GENERATION', params: makeParams(10_000) });
    await waitFor(w, m => m.type === 'PROGRESS');
    w.postMessage({ type: 'STOP' });
    const stopped = await waitFor(w, m => m.type === 'STOPPED');
    expect(stopped.payload.reason).toBe('stopped');
    w.terminate();
  });
});
}
