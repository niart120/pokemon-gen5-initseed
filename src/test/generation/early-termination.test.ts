import { describe, it, expect } from 'vitest';
import { initWasm, getWasm } from '@/lib/core/wasm-interface';

// Worker 利用可否でスキップ制御 (CI Node 環境では未定義想定)
if (typeof Worker === 'undefined') {
  describe.skip('generation-worker early termination (no Worker env)', () => {
    it('skipped', () => { expect(true).toBe(true); });
  });
} else {
  function createWorker() {
    return new Worker(new URL('@/workers/generation-worker.ts', import.meta.url), { type: 'module' });
  }

  async function waitFor(worker: Worker, predicate: (m: any) => boolean, timeoutMs = 2000): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
      worker.addEventListener('message', (ev) => {
        if (predicate(ev.data)) { clearTimeout(timer); resolve(ev.data); }
      });
    });
  }

  // 指定 baseSeed の最初の PID を取得
  async function firstPidAndShinySid(baseSeed: bigint): Promise<{ pid: number; sid: number }> {
    await initWasm();
    const { BWGenerationConfig, GameVersion, PokemonGenerator } = getWasm();
    const cfg = new BWGenerationConfig(GameVersion.B, 0, 0, 0, false, 0);
    const raw = PokemonGenerator.generate_single_pokemon_bw(baseSeed, cfg);
  const pid = raw.get_pid; // wasm getter property
    const pidHi = (pid >>> 16) & 0xffff; const pidLo = pid & 0xffff;
    const sid = pidHi ^ pidLo; // tid=0 と組で SV=0 → shiny
    return { pid, sid };
  }

  // first-shiny テスト: 最初の個体を強制的に色違い化する TID/SID を構築
  // SV = (tid ^ sid ^ pid_hi ^ pid_lo) < 8 で色違い。tid=0 として sid = pid_hi ^ pid_lo なら SV=0。
  function shinySidForPid(pid: number): number {
    const pidHi = (pid >>> 16) & 0xffff;
    const pidLo = pid & 0xffff;
    return pidHi ^ pidLo; // 0..65535 に収まる
  }

  // max-results テスト用: 先頭3体が非色違いとなる baseSeed を探索 (tid=sid=0 で半 XOR >=8)
  // シンプル化: max-results テスト用 seed は固定 (先頭3体が shiny でない保証は不要: shiny が出ても stopAtFirstShiny=false なので max-results 先行)
  function fixedSeed(): bigint { return 6000n; }

  describe('generation-worker early termination', () => {
    it('terminates on first shiny when stopAtFirstShiny=true', async () => {
  const baseSeed = 1234n;
  const { pid, sid } = await firstPidAndShinySid(baseSeed);
      const params = {
        baseSeed,
        offset: 0n,
        maxAdvances: 100, // 十分に大きければ良い
        maxResults: 50,
        version: 'B' as const,
        encounterType: 0,
        tid: 0,
        sid,
        syncEnabled: false,
        syncNatureId: 0,
        stopAtFirstShiny: true,
        stopOnCap: true, // 併用可
        batchSize: 25,
      };
      const w = createWorker();
      await waitFor(w, m => m.type === 'READY');
      w.postMessage({ type: 'START_GENERATION', params });
      const complete: any = await waitFor(w, m => m.type === 'COMPLETE');
      expect(complete.payload.reason).toBe('first-shiny');
      expect(complete.payload.shinyFound).toBe(true);
      expect(complete.payload.processedAdvances).toBe(1); // 最初の1体で停止
      w.terminate();
    });

    it('terminates on max-results cap when stopOnCap=true', async () => {
  const baseSeed = fixedSeed();
      const params = {
        baseSeed,
        offset: 0n,
        maxAdvances: 100,
        maxResults: 3,
        version: 'B' as const,
        encounterType: 0,
        tid: 0,
        sid: 0,
        syncEnabled: false,
        syncNatureId: 0,
        stopAtFirstShiny: false,
        stopOnCap: true,
        batchSize: 10,
      };
      const w = createWorker();
      await waitFor(w, m => m.type === 'READY');
      w.postMessage({ type: 'START_GENERATION', params });
      const complete: any = await waitFor(w, m => m.type === 'COMPLETE');
      expect(complete.payload.reason).toBe('max-results');
      expect(complete.payload.resultsCount).toBe(3);
      expect(complete.payload.shinyFound).toBe(false);
      w.terminate();
    });
  });
}
