import { describe, it, expect } from 'vitest';
import { initWasm, getWasm } from '@/lib/core/wasm-interface';
import { parseFromWasmRaw } from '@/lib/generation/raw-parser';
import { domainEncounterTypeToWasm } from '@/lib/core/mapping/encounter-type';
import { domainGameModeToWasm } from '@/lib/core/mapping/game-mode';
import type { DomainEncounterType } from '@/types/domain';
import { deriveDomainGameMode, type GenerationParams } from '@/types/generation';

// Worker 利用可否でスキップ制御 (CI Node 環境では未定義想定)
if (typeof Worker === 'undefined') {
  describe.skip('generation-worker early termination (no Worker env)', () => {
    it('skipped', () => { expect(true).toBe(true); });
  });
} else {
  function createWorker() {
    return new Worker(new URL('@/workers/generation-worker.ts', import.meta.url), { type: 'module' });
  }

  type WorkerMsg = { type?: string; payload?: unknown };
  async function waitFor<T extends WorkerMsg>(worker: Worker, predicate: (m: WorkerMsg) => m is T, timeoutMs?: number): Promise<T>;
  async function waitFor(worker: Worker, predicate: (m: WorkerMsg) => boolean, timeoutMs?: number): Promise<WorkerMsg>;
  async function waitFor(worker: Worker, predicate: (m: WorkerMsg) => boolean, timeoutMs = 15000): Promise<WorkerMsg> {
    return new Promise<WorkerMsg>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
      worker.addEventListener('message', (ev) => {
        const data = ev.data as WorkerMsg;
        if (predicate(data)) { clearTimeout(timer); resolve(data); }
      });
    });
  }

  const VERSION_TO_WASM: Record<GenerationParams['version'], number> = { B: 0, W: 1, B2: 2, W2: 3 };

  async function findFirstShinyAdvance(template: GenerationParams, searchLimit = 50000): Promise<number> {
    await initWasm();
    const wasm = getWasm();
    const encounter = domainEncounterTypeToWasm(template.encounterType as DomainEncounterType);
    const cfg = new wasm.BWGenerationConfig(
      VERSION_TO_WASM[template.version],
      encounter,
      template.tid,
      template.sid,
      template.syncEnabled,
      template.syncNatureId,
      template.isShinyLocked,
      template.shinyCharm,
    );
    const domainMode = deriveDomainGameMode(template);
    const wasmMode = domainGameModeToWasm(domainMode);
    const effectiveOffset = BigInt(wasm.calculate_game_offset(template.baseSeed, wasmMode)) + template.offset;
    const enumerator = new wasm.SeedEnumerator(template.baseSeed, effectiveOffset, searchLimit, cfg);
    for (let i = 0; i < searchLimit; i++) {
      const raw = enumerator.next_pokemon();
      if (!raw) break;
      const parsed = parseFromWasmRaw(raw);
      if ((parsed.shiny_type ?? 0) !== 0) {
        return i;
      }
    }
    throw new Error('Shiny Pokemon not found within search limit');
  }

  // first-shiny テスト: 最初の個体を強制的に色違い化する TID/SID を構築
  // SV = (tid ^ sid ^ pid_hi ^ pid_lo) < 8 で色違い。tid=0 として sid = pid_hi ^ pid_lo なら SV=0。
  function _shinySidForPid(pid: number): number {
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
      const paramsBase: GenerationParams = {
        baseSeed,
        offset: 0n,
        maxAdvances: 60000,
        maxResults: 200,
        version: 'B' as const,
        encounterType: 0,
        tid: 0,
        sid: 0,
        syncEnabled: false,
        syncNatureId: 0,
        shinyCharm: false,
        isShinyLocked: false,
        stopAtFirstShiny: true,
        stopOnCap: false,
        batchSize: 25,
        newGame: true,
        withSave: true,
        memoryLink: false,
      };
      const firstShinyAdvance = await findFirstShinyAdvance(paramsBase);
      const params: GenerationParams = {
        ...paramsBase,
        maxAdvances: firstShinyAdvance + 5,
        stopAtFirstShiny: true,
        stopOnCap: false,
      };
      const w = createWorker();
      await waitFor(w, m => m.type === 'READY');
      w.postMessage({ type: 'START_GENERATION', params });
      const complete = await waitFor(w, (m): m is { type: 'COMPLETE'; payload: { reason: string; shinyFound: boolean; processedAdvances: number } } => m.type === 'COMPLETE');
      expect(complete.payload.reason).toBe('first-shiny');
      expect(complete.payload.shinyFound).toBe(true);
      expect(complete.payload.processedAdvances).toBe(firstShinyAdvance + 1);
      w.terminate();
    });

    it('terminates on max-results cap when stopOnCap=true', async () => {
      const baseSeed = fixedSeed();
      const params: GenerationParams = {
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
        shinyCharm: false,
        isShinyLocked: false,
        stopAtFirstShiny: false,
        stopOnCap: true,
        batchSize: 10,
        newGame: true,
        withSave: true,
        memoryLink: false,
      };
      const w = createWorker();
      await waitFor(w, m => m.type === 'READY');
      w.postMessage({ type: 'START_GENERATION', params });
      const complete = await waitFor(w, (m): m is { type: 'COMPLETE'; payload: { reason: string; resultsCount: number; shinyFound: boolean } } => m.type === 'COMPLETE');
      expect(complete.payload.reason).toBe('max-results');
      expect(complete.payload.resultsCount).toBe(3);
      expect(complete.payload.shinyFound).toBe(false);
      w.terminate();
    });
  });
}
