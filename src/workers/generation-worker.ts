// Generation Worker (Phase3/4 skeleton)
// 目的: プロトコル応答/状態遷移/進捗計測 + WASM 列挙

import {
  type GenerationWorkerRequest,
  type GenerationWorkerResponse,
  type GenerationParams,
  type GenerationProgress,
  type GenerationResult,
  validateGenerationParams,
  FIXED_PROGRESS_INTERVAL_MS,
  deriveDomainGameMode,
} from '@/types/generation';
import { parseFromWasmRaw } from '@/lib/generation/raw-parser';
import { initWasm, getWasm, isWasmReady } from '@/lib/core/wasm-interface';
import { domainGameModeToWasm } from '@/lib/core/mapping/game-mode';
import { domainEncounterTypeToWasm } from '@/lib/core/mapping/encounter-type';
import type { DomainEncounterType } from '@/types/domain';

// WASM コンストラクタ最小型 (必要最小限のみ表現)
interface BWGenerationConfigCtor {
  new(
    version: number,
    encounterType: number,
    tid: number,
    sid: number,
    syncEnabled: boolean,
    syncNatureId: number,
    isShinyLocked: boolean,
    shinyCharm: boolean,
  ): object;
}
interface SeedEnumeratorCtor { new(baseSeed: bigint, offset: bigint, count: number, config: object): SeedEnumeratorInstance; }
interface SeedEnumeratorInstance { next_pokemon(): unknown; }

let BWGenerationConfig: BWGenerationConfigCtor | undefined;
let SeedEnumerator: SeedEnumeratorCtor | undefined;

// BW/BW2 version string -> wasm GameVersion enum (wasm側: B=0,W=1,B2=2,W2=3)
function versionToWasm(v: GenerationParams['version']): number {
  switch (v) {
    case 'B': return 0;
    case 'W': return 1;
    case 'B2': return 2;
    case 'W2': return 3;
    default: return 0;
  }
}

interface InternalState {
  params: GenerationParams | null;
  progress: GenerationProgress;
  startTime: number | null;
  intervalId: number | null;
  enumerator: SeedEnumeratorInstance | null;
  config: object | null;
  emaThroughput: number | null;
  pendingResults: GenerationResult[];
  stopped: boolean;
  batchIndex: number;
  cumulativeResults: number;
  shinyFound: boolean;
  baseAdvance: number;
}

// self の型を拡張 (WebWorker lib 未設定環境でもコンパイル可能にする)
const ctx = self as typeof self & { onclose?: () => void };
const post = (message: GenerationWorkerResponse) => ctx.postMessage(message);

const DEFAULT_PROGRESS_INTERVAL = FIXED_PROGRESS_INTERVAL_MS;
const DUMMY_STEP = 1000;

const blankProgress = (): GenerationProgress => ({
  processedAdvances: 0,
  totalAdvances: 0,
  resultsCount: 0,
  elapsedMs: 0,
  throughput: 0,
  throughputRaw: 0,
  throughputEma: 0,
  etaMs: 0,
  status: 'idle',
});

const state: InternalState = {
  params: null,
  progress: blankProgress(),
  startTime: null,
  intervalId: null,
  enumerator: null,
  config: null,
  emaThroughput: null,
  pendingResults: [],
  stopped: false,
  batchIndex: 0,
  cumulativeResults: 0,
  shinyFound: false,
  baseAdvance: 0,
};

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<GenerationWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START_GENERATION':
          await handleStart(msg.params, msg.staticEncounterId);
          break;
        case 'PAUSE':
          handlePause();
          break;
        case 'RESUME':
          handleResume();
          break;
        case 'STOP':
          handleStop();
          break;
        default:
          break;
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      post({ type: 'ERROR', message, category: 'RUNTIME', fatal: false });
    }
  })();
};

async function handleStart(params: GenerationParams, staticEncounterId?: string | null) {
  if (state.progress.status === 'running') return;
  const errors = validateGenerationParams(params, { staticEncounterId });
  if (errors.length) {
    post({ type: 'ERROR', message: errors.join(', '), category: 'VALIDATION', fatal: false });
    return;
  }
  cleanupInterval();
  state.params = params;
  state.startTime = performance.now();
  const baseAdvance = Number(params.offset);
  const totalAdvances = params.maxAdvances - baseAdvance;
  if (totalAdvances <= 0) {
    post({ type: 'ERROR', message: 'maxAdvances must be greater than offset', category: 'VALIDATION', fatal: false });
    return;
  }
  state.progress = {
    processedAdvances: 0,
    totalAdvances,
    resultsCount: 0,
    elapsedMs: 0,
    throughput: 0,
    throughputRaw: 0,
    throughputEma: 0,
    etaMs: 0,
    status: 'running',
  };
  state.emaThroughput = null;
  state.stopped = false;
  state.pendingResults = [];
  state.batchIndex = 0;
  state.cumulativeResults = 0;
  state.shinyFound = false;
  try {
    if (!isWasmReady()) await initWasm();
  const wasm = getWasm() as unknown as {
    BWGenerationConfig: BWGenerationConfigCtor;
    SeedEnumerator?: SeedEnumeratorCtor;
    calculate_game_offset(initial_seed: bigint, mode: number): number;
  };
  BWGenerationConfig = wasm.BWGenerationConfig;
  SeedEnumerator = wasm.SeedEnumerator;
  if (!SeedEnumerator) throw new Error('SeedEnumerator not exposed');
  const wasmEncounterType = domainEncounterTypeToWasm(params.encounterType as DomainEncounterType);
    state.config = new BWGenerationConfig(
      versionToWasm(params.version),
      wasmEncounterType,
      params.tid,
      params.sid,
      params.syncEnabled,
      params.syncNatureId,
      params.isShinyLocked,
      params.shinyCharm,
    );
    const domainMode = deriveDomainGameMode(params);
    const wasmMode = domainGameModeToWasm(domainMode);
    const gameOffset = BigInt(wasm.calculate_game_offset(params.baseSeed, wasmMode));
    const effectiveOffset = gameOffset + params.offset;
    state.baseAdvance = baseAdvance;
    state.enumerator = new SeedEnumerator(
      params.baseSeed,
      effectiveOffset,
      totalAdvances,
      state.config,
    );
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'ERROR', message, category: 'WASM_INIT', fatal: true });
    return;
  }
  state.intervalId = setInterval(tick, DEFAULT_PROGRESS_INTERVAL) as unknown as number;
  post({ type: 'PROGRESS', payload: { ...state.progress } });
}

function tick() {
  if (!state.params || state.progress.status !== 'running') return;
  if (state.stopped) return;
  advanceEnumerationChunk();
  const p = state.progress;
  const now = performance.now();
  p.elapsedMs = now - (state.startTime || now);
  if (p.elapsedMs > 0) {
    const raw = p.processedAdvances / (p.elapsedMs / 1000);
    p.throughputRaw = raw;
    const ALPHA = 0.2;
    state.emaThroughput = state.emaThroughput == null ? raw : (ALPHA * raw + (1 - ALPHA) * state.emaThroughput);
    p.throughputEma = state.emaThroughput;
    p.throughput = raw; // 互換
  }
  const remaining = p.totalAdvances - p.processedAdvances;
  const basis = p.throughputEma && p.throughputEma > 0 ? p.throughputEma : (p.throughputRaw || 0);
  p.etaMs = basis > 0 ? (remaining / basis) * 1000 : 0;
  post({ type: 'PROGRESS', payload: { ...p } });
  if (p.processedAdvances >= p.totalAdvances) complete('max-advances');
}

function handlePause() {
  if (state.progress.status !== 'running') return;
  state.progress.status = 'paused';
  cleanupInterval();
  post({ type: 'PAUSED' });
}

function handleResume() {
  if (state.progress.status !== 'paused' || !state.params) return;
  state.progress.status = 'running';
  state.intervalId = setInterval(tick, DEFAULT_PROGRESS_INTERVAL) as unknown as number;
  post({ type: 'RESUMED' });
}

function handleStop() {
  if (['idle', 'stopped', 'completed'].includes(state.progress.status)) return;
  state.stopped = true;
  cleanupInterval();
  state.progress.status = 'stopped';
  post({
    type: 'STOPPED',
    payload: {
      reason: 'stopped',
      processedAdvances: state.progress.processedAdvances,
      resultsCount: state.progress.resultsCount,
      elapsedMs: state.progress.elapsedMs,
      shinyFound: false,
    },
  });
}

function complete(reason: 'max-advances' | 'max-results' | 'first-shiny' | 'stopped' | 'error') {
  cleanupInterval();
  flushBatch(true);
  state.progress.status = 'completed';
  if (process.env.NODE_ENV !== 'production') {
    if (state.cumulativeResults !== state.progress.resultsCount) {
      console.error('[generation-worker] final results mismatch', {
        cumulativeResults: state.cumulativeResults,
        resultsCount: state.progress.resultsCount,
      });
    }
  }
  post({
    type: 'COMPLETE',
    payload: {
      reason,
      processedAdvances: state.progress.processedAdvances,
      resultsCount: state.progress.resultsCount,
      elapsedMs: state.progress.elapsedMs,
      shinyFound: state.shinyFound,
    },
  });
}

function cleanupInterval() {
  if (state.intervalId != null) {
    clearInterval(state.intervalId);
    state.intervalId = null;
  }
}

function advanceEnumerationChunk() {
  if (!state.enumerator || !state.params) return;
  const params = state.params;
  const p = state.progress;
  let steps = DUMMY_STEP;
  const remainingNeeded = p.totalAdvances - p.processedAdvances;
  if (remainingNeeded <= 0) return;
  if (steps > remainingNeeded) steps = remainingNeeded;

  let produced = 0;
  let earlyReason: 'first-shiny' | 'max-results' | null = null;
  for (let i = 0; i < steps; i++) {
    const raw = state.enumerator.next_pokemon();
    if (!raw) break;
    let unresolved;
    try {
      unresolved = parseFromWasmRaw(raw);
    } catch {
      complete('error');
      return;
    }
  const advanceVal = state.baseAdvance + p.processedAdvances + i;
  const result: GenerationResult = { ...unresolved, advance: advanceVal };
  const isShiny = (result.shiny_type ?? 0) !== 0;
    if (isShiny && !state.shinyFound) state.shinyFound = true;
    if (p.resultsCount < params.maxResults) {
      state.pendingResults.push(result);
      p.resultsCount += 1;
    }
    produced++;
    if (state.pendingResults.length >= params.batchSize) flushBatch(false);
    if (!earlyReason) {
      if (params.stopAtFirstShiny && isShiny) earlyReason = 'first-shiny';
      else if (params.stopOnCap && p.resultsCount >= params.maxResults) earlyReason = 'max-results';
    }
    if (earlyReason) break;
  }
  p.processedAdvances += produced;
  if (earlyReason) {
    complete(earlyReason);
    return;
  }
  if (p.processedAdvances >= p.totalAdvances) complete('max-advances');
}

function flushBatch(force: boolean) {
  if (!state.params) return;
  if (state.pendingResults.length === 0) return;
  if (!force && state.pendingResults.length < state.params.batchSize) return;
  const batch = state.pendingResults.splice(0, state.pendingResults.length);
  state.batchIndex += 1;
  state.cumulativeResults += batch.length;
  if (process.env.NODE_ENV !== 'production') {
    if (state.cumulativeResults > state.progress.resultsCount) {
      console.error('[generation-worker] cumulativeResults exceeded resultsCount', {
        cumulativeResults: state.cumulativeResults,
        resultsCount: state.progress.resultsCount,
        batchLength: batch.length,
      });
    }
  }
  post({
    type: 'RESULT_BATCH',
    payload: {
      batchIndex: state.batchIndex,
      batchSize: batch.length,
      results: batch,
      cumulativeResults: state.cumulativeResults,
    },
  });
}

ctx.onclose = () => cleanupInterval();

export {};
