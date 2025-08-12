// Generation Worker (Phase3/4 skeleton)
// 目的: プロトコル応答/状態遷移/ダミー進捗のみ。WASMと結果生成は後続 TODO。

import {
  type GenerationWorkerRequest,
  type GenerationWorkerResponse,
  type GenerationParams,
  type GenerationProgress,
  type GenerationResult,
  validateGenerationParams,
  FIXED_PROGRESS_INTERVAL_MS,
} from '@/types/generation';
import { parseFromWasmRaw } from '@/lib/generation/raw-parser';
// 既存 wasm ビルド (vite alias '@/wasm' 前提) ― 追加アダプタは作らず直接利用
import { BWGenerationConfig, SeedEnumerator } from '@/wasm/wasm_pkg';

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
  enumerator: SeedEnumerator | null;
  config: BWGenerationConfig | null;
  // Task6: バッチ送信はまだ行わない。後続 Task7 で flush 予定。
  pendingResults: GenerationResult[];
  stopped: boolean; // STOP 要求フラグ (完全処理は後続 Task10)
  batchIndex: number;
  cumulativeResults: number;
  shinyFound: boolean;
}

// DedicatedWorkerGlobalScope 型は lib.dom.d.ts で提供されるが build 設定により認識しない場合があるため any fallback
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ctx: any = self as any;

function post(message: GenerationWorkerResponse) {
  ctx.postMessage(message);
}

const DEFAULT_PROGRESS_INTERVAL = FIXED_PROGRESS_INTERVAL_MS; // 仕様: 固定間隔
const DUMMY_STEP = 1000; // 1 tick で進めるダミーadvance数 (テスト容易性優先)

const state: InternalState = {
  params: null,
  progress: {
    processedAdvances: 0,
    totalAdvances: 0,
    resultsCount: 0,
    elapsedMs: 0,
    throughput: 0,
    etaMs: 0,
    status: 'idle',
  },
  startTime: null,
  intervalId: null,
  enumerator: null,
  config: null,
  pendingResults: [],
  stopped: false,
  batchIndex: 0,
  cumulativeResults: 0,
  shinyFound: false,
};

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<GenerationWorkerRequest>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case 'START_GENERATION':
        handleStart(msg.params);
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
        // no-op
        break;
    }
  } catch (e: any) {
    post({ type: 'ERROR', message: e?.message || String(e), category: 'RUNTIME', fatal: false });
  }
};

function handleStart(params: GenerationParams) {
  if (state.progress.status === 'running') {
    // 二重起動は一旦無視 (後でSTOP推奨通知検討)
    return;
  }
  const errors = validateGenerationParams(params);
  if (errors.length) {
    post({ type: 'ERROR', message: errors.join(', '), category: 'VALIDATION', fatal: false });
    return;
  }
  cleanupInterval();

  const progressInterval = DEFAULT_PROGRESS_INTERVAL;
  state.params = params;
  state.startTime = performance.now();
  state.progress = {
    processedAdvances: 0,
    totalAdvances: params.maxAdvances,
    resultsCount: 0,
    elapsedMs: 0,
    throughput: 0,
    etaMs: 0,
    status: 'running',
  };
  state.stopped = false;
  state.pendingResults = [];
  state.batchIndex = 0;
  state.cumulativeResults = 0;
  state.shinyFound = false;
  try {
    // Task6: WASM 初期化 & SeedEnumerator 準備 (後続機能は未実装)
    state.config = new BWGenerationConfig(
      versionToWasm(params.version),
      params.encounterType,
      params.tid,
      params.sid,
      params.syncEnabled,
      params.syncNatureId,
    );
    // count は maxAdvances - offset を一旦全量渡す (早期終了は Task8)
    const totalCount = Number(params.maxAdvances - Number(params.offset));
    state.enumerator = new SeedEnumerator(
      params.baseSeed,
      BigInt(params.offset),
      totalCount,
      state.config,
    );
  } catch (e: any) {
    post({ type: 'ERROR', message: e?.message || String(e), category: 'WASM_INIT', fatal: true });
    return;
  }
  // 列挙ループを tick 内に統合せず、軽量 step を実行 (batch は次タスク)

  state.intervalId = setInterval(tick, progressInterval) as unknown as number;
  // 直後に初回進捗
  post({ type: 'PROGRESS', payload: { ...state.progress } });
}

function tick() {
  if (!state.params || state.progress.status !== 'running') return;
  // STOP 要求されていたら後続タスクでの完全処理まで進捗のみ抑制
  if (state.stopped) return;
  advanceEnumerationChunk();
  const p = state.progress;
  // processedAdvances は advanceEnumerationChunk 内で更新済
  const now = performance.now();
  p.elapsedMs = now - (state.startTime || now);
  if (p.elapsedMs > 0) {
    p.throughput = p.processedAdvances / (p.elapsedMs / 1000);
  }
  const remaining = p.totalAdvances - p.processedAdvances;
  p.etaMs = p.throughput > 0 ? (remaining / p.throughput) * 1000 : 0;

  post({ type: 'PROGRESS', payload: { ...p } });

  if (p.processedAdvances >= p.totalAdvances) {
    complete('max-advances');
  }
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
  if (state.progress.status === 'idle' || state.progress.status === 'stopped' || state.progress.status === 'completed') return;
  state.stopped = true; // Task6: 即 STOPPED 送信せず (既存挙動保持) – 既存テスト維持のため従来通り送信
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
  // FINAL FLUSH (残り分送信)
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

// ===== Task6 Core Enumeration (no batching, no early termination) =====
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
    if (!raw) break; // 枯渇 (自然終了)

    // WASM Raw -> UnresolvedPokemonData
    let unresolved;
    try {
      unresolved = parseFromWasmRaw(raw);
    } catch (e) {
      // パース失敗は致命的扱い
      complete('error');
      return;
    }

    const advanceVal = p.processedAdvances + i + 1; // 現仕様: 1-based インデックス
    const result: GenerationResult = { ...unresolved, advance: advanceVal };

    const isShiny = result.shiny_type !== 0;
    if (isShiny && !state.shinyFound) state.shinyFound = true;

    // 結果保持ガード (容量抑制) - stopOnCap=false の場合は保存せず走査継続
    if (p.resultsCount < params.maxResults) {
      state.pendingResults.push(result);
      p.resultsCount += 1;
    }

    produced++;

    // バッチ閾値
    if (state.pendingResults.length >= params.batchSize) {
      flushBatch(false);
    }

    // 早期終了判定 (優先順位: first-shiny > max-results)
    if (!earlyReason) {
      if (params.stopAtFirstShiny && isShiny) {
        earlyReason = 'first-shiny';
      } else if (params.stopOnCap && p.resultsCount >= params.maxResults) {
        earlyReason = 'max-results';
      }
    }

    if (earlyReason) {
      // 以降の列挙は打ち切り
      break;
    }
  }

  p.processedAdvances += produced;

  if (earlyReason) {
    complete(earlyReason);
    return;
  }

  if (p.processedAdvances >= p.totalAdvances) {
    complete('max-advances');
  }
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

// 終了時クリーンアップ (念のため)
ctx.onclose = () => cleanupInterval();

export {}; // モジュール化
