// Generation Worker (Phase3/4 skeleton)
// 目的: プロトコル応答/状態遷移/ダミー進捗のみ。WASMと結果生成は後続 TODO。

import {
  type GenerationWorkerRequest,
  type GenerationWorkerResponse,
  type GenerationParams,
  type GenerationProgress,
  validateGenerationParams,
  FIXED_PROGRESS_INTERVAL_MS,
} from '@/types/generation';

interface InternalState {
  params: GenerationParams | null;
  progress: GenerationProgress;
  startTime: number | null;
  intervalId: number | null;
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

  // TODO: WASM 初期化 & SeedEnumerator 準備
  // TODO: バッチ生成 & RESULT_BATCH 送信
  // TODO: stopAtFirstShiny, stopOnCap 実装

  state.intervalId = setInterval(tick, progressInterval) as unknown as number;
  // 直後に初回進捗
  post({ type: 'PROGRESS', payload: { ...state.progress } });
}

function tick() {
  if (!state.params || state.progress.status !== 'running') return;
  const p = state.progress;
  p.processedAdvances = Math.min(p.totalAdvances, p.processedAdvances + DUMMY_STEP);
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
  state.progress.status = 'completed';
  post({
    type: 'COMPLETE',
    payload: {
      reason,
      processedAdvances: state.progress.processedAdvances,
      resultsCount: state.progress.resultsCount,
      elapsedMs: state.progress.elapsedMs,
      shinyFound: false, // TODO: shiny 検出時更新
    },
  });
}

function cleanupInterval() {
  if (state.intervalId != null) {
    clearInterval(state.intervalId);
    state.intervalId = null;
  }
}

// 終了時クリーンアップ (念のため)
ctx.onclose = () => cleanupInterval();

export {}; // モジュール化
