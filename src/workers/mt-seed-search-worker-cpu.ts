/**
 * MT Seed 32bit全探索 CPU Worker
 *
 * WASMのSIMD最適化版MT19937を使用してMT Seedを全探索する。
 * セグメントベースの検索パターンを採用:
 * - 指定されたジョブ範囲に対してWASMの mt_seed_search_segment を呼び出す
 * - 進捗を定期的に報告
 * - pause/resume/stopに対応
 */
import type {
  MtSeedSearchWorkerRequest,
  MtSeedSearchWorkerResponse,
  MtSeedSearchJob,
  MtSeedSearchResult,
  MtSeedMatch,
  MtSeedSearchProgress,
  MtSeedSearchCompletion,
  IvCode,
} from '@/types/mt-seed-search';
import { decodeIvCode } from '@/types/mt-seed-search';
import {
  initWasm,
  getWasm,
  isWasmReady,
} from '@/lib/core/wasm-interface';

interface InternalState {
  job: MtSeedSearchJob | null;
  running: boolean;
  stopRequested: boolean;
  isPaused: boolean;
  pauseResolve: (() => void) | null;
}

const state: InternalState = {
  job: null,
  running: false,
  stopRequested: false,
  isPaused: false,
  pauseResolve: null,
};

const ctx = self as typeof self & { onclose?: () => void };
const post = (message: MtSeedSearchWorkerResponse) => ctx.postMessage(message);

// 初期化完了通知
post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<MtSeedSearchWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START':
          await handleStart(msg.job);
          break;
        case 'PAUSE':
          handlePause();
          break;
        case 'RESUME':
          handleResume();
          break;
        case 'STOP':
          state.stopRequested = true;
          // 一時停止中の場合は解除して終了させる
          if (state.isPaused && state.pauseResolve) {
            state.pauseResolve();
            state.pauseResolve = null;
            state.isPaused = false;
          }
          break;
        default:
          break;
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      post({ type: 'ERROR', message, category: 'RUNTIME' });
    }
  })();
};

/**
 * 一時停止処理
 */
function handlePause(): void {
  if (!state.running || state.isPaused) {
    return;
  }
  state.isPaused = true;
}

/**
 * 再開処理
 */
function handleResume(): void {
  if (!state.running || !state.isPaused) {
    return;
  }
  state.isPaused = false;
  if (state.pauseResolve) {
    state.pauseResolve();
    state.pauseResolve = null;
  }
}

/**
 * 一時停止中は待機する
 */
async function waitWhilePaused(): Promise<void> {
  // イベントループに制御を戻してPAUSEメッセージを処理可能にする
  await new Promise<void>((resolve) => setTimeout(resolve, 0));

  if (!state.isPaused) {
    return;
  }
  await new Promise<void>((resolve) => {
    state.pauseResolve = resolve;
  });
}

/**
 * 検索開始
 */
async function handleStart(job: MtSeedSearchJob) {
  if (state.running) {
    return;
  }

  state.job = job;
  state.stopRequested = false;
  state.running = true;

  const startTime = performance.now();

  try {
    await ensureWasm();

    // 検索実行
    const result = await executeSearch(job, startTime);

    // 完了通知
    const completion: MtSeedSearchCompletion = {
      reason: state.stopRequested ? 'stopped' : 'finished',
      totalProcessed: result.processedCount,
      totalMatches: result.matchesFound,
      elapsedMs: performance.now() - startTime,
    };
    post({ type: 'COMPLETE', payload: completion });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'ERROR', message, category: 'RUNTIME' });
  } finally {
    cleanupState();
  }
}

async function ensureWasm() {
  if (!isWasmReady()) {
    await initWasm();
  }
}

/**
 * チャンクサイズ（1チャンクあたりの検索Seed数）
 * 大きすぎるとUIが固まり、小さすぎるとオーバーヘッドが増える
 */
const CHUNK_SIZE = 0x100000; // 約100万Seed/チャンク

/**
 * 進捗報告間隔（ミリ秒）
 */
const PROGRESS_INTERVAL_MS = 500;

/**
 * 検索実行
 */
async function executeSearch(
  job: MtSeedSearchJob,
  startTime: number
): Promise<{ processedCount: number; matchesFound: number }> {
  const wasm = getWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasmAny = wasm as any;

  if (!wasmAny.mt_seed_search_segment) {
    throw new Error('mt_seed_search_segment not exposed in WASM');
  }

  const { searchRange, ivCodes, mtAdvances, jobId } = job;
  const { start, end } = searchRange;

  // IVコードをUint32Arrayに変換
  const ivCodesArray = new Uint32Array(ivCodes);

  // 全体の検索範囲
  const totalCount = end - start + 1;
  let processedCount = 0;
  let matchesFound = 0;
  let lastProgressTime = startTime;

  // チャンクに分割して検索
  let chunkStart = start;

  while (chunkStart <= end) {
    // 一時停止中は待機
    await waitWhilePaused();

    if (state.stopRequested) {
      break;
    }

    // チャンク終端を計算（オーバーフロー対策）
    const chunkEnd = Math.min(chunkStart + CHUNK_SIZE - 1, end);

    // WASM呼び出し（SIMD最適化版）
    const resultArray: Uint32Array = wasmAny.mt_seed_search_segment(
      chunkStart,
      chunkEnd,
      mtAdvances,
      ivCodesArray
    );

    // 結果をパース [seed0, code0, seed1, code1, ...]
    if (resultArray.length > 0) {
      const matches: MtSeedMatch[] = [];
      for (let i = 0; i < resultArray.length; i += 2) {
        const mtSeed = resultArray[i];
        const ivCode: IvCode = resultArray[i + 1];
        const ivSet = decodeIvCode(ivCode);
        matches.push({ mtSeed, ivCode, ivSet });
        matchesFound++;
      }

      // 結果を送信
      const result: MtSeedSearchResult = { jobId, matches };
      post({ type: 'RESULTS', payload: result });
    }

    // 進捗更新
    processedCount += chunkEnd - chunkStart + 1;

    // 進捗報告（一定間隔で）
    const now = performance.now();
    if (now - lastProgressTime >= PROGRESS_INTERVAL_MS) {
      const progress: MtSeedSearchProgress = {
        jobId,
        processedCount,
        totalCount,
        elapsedMs: now - startTime,
        matchesFound,
      };
      post({ type: 'PROGRESS', payload: progress });
      lastProgressTime = now;
    }

    // 次のチャンクへ（オーバーフロー対策）
    if (chunkEnd >= end) {
      break;
    }
    chunkStart = chunkEnd + 1;
  }

  // 最終進捗報告
  const finalProgress: MtSeedSearchProgress = {
    jobId,
    processedCount,
    totalCount,
    elapsedMs: performance.now() - startTime,
    matchesFound,
  };
  post({ type: 'PROGRESS', payload: finalProgress });

  return { processedCount, matchesFound };
}

function cleanupState() {
  state.running = false;
  state.job = null;
  state.stopRequested = false;
  state.isPaused = false;
  state.pauseResolve = null;
}

ctx.onclose = () => {
  cleanupState();
};

export {};
