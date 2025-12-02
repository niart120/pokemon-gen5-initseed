/**
 * MT Seed 32bit全探索 GPU Worker
 *
 * WebGPUを使用してMT Seedを全探索する。
 * GPUエンジンをラップし、ジョブの分割実行と進捗報告を行う。
 */
import type {
  MtSeedSearchWorkerRequest,
  MtSeedSearchWorkerResponse,
  MtSeedSearchJob,
  MtSeedSearchResult,
  MtSeedSearchProgress,
  MtSeedSearchCompletion,
} from '@/types/mt-seed-search';
import {
  createMtSeedSearchGpuEngine,
  type MtSeedSearchGpuEngine,
} from '@/lib/webgpu/mt-seed-search';

interface InternalState {
  job: MtSeedSearchJob | null;
  engine: MtSeedSearchGpuEngine | null;
  running: boolean;
  stopRequested: boolean;
  isPaused: boolean;
  pauseResolve: (() => void) | null;
}

const state: InternalState = {
  job: null,
  engine: null,
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
    // GPUエンジンを初期化
    if (!state.engine) {
      state.engine = createMtSeedSearchGpuEngine();
      if (!state.engine.isAvailable()) {
        throw new Error('WebGPU is not available');
      }
      await state.engine.initialize();
    }

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
    const category = message.includes('WebGPU') ? 'GPU_INIT' : 'RUNTIME';
    post({ type: 'ERROR', message, category });
  } finally {
    cleanupState();
  }
}

/**
 * チャンクサイズ（Worker側での分割）
 * GPUエンジンが内部でディスパッチ分割するため、ここでは大きめのチャンクで分割
 * → エンジンの maxMessagesPerDispatch に基づいて動的に決定
 */

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
  const engine = state.engine;
  if (!engine) {
    throw new Error('GPU engine not initialized');
  }

  const { searchRange, ivCodes, mtAdvances, jobId } = job;
  const { start, end } = searchRange;

  // エンジンの制限値を取得してチャンクサイズを決定
  const jobLimits = engine.getJobLimits();
  // Worker側で1 dispatch分ずつ分割（Engine内での再分割を回避）
  const chunkSize = jobLimits
    ? jobLimits.maxMessagesPerDispatch
    : 0x1000000; // フォールバック: 約1600万

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
    const chunkEnd = Math.min(chunkStart + chunkSize - 1, end);

    // GPUで検索実行
    const chunkJob: MtSeedSearchJob = {
      searchRange: { start: chunkStart, end: chunkEnd },
      ivCodes,
      mtAdvances,
      jobId,
    };

    const result = await engine.executeJob(chunkJob);

    // 結果を送信
    if (result.matches.length > 0) {
      const searchResult: MtSeedSearchResult = {
        jobId,
        matches: result.matches,
      };
      post({ type: 'RESULTS', payload: searchResult });
      matchesFound += result.matches.length;
    }

    // 進捗更新
    processedCount += result.processedCount;

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
  // エンジンは再利用のため保持
}

ctx.onclose = () => {
  state.engine?.dispose();
  state.engine = null;
  cleanupState();
};

export {};
