/**
 * MT Seed Boot Timing Worker - MT Seed起動時間検索用Worker
 *
 * 複数のMT Seedに対応する起動時間条件を検索する。
 * セグメントベースのパターンを採用:
 * - TypeScript側で timer0 × vcount × keyCode のセグメントループを実装
 * - 各セグメントに対して MtSeedBootTimingSearchIterator を作成
 * - 結果はストリーミングで送信
 */
import type {
  MtSeedBootTimingWorkerRequest,
  MtSeedBootTimingWorkerResponse,
  MtSeedBootTimingSearchParams,
  MtSeedBootTimingSearchResult,
  MtSeedBootTimingCompletion,
  MtSeedBootTimingProgress,
  WasmMtSeedBootTimingSearchResult,
} from '@/types/mt-seed-boot-timing-search';
import { generateValidKeyCodes } from '@/lib/utils/key-input';
import {
  initWasm,
  getWasm,
  isWasmReady,
  type DSConfigJs,
  type SegmentParamsJs,
  type TimeRangeParamsJs,
  type SearchRangeParamsJs,
} from '@/lib/core/wasm-interface';
import { keyCodeToNames } from '@/lib/utils/key-input';
import romParameters from '@/data/rom-parameters';

interface InternalState {
  params: MtSeedBootTimingSearchParams | null;
  running: boolean;
  stopRequested: boolean;
  isPaused: boolean;
  pauseResolve: (() => void) | null;
}

const state: InternalState = {
  params: null,
  running: false,
  stopRequested: false,
  isPaused: false,
  pauseResolve: null,
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type WasmAny = any;

/**
 * DS設定パラメータを構築
 */
function buildDSConfig(
  wasmAny: WasmAny,
  params: MtSeedBootTimingSearchParams,
  nazo: readonly number[]
): DSConfigJs {
  return new wasmAny.DSConfigJs(
    new Uint8Array(params.macAddress),
    new Uint32Array(nazo),
    params.hardware
  );
}

/**
 * セグメントパラメータを構築
 */
function buildSegmentParams(
  wasmAny: WasmAny,
  timer0: number,
  vcount: number,
  keyCode: number
): SegmentParamsJs {
  return new wasmAny.SegmentParamsJs(timer0, vcount, keyCode);
}

/**
 * 時刻範囲パラメータを構築
 */
function buildTimeRangeParams(
  wasmAny: WasmAny,
  timeRange: MtSeedBootTimingSearchParams['timeRange']
): TimeRangeParamsJs {
  return new wasmAny.TimeRangeParamsJs(
    timeRange.hour.start,
    timeRange.hour.end,
    timeRange.minute.start,
    timeRange.minute.end,
    timeRange.second.start,
    timeRange.second.end
  );
}

/**
 * 検索範囲パラメータを構築
 */
function buildSearchRangeParams(
  wasmAny: WasmAny,
  searchStartDate: Date,
  rangeSeconds: number
): SearchRangeParamsJs {
  return new wasmAny.SearchRangeParamsJs(
    searchStartDate.getFullYear(),
    searchStartDate.getMonth() + 1,
    searchStartDate.getDate(),
    rangeSeconds
  );
}

const ctx = self as typeof self & { onclose?: () => void };
const post = (message: MtSeedBootTimingWorkerResponse) => ctx.postMessage(message);

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<MtSeedBootTimingWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START_SEARCH':
          await handleStart(msg.params);
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
      post({ type: 'ERROR', message, category: 'RUNTIME', fatal: false });
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
  post({ type: 'PROGRESS', payload: createPausedProgress() });
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
 * 一時停止中の進捗情報を作成
 */
function createPausedProgress(): MtSeedBootTimingProgress {
  return {
    processedCombinations: 0,
    totalCombinations: 0,
    foundCount: 0,
    progressPercent: 0,
    elapsedMs: 0,
    estimatedRemainingMs: 0,
  };
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

async function handleStart(params: MtSeedBootTimingSearchParams) {
  if (state.running) {
    return;
  }

  state.params = params;
  state.stopRequested = false;
  state.running = true;

  const startTime = performance.now();

  try {
    await ensureWasm();

    // 検索実行
    const result = await executeSearch(params, startTime);

    // 完了通知
    const completion: MtSeedBootTimingCompletion = {
      reason: state.stopRequested ? 'stopped' : 'completed',
      processedCombinations: result.processedSegments,
      totalCombinations: result.totalSegments,
      resultsCount: result.resultsCount,
      elapsedMs: performance.now() - startTime,
    };
    post({ type: 'COMPLETE', payload: completion });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'ERROR', message, category: 'RUNTIME', fatal: true });
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
 * Nazo値を解決
 */
function resolveNazoValue(
  romVersion: MtSeedBootTimingSearchParams['romVersion'],
  romRegion: MtSeedBootTimingSearchParams['romRegion']
): readonly number[] {
  const romData = romParameters[romVersion];
  if (!romData) {
    throw new Error(`Unknown ROM version: ${romVersion}`);
  }
  const regionData = romData[romRegion];
  if (!regionData) {
    throw new Error(`Unknown region for ${romVersion}: ${romRegion}`);
  }
  return regionData.nazo;
}

/**
 * WASM結果をドメイン型に変換
 */
function convertWasmResult(
  wasmResult: WasmMtSeedBootTimingSearchResult,
  macAddress: readonly [number, number, number, number, number, number]
): MtSeedBootTimingSearchResult {
  return {
    boot: {
      datetime: new Date(
        Date.UTC(
          wasmResult.year,
          wasmResult.month - 1,
          wasmResult.day,
          wasmResult.hour,
          wasmResult.minute,
          wasmResult.second
        )
      ),
      timer0: wasmResult.timer0,
      vcount: wasmResult.vcount,
      keyCode: wasmResult.keyCode,
      keyInputNames: keyCodeToNames(wasmResult.keyCode),
      macAddress,
    },
    mtSeedHex: wasmResult.mtSeedHex,
    mtSeed: wasmResult.mtSeed,
    lcgSeedHex: wasmResult.lcgSeedHex,
  };
}

/**
 * 検索結果の統計情報
 */
interface SearchResult {
  resultsCount: number;
  processedSegments: number;
  totalSegments: number;
}

/**
 * 検索実行（セグメントベースパターン）
 *
 * timer0 × vcount × keyCode のセグメントループを実装し、
 * 各セグメントに対して MtSeedBootTimingSearchIterator を作成する。
 */
async function executeSearch(
  params: MtSeedBootTimingSearchParams,
  startTime: number
): Promise<SearchResult> {
  const wasm = getWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasmAny = wasm as any;

  if (!wasmAny.MtSeedBootTimingSearchIterator) {
    throw new Error('MtSeedBootTimingSearchIterator not exposed in WASM');
  }

  // nazo値を解決
  const nazo = resolveNazoValue(params.romVersion, params.romRegion);

  // キーコード一覧を生成（TS実装）
  const keyCodes: number[] = generateValidKeyCodes(params.keyInputMask);

  // dateRangeから検索期間を計算
  const { dateRange } = params;
  const searchStartDate = new Date(
    dateRange.startYear,
    dateRange.startMonth - 1,
    dateRange.startDay
  );

  // rangeSecondsを計算
  let rangeSeconds: number;
  if (params.rangeSeconds !== undefined) {
    rangeSeconds = params.rangeSeconds;
  } else {
    const endDate = new Date(
      dateRange.endYear,
      dateRange.endMonth - 1,
      dateRange.endDay
    );
    const totalDays = Math.max(
      1,
      Math.floor(
        (endDate.getTime() - searchStartDate.getTime()) / (1000 * 60 * 60 * 24)
      ) + 1
    );
    rangeSeconds = totalDays * 24 * 60 * 60;
  }

  // セグメント数を計算（進捗報告用）
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  const totalSegments = timer0Count * vcountCount * keyCodes.length;

  // 総処理秒数（進捗計算用）: 全セグメント × 各セグメントの秒数
  const totalSecondsToProcess = totalSegments * rangeSeconds;

  // イテレータパラメータ
  const RESULT_LIMIT = 4;
  const CHUNK_SECONDS = 3600 * 24 * 5 * totalSegments; // 1チャンクあたりの秒数（5日分×セグメント数）
  const PROGRESS_INTERVAL_MS = 500;

  let resultsCount = 0;
  let processedSegments = 0;
  let totalProcessedSeconds = 0; // 累積処理済み秒数
  let lastProgressTime = startTime;

  // Target seeds を Uint32Array に変換
  const targetSeedsArray = new Uint32Array(params.targetSeeds);

  // セグメントループ: timer0 × vcount × keyCode
  for (
    let timer0 = params.timer0Range.min;
    timer0 <= params.timer0Range.max && resultsCount < params.maxResults;
    timer0++
  ) {
    for (
      let vcount = params.vcountRange.min;
      vcount <= params.vcountRange.max && resultsCount < params.maxResults;
      vcount++
    ) {
      for (const keyCode of keyCodes) {
        // 一時停止中は待機
        await waitWhilePaused();

        if (state.stopRequested || resultsCount >= params.maxResults) {
          return { resultsCount, processedSegments, totalSegments };
        }

        // 構造化パラメータを構築
        const dsConfig = buildDSConfig(wasmAny, params, nazo);
        const segmentParams = buildSegmentParams(wasmAny, timer0, vcount, keyCode);
        const timeRangeParams = buildTimeRangeParams(wasmAny, params.timeRange);
        const searchRangeParams = buildSearchRangeParams(
          wasmAny,
          searchStartDate,
          rangeSeconds
        );

        // イテレータを作成
        let iterator: {
          isFinished: boolean;
          progress: number; // 0.0〜1.0のセグメント内進捗
          next_batch: (
            limit: number,
            chunk: number
          ) => { to_array: () => unknown[] };
          free?: () => void;
        };

        try {
          iterator = new wasmAny.MtSeedBootTimingSearchIterator(
            dsConfig,
            segmentParams,
            timeRangeParams,
            searchRangeParams,
            targetSeedsArray
          );
        } catch (e) {
          // エラーが発生した場合はスキップ
          console.error('Failed to create MtSeedBootTimingSearchIterator:', e);
          processedSegments++;
          continue;
        }

        try {
          // イテレータループ（時刻方向）
          let processedSecondsInSegment = 0;

          while (!iterator.isFinished && resultsCount < params.maxResults) {
            // 一時停止中は待機
            await waitWhilePaused();

            if (state.stopRequested) break;

            const batchResults = iterator.next_batch(RESULT_LIMIT, CHUNK_SECONDS);
            const resultsArray = batchResults.to_array();
            // 処理済み秒数を加算（最後のチャンクは端数になる可能性があるが、次セグメントでリセットされるので問題なし）
            processedSecondsInSegment = Math.min(processedSecondsInSegment + CHUNK_SECONDS, rangeSeconds);

            // 結果をストリーミング送信
            if (resultsArray.length > 0) {
              const convertedResults: MtSeedBootTimingSearchResult[] = [];
              for (
                let i = 0;
                i < resultsArray.length && resultsCount < params.maxResults;
                i++
              ) {
                const wasmResult = resultsArray[i] as WasmMtSeedBootTimingSearchResult;
                convertedResults.push(
                  convertWasmResult(wasmResult, params.macAddress)
                );
                resultsCount++;
              }

              post({
                type: 'RESULTS',
                payload: { results: convertedResults, batchIndex: processedSegments },
              });
            }

            // 定期的にイベントループに制御を戻す（STOP受信のため）
            const now = performance.now();
            if (now - lastProgressTime >= PROGRESS_INTERVAL_MS) {
              const elapsedMs = now - startTime;
              // 現在の処理済み秒数（完了セグメント + 現在セグメント内）
              const currentProcessedSeconds = totalProcessedSeconds + processedSecondsInSegment;
              // 進捗パーセント（秒数ベース）
              const progressPercent = totalSecondsToProcess > 0
                ? (currentProcessedSeconds / totalSecondsToProcess) * 100
                : 0;
              const estimatedRemainingMs =
                currentProcessedSeconds > 0
                  ? (elapsedMs / currentProcessedSeconds) *
                    (totalSecondsToProcess - currentProcessedSeconds)
                  : 0;

              const progress: MtSeedBootTimingProgress = {
                processedCombinations: processedSegments,
                totalCombinations: totalSegments,
                foundCount: resultsCount,
                progressPercent,
                elapsedMs,
                estimatedRemainingMs,
                processedSeconds: currentProcessedSeconds,
              };
              post({ type: 'PROGRESS', payload: progress });
              lastProgressTime = now;
            }
          }
        } finally {
          iterator.free?.();
        }

        // セグメント完了時に累積秒数を更新
        totalProcessedSeconds += rangeSeconds;
        processedSegments++;
      }
    }
  }

  // 最終進捗報告
  const elapsedMs = performance.now() - startTime;
  const progress: MtSeedBootTimingProgress = {
    processedCombinations: processedSegments,
    totalCombinations: totalSegments,
    foundCount: resultsCount,
    progressPercent: 100,
    elapsedMs,
    estimatedRemainingMs: 0,
  };
  post({ type: 'PROGRESS', payload: progress });

  return { resultsCount, processedSegments, totalSegments };
}

function cleanupState() {
  state.running = false;
  state.params = null;
  state.stopRequested = false;
  state.isPaused = false;
  state.pauseResolve = null;
}

ctx.onclose = () => {
  cleanupState();
};

export {};
