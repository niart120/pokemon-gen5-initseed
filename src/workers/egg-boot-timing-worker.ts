/**
 * Egg Boot Timing Worker - 孵化乱数起動時間検索用Worker
 */
import type {
  EggBootTimingWorkerRequest,
  EggBootTimingWorkerResponse,
  EggBootTimingSearchParams,
  EggBootTimingSearchResult,
  EggBootTimingCompletion,
  EggBootTimingProgress,
  WasmEggBootTimingSearchResult,
} from '@/types/egg-boot-timing-search';
import type { IvSet, HiddenPowerInfo } from '@/types/egg';
import { EggGameMode } from '@/types/egg';
import {
  initWasm,
  getWasm,
  isWasmReady,
  type DSConfigJs,
  type SegmentParamsJs,
  type TimeRangeParamsJs,
  type SearchRangeParamsJs,
} from '@/lib/core/wasm-interface';
import { keyCodeToNames, generateValidKeyCodes } from '@/lib/utils/key-input';
import romParameters from '@/data/rom-parameters';

interface InternalState {
  params: EggBootTimingSearchParams | null;
  running: boolean;
  stopRequested: boolean;
}

const state: InternalState = {
  params: null,
  running: false,
  stopRequested: false,
};

/**
 * EggGameMode から WASM GameMode への変換
 */
function eggGameModeToWasm(mode: EggGameMode): number {
  switch (mode) {
    case EggGameMode.BwNew:
      return 0;
    case EggGameMode.BwContinue:
      return 2;
    case EggGameMode.Bw2New:
      return 5;
    case EggGameMode.Bw2Continue:
      return 7;
    default:
      return 2;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type WasmAny = any;

/**
 * DS設定パラメータを構築
 */
function buildDSConfig(
  wasmAny: WasmAny,
  params: EggBootTimingSearchParams,
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
  timeRange: EggBootTimingSearchParams['timeRange']
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
const post = (message: EggBootTimingWorkerResponse) => ctx.postMessage(message);

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<EggBootTimingWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START_SEARCH':
          await handleStart(msg.params);
          break;
        case 'STOP':
          state.stopRequested = true;
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

async function handleStart(params: EggBootTimingSearchParams) {
  if (state.running) {
    return;
  }

  state.params = params;
  state.stopRequested = false;
  state.running = true;

  const startTime = performance.now();

  try {
    await ensureWasm();

    // 検索実行（dateRange全体を一括処理）
    const result = await executeSearch(params, startTime);

    // 完了通知
    const completion: EggBootTimingCompletion = {
      reason: state.stopRequested ? 'stopped' : 'completed',
      processedCombinations: result.processedSegments,
      totalCombinations: result.totalSegments,
      resultsCount: result.resultsCount,
      elapsedMs: performance.now() - startTime,
    };
    post({ type: 'COMPLETE', payload: completion });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const stack = e instanceof Error ? e.stack : undefined;
    console.error('[EggBootTimingWorker] Search error:', message);
    if (stack) {
      console.error('[EggBootTimingWorker] Stack trace:', stack);
    }
    console.error('[EggBootTimingWorker] Params:', JSON.stringify(params, (k, v) => typeof v === 'bigint' ? v.toString() : v));
    post({ type: 'ERROR', message, category: 'WASM_INIT', fatal: true });
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
 * nazo値を解決
 */
function resolveNazoValue(
  romVersion: string,
  romRegion: string
): readonly [number, number, number, number, number] {
  const versionParams =
    romParameters[romVersion as keyof typeof romParameters];
  if (!versionParams) {
    throw new Error(`Unknown ROM version: ${romVersion}`);
  }
  const regionParams =
    versionParams[romRegion as keyof typeof versionParams];
  if (!regionParams) {
    throw new Error(`Unknown ROM region: ${romRegion}`);
  }
  return regionParams.nazo;
}

function buildEverstone(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  wasm: any,
  plan: EggBootTimingSearchParams['conditions']['everstone']
) {
  if (plan.type === 'none') {
    return wasm.EverstonePlanJs.None;
  } else {
    return wasm.EverstonePlanJs.fixed(plan.nature);
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildFilter(wasm: any, filter: EggBootTimingSearchParams['filter']) {
  const wasmFilter = new wasm.IndividualFilterJs();

  // フィルタがnullの場合はデフォルト（全pass-through）のフィルタを返す
  if (!filter) return wasmFilter;

  // ivRanges の設定（各ステータスのIV範囲を設定）
  if (filter.ivRanges && Array.isArray(filter.ivRanges)) {
    for (let i = 0; i < 6; i++) {
      const range = filter.ivRanges[i];
      if (range && typeof range.min === 'number' && typeof range.max === 'number') {
        wasmFilter.set_iv_range(i, range.min, range.max);
      }
    }
  }

  if (filter.nature !== undefined) {
    wasmFilter.set_nature(filter.nature);
  }
  if (filter.gender !== undefined) {
    wasmFilter.set_gender(genderToWasm(filter.gender));
  }
  if (filter.ability !== undefined) {
    wasmFilter.set_ability(filter.ability);
  }
  if (filter.shinyFilterMode !== undefined) {
    wasmFilter.set_shiny_filter_mode(shinyFilterModeToWasm(filter.shinyFilterMode));
  }
  if (filter.hiddenPowerType !== undefined) {
    wasmFilter.set_hidden_power_type(filter.hiddenPowerType);
  }
  if (filter.hiddenPowerPower !== undefined) {
    wasmFilter.set_hidden_power_power(filter.hiddenPowerPower);
  }

  return wasmFilter;
}

function genderToWasm(gender: string): number {
  switch (gender) {
    case 'male':
      return 0;
    case 'female':
      return 1;
    case 'genderless':
      return 2;
    default:
      return 0;
  }
}

/**
 * ShinyFilterMode を WASM u8 値に変換
 * 0 = All, 1 = Shiny, 2 = Star, 3 = Square, 4 = NonShiny
 */
function shinyFilterModeToWasm(mode: string): number {
  switch (mode) {
    case 'all':
      return 0;
    case 'shiny':
      return 1;
    case 'star':
      return 2;
    case 'square':
      return 3;
    case 'non-shiny':
      return 4;
    default:
      return 0;
  }
}

function wasmGenderToDomain(
  wasmGender: number
): 'male' | 'female' | 'genderless' {
  switch (wasmGender) {
    case 0:
      return 'male';
    case 1:
      return 'female';
    case 2:
      return 'genderless';
    default:
      return 'male';
  }
}

function parseHiddenPower(hpKnown: boolean, hpType: number, hpPower: number): HiddenPowerInfo {
  if (!hpKnown) {
    return { type: 'unknown' };
  }
  return {
    type: 'known',
    hpType,
    power: hpPower,
  };
}

function convertWasmResult(
  wasmResult: WasmEggBootTimingSearchResult,
  macAddress: readonly [number, number, number, number, number, number]
): EggBootTimingSearchResult {
  const advance = Number(wasmResult.advance);
  const ivs: IvSet = [
    wasmResult.ivs[0] ?? 0,
    wasmResult.ivs[1] ?? 0,
    wasmResult.ivs[2] ?? 0,
    wasmResult.ivs[3] ?? 0,
    wasmResult.ivs[4] ?? 0,
    wasmResult.ivs[5] ?? 0,
  ];

  return {
    boot: {
      datetime: new Date(
        Date.UTC(
          wasmResult.year,
          wasmResult.month - 1,
          wasmResult.date,
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
    lcgSeedHex: wasmResult.lcgSeedHex,
    egg: {
      advance,
      isStable: wasmResult.isStable,
      egg: {
        lcgSeedHex: wasmResult.lcgSeedHex,
        mtSeedHex: wasmResult.mtSeedHex,
        ivs,
        nature: wasmResult.nature,
        gender: wasmGenderToDomain(wasmResult.gender),
        ability: wasmResult.ability as 0 | 1 | 2,
        shiny: wasmResult.shiny as 0 | 1 | 2,
        pid: wasmResult.pid,
        hiddenPower: parseHiddenPower(
          wasmResult.hpKnown,
          wasmResult.hpType,
          wasmResult.hpPower
        ),
      },
    },
    isStable: wasmResult.isStable,
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
 * dateRange全体を一括検索（セグメントベースパターン）
 *
 * WebGPU検索と同様のセグメント分割パターンを採用:
 * TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
 * 各セグメントに対して EggBootTimingSearchIterator を作成する。
 * 結果はストリーミングで送信し、メモリ効率を確保する。
 *
 * async 関数として実装し、定期的にイベントループに制御を戻すことで
 * STOP メッセージを受け取れるようにする。
 */
async function executeSearch(
  params: EggBootTimingSearchParams,
  startTime: number
): Promise<SearchResult> {
  const wasm = getWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasmAny = wasm as any;

  if (!wasmAny.EggBootTimingSearchIterator) {
    throw new Error('EggBootTimingSearchIterator not exposed in WASM');
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

  // rangeSecondsが明示的に指定されている場合はそれを使用（チャンク分割時）
  // 指定されていない場合はdateRangeから計算（後方互換性）
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
      Math.floor((endDate.getTime() - searchStartDate.getTime()) / (1000 * 60 * 60 * 24)) + 1
    );
    rangeSeconds = totalDays * 24 * 60 * 60;
  }

  // セグメント数を計算（進捗報告用）
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  const totalSegments = timer0Count * vcountCount * keyCodes.length;

  // イテレータパラメータ
  const RESULT_LIMIT = 32;
  const CHUNK_SECONDS = 3600;
  const PROGRESS_INTERVAL_MS = 500;

  let resultsCount = 0;
  let processedSegments = 0;
  let lastProgressTime = startTime;

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
        if (state.stopRequested || resultsCount >= params.maxResults) {
          return { resultsCount, processedSegments, totalSegments };
        }

        // ParentsIVsJs 構築
        const parentsIVs = new wasmAny.ParentsIVsJs();
        parentsIVs.male = params.parents.male;
        parentsIVs.female = params.parents.female;

        // GenerationConditionsJs 構築
        const conditions = new wasmAny.GenerationConditionsJs();
        conditions.has_nidoran_flag = params.conditions.hasNidoranFlag;
        conditions.set_everstone(
          buildEverstone(wasmAny, params.conditions.everstone)
        );
        conditions.uses_ditto = params.conditions.usesDitto;
        const HIDDEN_ABILITY_SLOT = 2;
        const isHiddenAbilityParent =
          params.conditions.femaleParentAbility === HIDDEN_ABILITY_SLOT;
        conditions.allow_hidden_ability = isHiddenAbilityParent;
        conditions.female_parent_has_hidden = isHiddenAbilityParent;
        conditions.reroll_count = params.conditions.masudaMethod ? 3 : 0;
        conditions.set_trainer_ids(
          new wasmAny.TrainerIds(params.conditions.tid, params.conditions.sid)
        );
        conditions.set_gender_ratio(
          new wasmAny.GenderRatio(
            params.conditions.genderRatio.threshold,
            params.conditions.genderRatio.genderless
          )
        );

        // IndividualFilterJs 構築
        const filter = buildFilter(
          wasmAny,
          params.filterDisabled ? null : params.filter
        );

        // 構造化パラメータを構築
        const dsConfig = buildDSConfig(wasmAny, params, nazo);
        const segmentParams = buildSegmentParams(wasmAny, timer0, vcount, keyCode);
        const timeRangeParams = buildTimeRangeParams(wasmAny, params.timeRange);
        const searchRangeParams = buildSearchRangeParams(wasmAny, searchStartDate, rangeSeconds);

        // イテレータを作成（単一セグメント: 固定 timer0/vcount/keyCode）
        const iterator: {
          isFinished: boolean;
          next_batch: (limit: number, chunk: number) => unknown[];
          free?: () => void;
        } = new wasmAny.EggBootTimingSearchIterator(
          dsConfig,
          segmentParams,
          timeRangeParams,
          searchRangeParams,
          conditions,
          parentsIVs,
          filter,
          params.considerNpcConsumption,
          eggGameModeToWasm(params.gameMode),
          BigInt(params.userOffset),
          params.advanceCount
        );

        try {
          // イテレータループ（時刻方向）
          while (
            !iterator.isFinished &&
            resultsCount < params.maxResults
          ) {
            if (state.stopRequested) break;

            const batchResults = iterator.next_batch(RESULT_LIMIT, CHUNK_SECONDS);

            // 結果をストリーミング送信
            if (batchResults.length > 0) {
              const convertedResults: EggBootTimingSearchResult[] = [];
              for (
                let i = 0;
                i < batchResults.length && resultsCount < params.maxResults;
                i++
              ) {
                const wasmResult = batchResults[i] as WasmEggBootTimingSearchResult;
                convertedResults.push(convertWasmResult(wasmResult, params.macAddress));
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
              const progressPercent = (processedSegments / totalSegments) * 100;
              const estimatedRemainingMs = processedSegments > 0
                ? (elapsedMs / processedSegments) * (totalSegments - processedSegments)
                : 0;

              const progress: EggBootTimingProgress = {
                processedCombinations: processedSegments,
                totalCombinations: totalSegments,
                foundCount: resultsCount,
                progressPercent,
                elapsedMs,
                estimatedRemainingMs,
              };
              post({ type: 'PROGRESS', payload: progress });
              lastProgressTime = now;

            }
          }
        } finally {
          iterator.free?.();
          conditions.free?.();
          parentsIVs.free?.();
        }

        processedSegments++;
      }
    }
  }

  // 最終進捗報告
  const elapsedMs = performance.now() - startTime;
  const progress: EggBootTimingProgress = {
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
}

ctx.onclose = () => {
  cleanupState();
};

export {};
