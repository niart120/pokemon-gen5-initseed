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
import { initWasm, getWasm, isWasmReady } from '@/lib/core/wasm-interface';
import { keyCodeToNames } from '@/lib/utils/key-input';
import romParameters from '@/data/rom-parameters';
import type { Hardware } from '@/types/rom';

// Hardware別のframe値
const HARDWARE_FRAME_VALUES: Record<Hardware, number> = {
  DS: 8,
  DS_LITE: 6,
  '3DS': 9
};

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

    // 日単位でチャンク分割して検索
    const { dateRange } = params;
    const startDate = new Date(
      dateRange.startYear,
      dateRange.startMonth - 1,
      dateRange.startDay
    );
    const endDate = new Date(
      dateRange.endYear,
      dateRange.endMonth - 1,
      dateRange.endDay
    );

    // 日数計算
    const totalDays = Math.max(
      1,
      Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1
    );

    // 1バッチあたりの日数（進捗報告間隔を調整）
    const DAYS_PER_BATCH = 1;

    let totalResultsCount = 0; // 結果はストリーミングのみ、配列保持しない（OOM対策）
    let batchIndex = 0;

    for (let dayOffset = 0; dayOffset < totalDays; dayOffset += DAYS_PER_BATCH) {
      if (state.stopRequested) break;

      // バッチの開始・終了日を計算
      const batchStartDate = new Date(startDate.getTime() + dayOffset * 24 * 60 * 60 * 1000);
      const batchDays = Math.min(DAYS_PER_BATCH, totalDays - dayOffset);

      // このバッチを検索
      const batchResults = executeSearchBatch(params, batchStartDate, batchDays);

      if (batchResults.length > 0) {
        totalResultsCount += batchResults.length;

        // 結果送信（ストリーミング）
        post({
          type: 'RESULTS',
          payload: { results: batchResults, batchIndex },
        });
      }

      // 進捗報告
      const processedDays = Math.min(dayOffset + DAYS_PER_BATCH, totalDays);
      const elapsedMs = performance.now() - startTime;
      const progressPercent = (processedDays / totalDays) * 100;
      const estimatedRemainingMs = processedDays > 0
        ? (elapsedMs / processedDays) * (totalDays - processedDays)
        : 0;

      const progress: EggBootTimingProgress = {
        processedCombinations: processedDays,
        totalCombinations: totalDays,
        foundCount: totalResultsCount,
        progressPercent,
        elapsedMs,
        estimatedRemainingMs,
      };
      post({ type: 'PROGRESS', payload: progress });

      batchIndex++;

      // maxResultsに達したら終了
      if (totalResultsCount >= params.maxResults) break;
    }

    // 完了通知
    const completion: EggBootTimingCompletion = {
      reason: state.stopRequested ? 'stopped' : 'completed',
      processedCombinations: totalDays,
      totalCombinations: totalDays,
      resultsCount: totalResultsCount,
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
  if (filter.shiny !== undefined) {
    wasmFilter.set_shiny(filter.shiny);
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
 * 指定された開始日から指定日数分の検索を実行
 */
function executeSearchBatch(
  params: EggBootTimingSearchParams,
  batchStartDate: Date,
  batchDays: number
): EggBootTimingSearchResult[] {
  console.log(`[EggBootTimingWorker] executeSearchBatch: date=${batchStartDate.toISOString()}, days=${batchDays}`);

  const wasm = getWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasmAny = wasm as any;

  if (!wasmAny.EggBootTimingSearcher) {
    throw new Error('EggBootTimingSearcher not exposed in WASM');
  }

  // nazo値を解決
  const nazo = resolveNazoValue(params.romVersion, params.romRegion);
  console.log(`[EggBootTimingWorker] nazo resolved: [${nazo.join(', ')}]`);

  // ParentsIVsJs 構築
  const parentsIVs = new wasmAny.ParentsIVsJs();
  parentsIVs.male = params.parents.male;
  parentsIVs.female = params.parents.female;

  // GenerationConditionsJs 構築
  const conditions = new wasmAny.GenerationConditionsJs();
  conditions.has_nidoran_flag = params.conditions.hasNidoranFlag;
  conditions.set_everstone(buildEverstone(wasmAny, params.conditions.everstone));
  conditions.uses_ditto = params.conditions.usesDitto;
  // femaleParentAbility: 0=Ability1, 1=Ability2, 2=HiddenAbility
  const HIDDEN_ABILITY_SLOT = 2;
  const isHiddenAbilityParent = params.conditions.femaleParentAbility === HIDDEN_ABILITY_SLOT;
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
  // filterDisabled の場合は全pass-throughフィルタ、それ以外は指定フィルタ
  const filter = buildFilter(wasmAny, params.filterDisabled ? null : params.filter);

  // Searcher構築 - frameはhardwareから導出
  const frameValue = HARDWARE_FRAME_VALUES[params.hardware];
  const searcher = new wasmAny.EggBootTimingSearcher(
    new Uint8Array(params.macAddress),
    new Uint32Array(nazo),
    params.hardware,
    params.keyInputMask,
    frameValue,
    params.timeRange.hour.start,
    params.timeRange.hour.end,
    params.timeRange.minute.start,
    params.timeRange.minute.end,
    params.timeRange.second.start,
    params.timeRange.second.end,
    conditions,
    parentsIVs,
    filter,
    params.considerNpcConsumption,
    eggGameModeToWasm(params.gameMode),
    BigInt(params.userOffset),
    params.advanceCount
  );

  const results: EggBootTimingSearchResult[] = [];

  try {
    // 検索実行 - バッチの開始日から指定日数分
    const rangeSeconds = batchDays * 24 * 60 * 60;

    console.log(`[EggBootTimingWorker] Calling search_eggs_integrated_simd: year=${batchStartDate.getFullYear()}, month=${batchStartDate.getMonth() + 1}, day=${batchStartDate.getDate()}, rangeSeconds=${rangeSeconds}`);

    const wasmResults = searcher.search_eggs_integrated_simd(
      batchStartDate.getFullYear(),
      batchStartDate.getMonth() + 1,
      batchStartDate.getDate(),
      0,  // 開始時刻は0時から
      0,
      0,
      rangeSeconds,
      params.timer0Range.min,
      params.timer0Range.max,
      params.vcountRange.min,
      params.vcountRange.max
    );

    console.log(`[EggBootTimingWorker] search_eggs_integrated_simd returned ${wasmResults.length} results`);

    // 結果変換
    for (let i = 0; i < wasmResults.length && i < params.maxResults; i++) {
      if (state.stopRequested) break;
      const wasmResult = wasmResults[i] as WasmEggBootTimingSearchResult;
      results.push(convertWasmResult(wasmResult, params.macAddress));
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`[EggBootTimingWorker] executeSearchBatch error: ${message}`);
    throw e;
  } finally {
    // Note: filter は EggBootTimingSearcher コンストラクタで所有権が移動するため、
    // ここで free() を呼ぶ必要がない（呼ぶと nullpo になる）
    searcher.free?.();
    conditions.free?.();
    parentsIVs.free?.();
  }

  return results;
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
