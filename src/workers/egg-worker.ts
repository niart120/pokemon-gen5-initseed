// Egg Worker - タマゴ個体列挙専用Worker
import {
  type EggWorkerRequest,
  type EggWorkerResponse,
  type EggGenerationParams,
  type EnumeratedEggData,
  type EggResultsPayload,
  type EggCompletion,
  type IvSet,
  type HiddenPowerInfo,
  EggGameMode,
  validateEggParams,
} from '@/types/egg';
import {
  initWasm,
  getWasm,
  isWasmReady,
} from '@/lib/core/wasm-interface';

interface InternalState {
  params: EggGenerationParams | null;
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
 * EggGameMode は簡易的な4値、GameMode は詳細な8値
 */
function eggGameModeToWasm(mode: EggGameMode): number {
  // WASM GameMode enum values:
  // BwNewGameWithSave = 0, BwNewGameNoSave = 1, BwContinue = 2
  // Bw2NewGameWithMemoryLinkSave = 3, Bw2NewGameNoMemoryLinkSave = 4, Bw2NewGameNoSave = 5
  // Bw2ContinueWithMemoryLink = 6, Bw2ContinueNoMemoryLink = 7
  switch (mode) {
    case EggGameMode.BwNew:
      return 0; // BwNewGameWithSave (デフォルトでセーブありを想定)
    case EggGameMode.BwContinue:
      return 2; // BwContinue
    case EggGameMode.Bw2New:
      return 5; // Bw2NewGameNoSave (デフォルト)
    case EggGameMode.Bw2Continue:
      return 7; // Bw2ContinueNoMemoryLink (デフォルト)
    default:
      return 2; // BwContinue as fallback
  }
}

const ctx = self as typeof self & { onclose?: () => void };
const post = (message: EggWorkerResponse) => ctx.postMessage(message);

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<EggWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START_GENERATION':
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

async function handleStart(params: EggGenerationParams) {
  if (state.running) {
    return;
  }

  const errors = validateEggParams(params);
  if (errors.length) {
    post({ type: 'ERROR', message: errors.join(', '), category: 'VALIDATION', fatal: false });
    return;
  }

  state.params = params;
  state.stopRequested = false;
  state.running = true;

  try {
    await ensureWasm();
    const runOutcome = executeEnumeration(params);
    postResults(runOutcome.results);
    post({ type: 'COMPLETE', payload: runOutcome.completion });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildEverstone(wasm: any, plan: EggGenerationParams['conditions']['everstone']) {
  if (plan.type === 'none') {
    return wasm.EverstonePlanJs.None;
  } else {
    return wasm.EverstonePlanJs.fixed(plan.nature);
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildFilter(wasm: any, filter: EggGenerationParams['filter']) {
  const wasmFilter = new wasm.IndividualFilterJs();

  // フィルタがnullの場合はデフォルト（全pass-through）のフィルタを返す
  if (!filter) return wasmFilter;

  // IV範囲設定
  for (let i = 0; i < 6; i++) {
    const range = filter.ivRanges[i];
    wasmFilter.set_iv_range(i, range.min, range.max);
  }

  // Optional条件設定
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
    case 'male': return 0;
    case 'female': return 1;
    case 'genderless': return 2;
    default: return 0;
  }
}

function wasmGenderToDomain(wasmGender: number): 'male' | 'female' | 'genderless' {
  switch (wasmGender) {
    case 0: return 'male';
    case 1: return 'female';
    case 2: return 'genderless';
    default: return 'male';
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseHiddenPower(raw: any): HiddenPowerInfo {
  if (!raw || raw.type === 'unknown' || raw === 'Unknown') {
    return { type: 'unknown' };
  }
  return {
    type: 'known',
    hpType: raw.hp_type ?? raw.hpType ?? 0,
    power: raw.power ?? 0,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseEnumeratedEggData(raw: any): EnumeratedEggData | null {
  if (!raw) return null;

  try {
    const advance = Number(raw.advance);
    const egg = raw.egg;
    const lcgSeedHex = egg.lcg_seed_hex ?? egg.lcgSeedHex ?? '0x0';

    const resolvedEgg = {
      lcgSeedHex,
      ivs: [
        egg.ivs[0], egg.ivs[1], egg.ivs[2],
        egg.ivs[3], egg.ivs[4], egg.ivs[5]
      ] as IvSet,
      nature: typeof egg.nature === 'number' ? egg.nature : (egg.nature as number),
      gender: wasmGenderToDomain(typeof egg.gender === 'number' ? egg.gender : 0),
      ability: (typeof egg.ability === 'number' ? egg.ability : 0) as 0 | 1 | 2,
      shiny: (typeof egg.shiny === 'number' ? egg.shiny : 0) as 0 | 1 | 2,
      pid: egg.pid,
      hiddenPower: parseHiddenPower(egg.hidden_power ?? egg.hiddenPower),
    };

    return {
      advance,
      egg: resolvedEgg,
      isStable: raw.is_stable ?? raw.isStable ?? false,
    };
  } catch {
    return null;
  }
}

function executeEnumeration(params: EggGenerationParams) {
  const wasm = getWasm();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasmAny = wasm as any;

  if (!wasmAny.EggSeedEnumeratorJs) {
    throw new Error('EggSeedEnumeratorJs not exposed in WASM');
  }

  const results: EnumeratedEggData[] = [];
  let processedCount = 0;
  let filteredCount = 0;
  let reason: EggCompletion['reason'] = 'max-count';

  const startTime = performance.now();

  // ParentsIVsJs 構築
  const parentsIVs = new wasmAny.ParentsIVsJs();
  parentsIVs.male = params.parents.male;
  parentsIVs.female = params.parents.female;

  // GenerationConditionsJs 構築
  const conditions = new wasmAny.GenerationConditionsJs();
  conditions.has_nidoran_flag = params.conditions.hasNidoranFlag;
  conditions.set_everstone(buildEverstone(wasmAny, params.conditions.everstone));
  conditions.uses_ditto = params.conditions.usesDitto;
  // femaleParentAbility === 2 (隠れ特性) の場合のみ夢特性判定を有効化
  const isHiddenAbilityParent = params.conditions.femaleParentAbility === 2;
  conditions.allow_hidden_ability = isHiddenAbilityParent;
  conditions.female_parent_has_hidden = isHiddenAbilityParent;
  // 国際孵化: masudaMethod が true の場合は reroll_count = 3
  conditions.reroll_count = params.conditions.masudaMethod ? 3 : 0;
  conditions.set_trainer_ids(new wasmAny.TrainerIds(params.conditions.tid, params.conditions.sid));
  conditions.set_gender_ratio(new wasmAny.GenderRatio(
    params.conditions.genderRatio.threshold,
    params.conditions.genderRatio.genderless
  ));

  // IndividualFilterJs 構築
  // filterDisabled の場合は全pass-throughフィルタ、それ以外は指定フィルタ
  const filter = buildFilter(wasmAny, params.filterDisabled ? null : params.filter);

  // EggSeedEnumeratorJs 作成
  const enumerator = new wasmAny.EggSeedEnumeratorJs(
    params.baseSeed,
    params.userOffset,
    params.count,
    conditions,
    parentsIVs,
    filter,
    params.considerNpcConsumption,
    eggGameModeToWasm(params.gameMode)
  );

  // 列挙ループ
  try {
    while (true) {
      if (state.stopRequested) {
        reason = 'stopped';
        break;
      }

      const rawData = enumerator.next_egg();
      if (!rawData) {
        reason = 'max-count';
        break;
      }

      processedCount++;

      // WASM から EnumeratedEggData を取得
      const eggData = parseEnumeratedEggData(rawData);
      if (eggData) {
        results.push(eggData);
        filteredCount++;
      }
    }
  } catch (e) {
    reason = 'error';
    throw e;
  } finally {
    enumerator.free();
    conditions.free?.();
    parentsIVs.free?.();
    if (filter) filter.free?.();
  }

  const completion: EggCompletion = {
    reason,
    processedCount,
    filteredCount,
    elapsedMs: performance.now() - startTime,
  };

  return { results, completion };
}

function postResults(results: EnumeratedEggData[]) {
  const payload: EggResultsPayload = { results };
  post({ type: 'RESULTS', payload });
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
