// Generation Worker (simplified)
// 目的: WASM 列挙と最小限の結果通知のみを担当

import {
  type GenerationWorkerRequest,
  type GenerationWorkerResponse,
  type GenerationParams,
  type GenerationResult,
  type GenerationResultsPayload,
  type GenerationCompletion,
  validateGenerationParams,
  deriveDomainGameMode,
} from '@/types/generation';
import { parseFromWasmRaw } from '@/lib/generation/raw-parser';
import { resolvePokemon } from '@/lib/generation/pokemon-resolver';
import {
  initWasm,
  getWasm,
  isWasmReady,
  type BWGenerationConfigCtor,
  type SeedEnumeratorCtor,
  type SeedEnumeratorInstance,
} from '@/lib/core/wasm-interface';
import { domainGameModeToWasm } from '@/lib/core/mapping/game-mode';
import { domainEncounterTypeToWasm } from '@/lib/core/mapping/encounter-type';
import type { DomainEncounterType } from '@/types/domain';
import type {
  ResolutionContext,
  SerializedResolutionContext,
  ResolvedPokemonData,
} from '@/types/pokemon-resolved';

type BWGenerationConfigInstance = InstanceType<BWGenerationConfigCtor>;

let BWGenerationConfig: BWGenerationConfigCtor | undefined;
let SeedEnumerator: SeedEnumeratorCtor | undefined;

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
  enumerator: SeedEnumeratorInstance | null;
  config: BWGenerationConfigInstance | null;
  resolutionContext: ResolutionContext;
  running: boolean;
  stopRequested: boolean;
}

const state: InternalState = {
  params: null,
  enumerator: null,
  config: null,
  resolutionContext: {},
  running: false,
  stopRequested: false,
};

const ctx = self as typeof self & { onclose?: () => void };
const post = (message: GenerationWorkerResponse) => ctx.postMessage(message);

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<GenerationWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START_GENERATION':
          await handleStart(msg.params, msg.resolutionContext);
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

async function handleStart(
  params: GenerationParams,
  serializedContext?: SerializedResolutionContext,
) {
  if (state.running) return;
  const errors = validateGenerationParams(params);
  if (errors.length) {
    post({ type: 'ERROR', message: errors.join(', '), category: 'VALIDATION', fatal: false });
    return;
  }
  const totalAdvances = params.maxAdvances - Number(params.offset);
  if (totalAdvances <= 0) {
    post({ type: 'ERROR', message: 'maxAdvances must be greater than offset', category: 'VALIDATION', fatal: false });
    return;
  }

  state.params = params;
  state.resolutionContext = hydrateResolutionContext(serializedContext);
  state.stopRequested = false;
  state.running = true;

  try {
    const enumerator = await prepareEnumerator(params, totalAdvances);
    state.enumerator = enumerator;
    const runOutcome = executeEnumeration(params, totalAdvances);
    postResults(runOutcome.results, runOutcome.resolved);
    post({ type: 'COMPLETE', payload: runOutcome.completion });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'ERROR', message, category: 'WASM_INIT', fatal: true });
  } finally {
    cleanupState();
  }
}

async function prepareEnumerator(params: GenerationParams, totalAdvances: number): Promise<SeedEnumeratorInstance> {
  if (!isWasmReady()) await initWasm();
  const wasm = getWasm();
  BWGenerationConfig = wasm.BWGenerationConfig;
  SeedEnumerator = wasm.SeedEnumerator;

  const ConfigCtor = BWGenerationConfig;
  const EnumeratorCtor = SeedEnumerator;
  if (!ConfigCtor) throw new Error('BWGenerationConfig not exposed');
  if (!EnumeratorCtor) throw new Error('SeedEnumerator not exposed');

  const wasmEncounterType = domainEncounterTypeToWasm(params.encounterType as DomainEncounterType);
  state.config = new ConfigCtor(
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

  return new EnumeratorCtor(
    params.baseSeed,
    params.offset,
    totalAdvances,
    state.config,
    wasmMode,
  );
}

function executeEnumeration(params: GenerationParams, totalAdvances: number) {
  const offsetFallbackBase = Number(params.offset);
  const results: GenerationResult[] = [];
  const resolved: ResolvedPokemonData[] = [];
  let processedAdvances = 0;
  let shinyFound = false;
  let reason: GenerationCompletion['reason'] | null = null;
  let encounteredError = false;

  const startTime = performance.now();

  if (!state.enumerator) {
    encounteredError = true;
  } else {
    for (let i = 0; i < totalAdvances; i++) {
      if (state.stopRequested) {
        reason = 'stopped';
        break;
      }
      const raw = state.enumerator.next_pokemon();
      if (!raw) {
        reason = 'max-advances';
        break;
      }
      let unresolved;
      try {
        unresolved = parseFromWasmRaw(raw);
      } catch {
        encounteredError = true;
        reason = 'error';
        break;
      }
      const advanceVal = readAdvanceOrFallback(raw, offsetFallbackBase + i);
      const result: GenerationResult = {
        ...unresolved,
        advance: advanceVal,
        report_needle_direction: readReportNeedleDirection(raw),
      };
      const isShiny = (result.shiny_type ?? 0) !== 0;
      if (isShiny) shinyFound = true;

      processedAdvances += 1;

      if (results.length < params.maxResults) {
        results.push(result);
        resolved.push(resolvePokemon(result, state.resolutionContext));
      }

      if (params.stopAtFirstShiny && isShiny) {
        reason = 'first-shiny';
        break;
      }
      if (params.stopOnCap && results.length >= params.maxResults) {
        reason = 'max-results';
        break;
      }
    }
  }

  if (!reason) {
    if (state.stopRequested) reason = 'stopped';
    else if (encounteredError) reason = 'error';
    else reason = 'max-advances';
  }

  const completion: GenerationCompletion = {
    reason,
    processedAdvances,
    resultsCount: results.length,
    elapsedMs: performance.now() - startTime,
    shinyFound,
  };

  return { results, resolved, completion };
}

function postResults(results: GenerationResult[], resolved: ResolvedPokemonData[]) {
  const payload: GenerationResultsPayload = {
    results,
    resolved: resolved.length ? resolved : undefined,
  };
  post({ type: 'RESULTS', payload });
}

function readAdvanceOrFallback(raw: unknown, fallback: number): number {
  if (!raw || typeof raw !== 'object') return fallback;
  const getter = (raw as Record<string, unknown>).get_advance;
  if (typeof getter === 'function') {
    try {
      const value = getter.call(raw);
      if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
      }
      if (typeof value === 'bigint') {
        const asNumber = Number(value);
        if (Number.isFinite(asNumber)) {
          return asNumber;
        }
      }
    } catch {
      return fallback;
    }
  }
  return fallback;
}

function readReportNeedleDirection(raw: unknown): number | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const getter = (raw as Record<string, unknown>).get_report_needle_direction;
  if (typeof getter === 'function') {
    try {
      const value = getter.call(raw);
      if (typeof value === 'number' && Number.isFinite(value)) return value;
      if (typeof value === 'bigint') return Number(value);
    } catch {
      return undefined;
    }
  }
  // wasm-bindgen exposes readonly accessors as properties; accept number/bigint directly
  if (typeof getter === 'number' && Number.isFinite(getter)) return getter;
  if (typeof getter === 'bigint') return Number(getter);

  const direct = (raw as Record<string, unknown>).report_needle_direction;
  if (typeof direct === 'number' && Number.isFinite(direct)) return direct;
  if (typeof direct === 'bigint') return Number(direct);
  return undefined;
}

function hydrateResolutionContext(serialized?: SerializedResolutionContext): ResolutionContext {
  if (!serialized) return {};
  const ctx: ResolutionContext = {};
  if (serialized.encounterTable) {
    ctx.encounterTable = serialized.encounterTable;
  }
  if (serialized.genderRatios) {
    ctx.genderRatios = new Map(serialized.genderRatios);
  }
  if (serialized.abilityCatalog) {
    ctx.abilityCatalog = new Map(serialized.abilityCatalog);
  }
  return ctx;
}

function cleanupState() {
  state.running = false;
  state.enumerator = null;
  state.config = null;
  state.params = null;
  state.stopRequested = false;
}

ctx.onclose = () => {
  cleanupState();
};

export {};
