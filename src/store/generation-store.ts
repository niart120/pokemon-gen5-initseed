import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationCompletion, GenerationResultsPayload, GenerationResult, GenerationParamsHex, BootTimingDraft } from '@/types/generation';
import { validateGenerationParams, hexParamsToGenerationParams, generationParamsToHex, requiresStaticSelection } from '@/types/generation';
import { createDefaultDeviceProfile } from '@/types/profile';
import { keyMaskToNames } from '@/lib/utils/key-input';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import { isLocationBasedEncounter, listEncounterLocations, listEncounterSpeciesOptions } from '@/data/encounters/helpers';
import { type DomainEncounterType } from '@/types/domain';
import { resolveBatch } from '@/lib/generation/pokemon-resolver';
import type { ResolvedPokemonData, SerializedResolutionContext } from '@/types/pokemon-resolved';
import { buildResolutionContextFromSources } from '@/lib/initialization/build-resolution-context';
import { BOOT_TIMING_PAIR_LIMIT, deriveBootTimingSeedJobs, type DerivedSeedMetadata } from '@/lib/generation/boot-timing-derivation';
import {
  cloneBootTimingDraft,
  createBootTimingDraftFromProfile,
  normalizeBootTimingDraft,
} from '@/store/utils/boot-timing-draft';
import {
  advanceDerivedSeedState,
  createDerivedSeedState,
  markDerivedSeedAbort,
  shouldAppendDerivedResults,
  type DerivedSeedRunState,
} from '@/store/modules/boot-timing-runner';

export type GenerationStatus = 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'error';

export type ShinyFilterMode = 'all' | 'shiny' | 'star' | 'square' | 'non-shiny';

export interface StatRange {
  min?: number;
  max?: number;
}

export type StatRangeFilters = Partial<Record<'hp' | 'attack' | 'defense' | 'specialAttack' | 'specialDefense' | 'speed', StatRange>>;

export interface GenerationFilters {
  sortField: 'advance' | 'pid' | 'nature' | 'shiny' | 'species' | 'ability' | 'level';
  sortOrder: 'asc' | 'desc';
  shinyMode: ShinyFilterMode;
  speciesIds: number[];
  natureIds: number[];
  abilityIndices: (0 | 1 | 2)[];
  genders: ('M' | 'F' | 'N')[];
  levelRange: StatRange | undefined;
  statRanges: StatRangeFilters;
}

export interface GenerationSliceState {
  params: GenerationParams | null;
  draftParams: Partial<GenerationParamsHex>;
  validationErrors: string[];
  status: GenerationStatus;
  results: GenerationResult[];
  resolvedResults: ResolvedPokemonData[];
  lastCompletion: GenerationCompletion | null;
  error: string | null;
  filters: GenerationFilters;
  internalFlags: { receivedResults: boolean };
  derivedSeedState: DerivedSeedRunState | null;
  activeSeedMetadata: DerivedSeedMetadata | null;
  // 解決用参照データ (任意設定)
  encounterTable?: EncounterTable;
  genderRatios?: Map<number, GenderRatio>;
  abilityCatalog?: Map<number, string[]>;
  // 動的Encounter UI 追加状態
  encounterField?: string; // 正規化 location key
  encounterSpeciesId?: number; // 単一選択 speciesId
  staticEncounterId?: string | null; // 選択した静的エンカウントエントリID
}

export interface GenerationSliceActions {
  setDraftParams: (partial: DraftParamsUpdate) => void;
  setEncounterField: (field: string | undefined) => void;
  setEncounterSpeciesId: (speciesId: number | undefined) => void;
  setStaticEncounterId: (staticId: string | null | undefined) => void;
  validateDraft: () => void;
  commitParams: () => boolean;
  startGeneration: () => Promise<boolean>;
  stopGeneration: () => void;
  clearResults: () => void;
  applyFilters: (partial: Partial<GenerationFilters>) => void;
  resetGenerationState: () => void;
  // 参照データ setter
  setEncounterTable: (table: EncounterTable | undefined) => void;
  setGenderRatios: (ratios: Map<number, GenderRatio> | undefined) => void;
  setAbilityCatalog: (catalog: Map<number, string[]> | undefined) => void;
  resetGenerationFilters: () => void;
  // 内部コールバック（manager から）
  _onWorkerResults: (payload: GenerationResultsPayload) => void;
  _onWorkerComplete: (c: GenerationCompletion) => void;
  _onWorkerError: (err: string) => void;
}

export type GenerationSlice = GenerationSliceState & GenerationSliceActions;

type DraftParamsUpdate = Partial<Omit<GenerationParamsHex, 'bootTiming'>> & {
  bootTiming?: Partial<BootTimingDraft>;
};

const DEFAULT_DEVICE_PROFILE_FOR_GENERATION = createDefaultDeviceProfile();

function validateBootTimingInputs(draft: BootTimingDraft): string[] {
  const errors: string[] = [];
  if (!draft.timestampIso) {
    errors.push('boot-timing timestamp required');
  } else {
    const time = Date.parse(draft.timestampIso);
    if (Number.isNaN(time)) {
      errors.push('boot-timing timestamp invalid');
    }
  }

  const timer0Min = draft.timer0Range.min;
  const timer0Max = draft.timer0Range.max;
  if (timer0Min < 0 || timer0Min > 0xFFFF || timer0Max < 0 || timer0Max > 0xFFFF) {
    errors.push('timer0 range out of bounds');
  } else if (timer0Min > timer0Max) {
    errors.push('timer0 range invalid');
  }

  const vcountMin = draft.vcountRange.min;
  const vcountMax = draft.vcountRange.max;
  if (vcountMin < 0 || vcountMin > 0xFF || vcountMax < 0 || vcountMax > 0xFF) {
    errors.push('vcount range out of bounds');
  } else if (vcountMin > vcountMax) {
    errors.push('vcount range invalid');
  }

  const timer0Span = timer0Max - timer0Min + 1;
  const vcountSpan = vcountMax - vcountMin + 1;
  const pairCount = timer0Span > 0 && vcountSpan > 0 ? timer0Span * vcountSpan : 0;
  if (pairCount <= 0) {
    errors.push('timer0/vcount range produces no combinations');
  } else if (pairCount > BOOT_TIMING_PAIR_LIMIT) {
    errors.push(`timer0/vcount combinations exceed limit (${pairCount} > ${BOOT_TIMING_PAIR_LIMIT})`);
  }

  return errors;
}


function resolveShinyLock(base: GenerationParams, staticEncounterId: string | null | undefined): GenerationParams {
  if (!staticEncounterId || !requiresStaticSelection(base.encounterType)) {
    return { ...base, isShinyLocked: false };
  }
  const encounterType = base.encounterType as DomainEncounterType;
  const candidates = listEncounterSpeciesOptions(base.version, encounterType);
  const match = candidates.find(opt => opt.kind === 'static' && opt.id === staticEncounterId);
  if (match && match.kind === 'static') {
    return { ...base, isShinyLocked: Boolean(match.isShinyLocked) };
  }
  return { ...base, isShinyLocked: false };
}

export const DEFAULT_GENERATION_DRAFT_PARAMS: GenerationParamsHex = {
  baseSeedHex: '1',
  offsetHex: '0',
  maxAdvances: 50,
  maxResults: 15000,
  version: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.romVersion,
  encounterType: 0,
  tid: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.tid,
  sid: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.sid,
  syncEnabled: false,
  syncNatureId: 0,
  stopAtFirstShiny: false,
  stopOnCap: true,
  abilityMode: 'none',
  shinyCharm: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.shinyCharm,
  isShinyLocked: false,
  memoryLink: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.memoryLink,
  newGame: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.newGame,
  withSave: DEFAULT_DEVICE_PROFILE_FOR_GENERATION.withSave,
  seedSourceMode: 'lcg',
  bootTiming: createBootTimingDraftFromProfile(DEFAULT_DEVICE_PROFILE_FOR_GENERATION),
};

export function createDefaultGenerationDraftParams(): GenerationParamsHex {
  return {
    ...DEFAULT_GENERATION_DRAFT_PARAMS,
    bootTiming: cloneBootTimingDraft(DEFAULT_GENERATION_DRAFT_PARAMS.bootTiming),
  };
}

// 単一インスタンスマネージャ（UI からは slice 経由で操作）
const manager = new GenerationWorkerManager();

// Zustand set/get 最小シグネチャ (型安全対象: GenerationSlice の部分更新)
type PartialState<T> = Partial<T> | ((state: T) => Partial<T>);
type SetFn = (partial: PartialState<GenerationSlice>, replace?: boolean) => void;
type GetFn<T> = () => T;

export function createDefaultGenerationFilters(): GenerationFilters {
  return {
    sortField: 'advance',
    sortOrder: 'asc',
    shinyMode: 'all',
    speciesIds: [],
    natureIds: [],
    abilityIndices: [],
    genders: [],
    levelRange: undefined,
    statRanges: {},
  };
}

export const createGenerationSlice = (set: SetFn, get: GetFn<GenerationSlice>): GenerationSlice => {
  const startWorkerRun = async (
    params: GenerationParams,
    options: { resetResults: boolean; metadata?: DerivedSeedMetadata | null },
  ): Promise<boolean> => {
    if (options.resetResults) {
      set({
        results: [],
        resolvedResults: [],
        lastCompletion: null,
        error: null,
        internalFlags: { receivedResults: false },
      });
    } else {
      set({
        lastCompletion: null,
        error: null,
        internalFlags: { receivedResults: false },
      });
    }

    set({
      params,
      status: 'starting',
      activeSeedMetadata: options.metadata ?? null,
    });

    try {
      const resolutionContext = serializeResolutionContextForWorker(get());
      await manager.start(params, { resolutionContext });
      set({ status: 'running' });
      return true;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ status: 'error', error: message || 'start-failed', activeSeedMetadata: null });
      return false;
    }
  };

  return {
    params: null,
    draftParams: createDefaultGenerationDraftParams(),
  // 動的Encounter UI用追加状態（WASMパラメータ未連動のため GenerationParamsHex 外）
  encounterField: undefined,
  encounterSpeciesId: undefined,
  staticEncounterId: null,
  validationErrors: [],
  status: 'idle',
  results: [],
  resolvedResults: [],
  lastCompletion: null,
  error: null,
  filters: createDefaultGenerationFilters(),
  internalFlags: { receivedResults: false },
  derivedSeedState: null,
  activeSeedMetadata: null,
  encounterTable: undefined,
  genderRatios: undefined,
  abilityCatalog: undefined,

  setDraftParams: (partial) => {
    set((state: GenerationSlice) => {
      const prevDraft = (state.draftParams ?? createDefaultGenerationDraftParams()) as GenerationParamsHex;
      const { bootTiming: partialBootTiming, ...rest } = partial;
      const nextDraft = { ...prevDraft, ...rest } as GenerationParamsHex;
      const baseBootTiming = prevDraft.bootTiming ?? DEFAULT_GENERATION_DRAFT_PARAMS.bootTiming;
      nextDraft.bootTiming = partialBootTiming !== undefined
        ? normalizeBootTimingDraft(partialBootTiming, baseBootTiming)
        : cloneBootTimingDraft(baseBootTiming);
      const version = nextDraft.version ?? 'B';
      let encounterField = state.encounterField;
      let encounterSpeciesId = state.encounterSpeciesId;
      let staticEncounterId = state.staticEncounterId ?? null;

      const encounterTypeChanged = partial.encounterType !== undefined && partial.encounterType !== prevDraft.encounterType;
      const versionChanged = partial.version !== undefined && partial.version !== prevDraft.version;

      if (encounterTypeChanged || versionChanged) {
        // 新しいエンカウントタイプ・バージョンに合わせて UI 選択肢を整理
        const encounterTypeValue = nextDraft.encounterType;
        if (encounterTypeValue === undefined) {
          encounterField = undefined;
          encounterSpeciesId = undefined;
          staticEncounterId = null;
          nextDraft.isShinyLocked = false;
        } else {
          const domainEncounterType = encounterTypeValue as DomainEncounterType;
          if (isLocationBasedEncounter(domainEncounterType)) {
            const locations = listEncounterLocations(version, domainEncounterType);
            const preferred = locations.find(loc => loc.key === encounterField) ?? locations[0];
            encounterField = preferred?.key;
            encounterSpeciesId = undefined;
            staticEncounterId = null;
            nextDraft.isShinyLocked = false;
          } else {
            const speciesOptions = listEncounterSpeciesOptions(version, domainEncounterType);
            const staticOptions = speciesOptions.filter(opt => opt.kind === 'static');
            const selected = staticOptions.find(opt => opt.id === staticEncounterId)
              ?? staticOptions.find(opt => opt.speciesId === encounterSpeciesId)
              ?? staticOptions[0];
            if (selected) {
              staticEncounterId = selected.id;
              encounterSpeciesId = selected.speciesId;
              nextDraft.isShinyLocked = Boolean(selected.isShinyLocked);
            } else {
              staticEncounterId = null;
              encounterSpeciesId = undefined;
              nextDraft.isShinyLocked = false;
            }
            encounterField = undefined;
          }
        }
      }

      // Memory Link 制約: BW と withSave=false は update 内で解決済みだが、ここでも安全側で補正
      if (!nextDraft.withSave) {
        nextDraft.memoryLink = false;
      }
      const resolvedVersion = nextDraft.version ?? 'B';
      if (resolvedVersion === 'B' || resolvedVersion === 'W') {
        nextDraft.memoryLink = false;
      }

      return {
        draftParams: nextDraft,
        encounterField,
        encounterSpeciesId,
        staticEncounterId,
      } as Partial<GenerationSlice>;
    });
  },
  setEncounterField: (field) => set((state: GenerationSlice) => ({
    encounterField: field,
    encounterSpeciesId: undefined,
    staticEncounterId: null,
    draftParams: { ...state.draftParams, isShinyLocked: false },
  })),
  setEncounterSpeciesId: (speciesId) => set({ encounterSpeciesId: speciesId }),
  setStaticEncounterId: (staticId) => set((state: GenerationSlice) => {
    const nextId = staticId ?? null;
    const nextDraft = { ...state.draftParams };
    if (!nextId) {
      nextDraft.isShinyLocked = false;
    } else {
      const version = state.draftParams.version;
      const encounterType = state.draftParams.encounterType;
      if (typeof version === 'string' && encounterType !== undefined && requiresStaticSelection(encounterType)) {
        const options = listEncounterSpeciesOptions(version, encounterType as DomainEncounterType);
        const match = options.find(opt => opt.kind === 'static' && opt.id === nextId);
        nextDraft.isShinyLocked = Boolean(match && match.kind === 'static' && match.isShinyLocked);
      } else {
        nextDraft.isShinyLocked = false;
      }
    }
    return { staticEncounterId: nextId, draftParams: nextDraft } as Partial<GenerationSlice>;
  }),
  validateDraft: () => {
    const { draftParams, staticEncounterId } = get();
    // hex → bigint へ一時変換
    const maybe: GenerationParams | null = canBuildFullHex(draftParams) ? hexParamsToGenerationParams(draftParams as GenerationParamsHex) : null;
    const baseErrors = maybe ? validateGenerationParams(maybe) : ['incomplete params'];
    const encounterType = maybe?.encounterType ?? (typeof draftParams.encounterType === 'number' ? draftParams.encounterType : undefined);
    const needsStaticSelection = typeof encounterType === 'number' && requiresStaticSelection(encounterType);
    let combinedErrors = needsStaticSelection && !staticEncounterId
      ? [...baseErrors, 'static encounter selection required']
      : baseErrors;
    const seedSourceMode = draftParams.seedSourceMode ?? 'lcg';
    if (seedSourceMode === 'boot-timing') {
      const bootTiming = draftParams.bootTiming ?? DEFAULT_GENERATION_DRAFT_PARAMS.bootTiming;
      combinedErrors = [...combinedErrors, ...validateBootTimingInputs(bootTiming)];
    }
    set({ validationErrors: combinedErrors });
  },
  commitParams: () => {
    const { draftParams, staticEncounterId } = get();
    if (!canBuildFullHex(draftParams)) {
      set({ validationErrors: ['incomplete params'] });
      return false;
    }
    const full = hexParamsToGenerationParams(draftParams as GenerationParamsHex);
    const paramsWithLock = resolveShinyLock(full, staticEncounterId);
    const baseErrors = validateGenerationParams(paramsWithLock);
    const needsStaticSelection = requiresStaticSelection(paramsWithLock.encounterType);
    let combinedErrors = needsStaticSelection && !staticEncounterId
      ? [...baseErrors, 'static encounter selection required']
      : baseErrors;
    const seedSourceMode = draftParams.seedSourceMode ?? 'lcg';
    if (seedSourceMode === 'boot-timing') {
      const bootTiming = draftParams.bootTiming ?? DEFAULT_GENERATION_DRAFT_PARAMS.bootTiming;
      combinedErrors = [...combinedErrors, ...validateBootTimingInputs(bootTiming)];
    }
    set({ validationErrors: combinedErrors });
    if (combinedErrors.length) return false;
    set({ params: paramsWithLock });
    return true;
  },
  startGeneration: async () => {
    if (typeof Worker === 'undefined') {
      set({ error: 'worker-not-supported' });
      return false;
    }
    const { status } = get();
    if (status === 'running' || status === 'starting' || status === 'stopping') return false;
    if (!get().commitParams()) return false;
    const { params, draftParams } = get();
    if (!params) return false;
    const seedSourceMode = (draftParams.seedSourceMode ?? 'lcg');
    if (seedSourceMode === 'boot-timing') {
      if (!canBuildFullHex(draftParams)) {
        set({ validationErrors: ['incomplete params'] });
        return false;
      }
      const derivation = deriveBootTimingSeedJobs(draftParams as GenerationParamsHex);
      if (!derivation.ok) {
        set({ validationErrors: [derivation.error] });
        return false;
      }
      if (!derivation.jobs.length) {
        set({ validationErrors: ['no derived seeds produced'] });
        return false;
      }
      set({ derivedSeedState: createDerivedSeedState(derivation.jobs) });
      const started = await startWorkerRun(derivation.jobs[0].params, { resetResults: true, metadata: derivation.jobs[0].metadata });
      if (!started) {
        set({ derivedSeedState: null });
      }
      return started;
    }

    set({ derivedSeedState: null, activeSeedMetadata: null });
    return startWorkerRun(params, { resetResults: true, metadata: null });
  },
  stopGeneration: () => {
    const st = get().status;
    if (st === 'running' || st === 'stopping') {
      set((state: GenerationSlice) => ({
        status: 'stopping',
        derivedSeedState: markDerivedSeedAbort(state.derivedSeedState),
      }));
      manager.stop();
    }
  },
  clearResults: () => set({ results: [], resolvedResults: [] }),
  applyFilters: (partial) => set((state: GenerationSlice) => {
    const current = state.filters;
    const nextSortField = partial.sortField ?? current.sortField;
    const nextSortOrder = partial.sortOrder ?? current.sortOrder;
    const nextShinyMode = partial.shinyMode ?? current.shinyMode;
    const nextSpecies = partial.speciesIds ? [...partial.speciesIds] : current.speciesIds;
    const nextNature = partial.natureIds ? [...partial.natureIds] : current.natureIds;
    const nextAbility = partial.abilityIndices ? [...partial.abilityIndices] : current.abilityIndices;
    const nextGenders = partial.genders ? [...partial.genders] as ('M' | 'F' | 'N')[] : current.genders;

    let nextLevelRange = current.levelRange;
    if (Object.prototype.hasOwnProperty.call(partial, 'levelRange')) {
      const range = partial.levelRange;
      if (!range) {
        nextLevelRange = undefined;
      } else {
        const hasMin = range.min != null;
        const hasMax = range.max != null;
        nextLevelRange = hasMin || hasMax
          ? {
              min: hasMin ? range.min : undefined,
              max: hasMax ? range.max : undefined,
            }
          : undefined;
      }
    }

    let nextStatRanges = current.statRanges;
    if (partial.statRanges) {
      nextStatRanges = {};
      const entries = Object.entries(partial.statRanges) as Array<[keyof StatRangeFilters, StatRange | undefined]>;
      for (const [key, range] of entries) {
        if (!range) continue;
        const hasMin = range.min != null;
        const hasMax = range.max != null;
        if (!hasMin && !hasMax) continue;
        nextStatRanges[key] = {
          min: hasMin ? range.min : undefined,
          max: hasMax ? range.max : undefined,
        };
      }
    }

    const next: GenerationFilters = {
      sortField: nextSortField,
      sortOrder: nextSortOrder,
      shinyMode: nextShinyMode,
      speciesIds: nextSpecies,
      natureIds: nextNature,
      abilityIndices: nextAbility,
      genders: nextGenders,
      levelRange: nextLevelRange,
      statRanges: nextStatRanges,
    };

    return { filters: next } as Partial<GenerationSlice>;
  }),
  // 追加: リセット
  resetGenerationFilters: () => set({ filters: createDefaultGenerationFilters() }),
  resetGenerationState: () => set({
    status: 'idle',
    results: [],
    resolvedResults: [],
    lastCompletion: null,
    error: null,
    internalFlags: { receivedResults: false },
    derivedSeedState: null,
    activeSeedMetadata: null,
    encounterTable: undefined,
    genderRatios: undefined,
    abilityCatalog: undefined,
    encounterField: undefined,
    encounterSpeciesId: undefined,
    staticEncounterId: null,
  }),
  setEncounterTable: (table) => set({ encounterTable: table }),
  setGenderRatios: (ratios) => set({ genderRatios: ratios }),
  setAbilityCatalog: (catalog) => set({ abilityCatalog: catalog }),

  _onWorkerResults: (payload) => {
    const state = get();
    const resolvedList = Array.isArray(payload.resolved) && payload.resolved.length === payload.results.length
      ? payload.resolved
      : resolveBatch(payload.results, buildResolutionContextFromSources({
        encounterTable: state.encounterTable,
        genderRatios: state.genderRatios,
        abilityCatalog: state.abilityCatalog,
      }));

    const metadata = state.activeSeedMetadata;
    const keyInputNames = metadata ? keyMaskToNames(metadata.keyMask) : undefined;
    const activeBaseSeed = state.params?.baseSeed;
    const enrichedResults = payload.results.map((result) => {
      const baseEnriched = {
        ...result,
        baseSeed: activeBaseSeed ?? result.baseSeed,
      };
      if (!metadata) {
        return baseEnriched;
      }
      return {
        ...baseEnriched,
        seedSourceMode: metadata.seedSourceMode,
        derivedSeedIndex: metadata.derivedSeedIndex,
        seedSourceSeedHex: metadata.seedSourceSeedHex,
        timer0: metadata.timer0,
        vcount: metadata.vcount,
        bootTimestampIso: metadata.bootTimestampIso,
        keyInputNames,
        macAddress: metadata.macAddress,
      };
    });

    const shouldAppend = shouldAppendDerivedResults(state.derivedSeedState);
    const nextResults = shouldAppend ? [...state.results, ...enrichedResults] : enrichedResults;
    const nextResolved = shouldAppend ? [...(state.resolvedResults ?? []), ...resolvedList] : resolvedList;

    set({
      results: nextResults,
      resolvedResults: nextResolved,
      internalFlags: { receivedResults: true },
    });
  },
  _onWorkerComplete: (c) => {
    const state = get();
    const derivedState = state.derivedSeedState;
    if (derivedState) {
      const advanceResult = advanceDerivedSeedState(derivedState, c);
      set({
        derivedSeedState: advanceResult.nextState,
        activeSeedMetadata: null,
      });

      const shouldStartNext = Boolean(
        advanceResult.nextJob &&
        !derivedState.abortRequested &&
        c.reason !== 'error',
      );
      if (shouldStartNext && advanceResult.nextJob) {
        const nextJob = advanceResult.nextJob;
        void (async () => {
          const started = await startWorkerRun(nextJob.params, { resetResults: false, metadata: nextJob.metadata });
          if (!started) {
            set({ derivedSeedState: null, status: 'error', activeSeedMetadata: null });
          }
        })();
        return;
      }

      const finalCompletion: GenerationCompletion = advanceResult.finalCompletion ?? {
        ...c,
        processedAdvances: advanceResult.aggregate.processedAdvances,
        resultsCount: advanceResult.aggregate.resultsCount,
        elapsedMs: advanceResult.aggregate.elapsedMs,
        shinyFound: advanceResult.aggregate.shinyFound,
      };
      const finalStatus: GenerationStatus = c.reason === 'stopped' ? 'idle' : 'completed';
      set({ status: finalStatus, lastCompletion: finalCompletion });
      return;
    }

    const nextStatus: GenerationStatus = c.reason === 'stopped' ? 'idle' : 'completed';
    set({ status: nextStatus, lastCompletion: c });
  },
  _onWorkerError: (err) => {
    set({ status: 'error', error: err, derivedSeedState: null, activeSeedMetadata: null });
  },
  } as GenerationSlice;
};

// マネージャーのイベントを slice にバインド（store 作成後に呼ばれる想定）
export const bindGenerationManager = (get: () => GenerationSlice) => {
  manager.onResults(payload => get()._onWorkerResults(payload));
  manager.onComplete(c => get()._onWorkerComplete(c));
  manager.onError(e => get()._onWorkerError(e));
};

export const getGenerationManager = () => manager;

function serializeResolutionContextForWorker(state: GenerationSlice): SerializedResolutionContext | undefined {
  const { encounterTable, genderRatios, abilityCatalog } = state;
  const hasTable = Boolean(encounterTable);
  const hasRatios = Boolean(genderRatios && genderRatios.size > 0);
  const hasAbilities = Boolean(abilityCatalog && abilityCatalog.size > 0);
  if (!hasTable && !hasRatios && !hasAbilities) {
    return undefined;
  }
  return {
    encounterTable: encounterTable ?? undefined,
    genderRatios: hasRatios ? Array.from((genderRatios ?? new Map()).entries()) : undefined,
    abilityCatalog: hasAbilities ? Array.from((abilityCatalog ?? new Map()).entries()) : undefined,
  };
}

function canBuildFullHex(d: Partial<GenerationParamsHex>): d is GenerationParamsHex {
  const required: (keyof GenerationParamsHex)[] = ['baseSeedHex','offsetHex','maxAdvances','maxResults','version','encounterType','tid','sid','syncEnabled','syncNatureId','shinyCharm','isShinyLocked','stopAtFirstShiny','stopOnCap','memoryLink','newGame','withSave','seedSourceMode','bootTiming'];
  return required.every(k => (d as Record<string, unknown>)[k] !== undefined);
}

export function getCurrentHexParams(state: GenerationSlice): GenerationParamsHex | null {
  return state.params ? generationParamsToHex(state.params) : null;
}

export {
  selectFilteredDisplayRows,
  selectFilteredSortedResults,
  selectResolvedResults,
} from '@/store/selectors/generation-results';

