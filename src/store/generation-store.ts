import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationCompletion, GenerationResultsPayload, GenerationResult, GenerationParamsHex, BootTimingDraft } from '@/types/generation';
import { validateGenerationParams, hexParamsToGenerationParams, generationParamsToHex, requiresStaticSelection } from '@/types/generation';
import type { DeviceProfile } from '@/types/profile';
import { createDefaultDeviceProfile } from '@/types/profile';
import { KEY_INPUT_DEFAULT, keyMaskToNames, normalizeKeyMask } from '@/lib/utils/key-input';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import { isLocationBasedEncounter, listEncounterLocations, listEncounterSpeciesOptions } from '@/data/encounters/helpers';
import { DomainShinyType, type DomainEncounterType } from '@/types/domain';
import { resolveBatch, toUiReadyPokemon } from '@/lib/generation/pokemon-resolver';
import type { ResolvedPokemonData, UiReadyPokemonData, SerializedResolutionContext } from '@/types/pokemon-resolved';
import { buildResolutionContextFromSources } from '@/lib/initialization/build-resolution-context';
import { BOOT_TIMING_PAIR_LIMIT, deriveBootTimingSeedJobs, type DerivedSeedJob, type DerivedSeedMetadata } from '@/lib/generation/boot-timing-derivation';
import { formatKeyInputDisplay } from '@/lib/i18n/strings/search-results';

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

type BootTimingProfileSource = Pick<DeviceProfile, 'romRegion' | 'hardware' | 'timer0Range' | 'vcountRange' | 'macAddress'>;

type DraftParamsUpdate = Partial<Omit<GenerationParamsHex, 'bootTiming'>> & {
  bootTiming?: Partial<BootTimingDraft>;
};

const DEFAULT_DEVICE_PROFILE_FOR_GENERATION = createDefaultDeviceProfile();

function ensureMacAddressTuple(
  values: readonly number[] | undefined,
  fallback: BootTimingDraft['macAddress'] = [0, 0, 0, 0, 0, 0],
): BootTimingDraft['macAddress'] {
  const result: number[] = [];
  for (let i = 0; i < 6; i += 1) {
    const fallbackValue = fallback[i] ?? 0;
    const raw = values?.[i];
    result.push(clampNumber(raw, 0, 255, fallbackValue));
  }
  return result as BootTimingDraft['macAddress'];
}

export function createBootTimingDraftFromProfile(profile: BootTimingProfileSource): BootTimingDraft {
  return {
    timestampIso: undefined,
    keyMask: KEY_INPUT_DEFAULT,
    timer0Range: { ...profile.timer0Range },
    vcountRange: { ...profile.vcountRange },
    romRegion: profile.romRegion,
    hardware: profile.hardware,
    macAddress: ensureMacAddressTuple(profile.macAddress),
  };
}

export function cloneBootTimingDraft(source: BootTimingDraft): BootTimingDraft {
  return {
    timestampIso: source.timestampIso,
    keyMask: source.keyMask,
    timer0Range: { ...source.timer0Range },
    vcountRange: { ...source.vcountRange },
    romRegion: source.romRegion,
    hardware: source.hardware,
    macAddress: ensureMacAddressTuple(source.macAddress),
  };
}

export function normalizeBootTimingDraft(
  partial: Partial<BootTimingDraft> | undefined,
  fallback: BootTimingDraft,
): BootTimingDraft {
  const base = fallback ?? createBootTimingDraftFromProfile(DEFAULT_DEVICE_PROFILE_FOR_GENERATION);
  const timer0Range = normalizeNumericRange(partial?.timer0Range, base.timer0Range, 0, 0xFFFF);
  const vcountRange = normalizeNumericRange(partial?.vcountRange, base.vcountRange, 0, 0xFF);
  const keyMask = partial?.keyMask != null ? normalizeKeyMask(partial.keyMask) : base.keyMask;
  return {
    timestampIso: partial?.timestampIso ?? base.timestampIso,
    keyMask,
    timer0Range,
    vcountRange,
    romRegion: partial?.romRegion ?? base.romRegion,
    hardware: partial?.hardware ?? base.hardware,
    macAddress: ensureMacAddressTuple(partial?.macAddress, base.macAddress),
  };
}

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

function normalizeNumericRange(
  partial: { min?: number; max?: number } | undefined,
  fallback: { min: number; max: number },
  minBound: number,
  maxBound: number,
): { min: number; max: number } {
  const nextMin = clampNumber(partial?.min, minBound, maxBound, fallback.min);
  const nextMax = clampNumber(partial?.max, minBound, maxBound, fallback.max);
  if (nextMin > nextMax) {
    return { min: nextMax, max: nextMin };
  }
  return { min: nextMin, max: nextMax };
}

function clampNumber(value: number | undefined, min: number, max: number, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return clampToBounds(fallback, min, max);
  }
  return clampToBounds(Math.round(value), min, max);
}

function clampToBounds(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

interface DerivedSeedAggregate {
  processedAdvances: number;
  resultsCount: number;
  elapsedMs: number;
  shinyFound: boolean;
}

interface DerivedSeedRunState {
  jobs: DerivedSeedJob[];
  cursor: number;
  total: number;
  aggregate: DerivedSeedAggregate;
  abortRequested: boolean;
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

  const createDerivedSeedState = (jobs: DerivedSeedJob[]): DerivedSeedRunState => ({
    jobs,
    cursor: 0,
    total: jobs.length,
    aggregate: {
      processedAdvances: 0,
      resultsCount: 0,
      elapsedMs: 0,
      shinyFound: false,
    },
    abortRequested: false,
  });

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
        derivedSeedState: state.derivedSeedState
          ? { ...state.derivedSeedState, abortRequested: true }
          : null,
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

    const shouldAppend = Boolean(state.derivedSeedState && state.derivedSeedState.cursor > 0);
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
      const aggregate: DerivedSeedAggregate = {
        processedAdvances: derivedState.aggregate.processedAdvances + c.processedAdvances,
        resultsCount: derivedState.aggregate.resultsCount + c.resultsCount,
        elapsedMs: derivedState.aggregate.elapsedMs + c.elapsedMs,
        shinyFound: derivedState.aggregate.shinyFound || c.shinyFound,
      };
      const nextCursor = derivedState.cursor + 1;
      const hasMore = nextCursor < derivedState.total;
      set({
        derivedSeedState: hasMore ? { ...derivedState, cursor: nextCursor, aggregate } : null,
        activeSeedMetadata: null,
      });

      if (hasMore && !derivedState.abortRequested && c.reason !== 'error') {
        const nextJob = derivedState.jobs[nextCursor];
        void (async () => {
          const started = await startWorkerRun(nextJob.params, { resetResults: false, metadata: nextJob.metadata });
          if (!started) {
            set({ derivedSeedState: null, status: 'error', activeSeedMetadata: null });
          }
        })();
        return;
      }

      const finalCompletion: GenerationCompletion = {
        ...c,
        processedAdvances: aggregate.processedAdvances,
        resultsCount: aggregate.resultsCount,
        elapsedMs: aggregate.elapsedMs,
        shinyFound: aggregate.shinyFound,
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

type FilteredRowsCache = {
  resultsRef: GenerationResult[];
  resolvedRef: ResolvedPokemonData[];
  filtersRef: GenerationFilters;
  encounterTableRef?: EncounterTable;
  genderRatiosRef?: Map<number, GenderRatio>;
  abilityCatalogRef?: Map<number, string[]>;
  locale: 'ja' | 'en';
  baseSeedRef?: bigint;
  versionRef: 'B' | 'W' | 'B2' | 'W2';
  ui: UiReadyPokemonData[];
  raw: GenerationResult[];
} | null;

let _filteredRowsCache: FilteredRowsCache = null;

function computeFilteredRowsCache(s: GenerationSlice, locale: 'ja' | 'en'): NonNullable<FilteredRowsCache> {
  const {
    results,
    filters,
    encounterTable,
    genderRatios,
    abilityCatalog,
  } = s as GenerationSlice & {
    encounterTable?: EncounterTable;
    genderRatios?: Map<number, GenderRatio>;
    abilityCatalog?: Map<number, string[]>;
    locale?: 'ja' | 'en';
  };

  const cache = _filteredRowsCache;
  const version = (s.params?.version ?? s.draftParams.version ?? 'B') as 'B' | 'W' | 'B2' | 'W2';
  const baseSeed = s.params?.baseSeed;
  const resolvedFromState = s.resolvedResults ?? [];
  const requiresFallback = results.length > 0 && resolvedFromState.length !== results.length;

  if (
    cache &&
    cache.resultsRef === results &&
    cache.filtersRef === filters &&
    cache.encounterTableRef === encounterTable &&
    cache.genderRatiosRef === genderRatios &&
    cache.abilityCatalogRef === abilityCatalog &&
    cache.locale === locale &&
    cache.versionRef === version &&
    cache.baseSeedRef === baseSeed &&
    (
      (!requiresFallback && cache.resolvedRef === resolvedFromState) ||
      (requiresFallback && cache.resolvedRef.length === results.length)
    )
  ) {
    return cache;
  }

  let resolved = resolvedFromState;

  if (requiresFallback) {
    const context = buildResolutionContextFromSources({ encounterTable, genderRatios, abilityCatalog }) ?? {};
    resolved = resolveBatch(results, context);
  }

  const shinyMode = filters.shinyMode;
  const natureSet = filters.natureIds.length ? new Set(filters.natureIds) : null;
  const speciesSet = filters.speciesIds.length ? new Set(filters.speciesIds) : null;
  const abilitySet = speciesSet && filters.abilityIndices.length ? new Set(filters.abilityIndices) : null;
  const genderSet = speciesSet && filters.genders.length ? new Set(filters.genders) : null;
  const statKeys: Array<keyof StatRangeFilters> = ['hp', 'attack', 'defense', 'specialAttack', 'specialDefense', 'speed'];
  const hasStatFilters = statKeys.some((key) => {
    const range = filters.statRanges[key];
    return !!range && (range.min != null || range.max != null);
  });
  const levelRange = filters.levelRange;
  const hasLevelFilter = Boolean(levelRange && (levelRange.min != null || levelRange.max != null));

  const entries: Array<{ raw: GenerationResult; ui: UiReadyPokemonData }> = [];

  for (let i = 0; i < results.length; i++) {
    const raw = results[i];
    const resolvedData = resolved[i];
    if (!resolvedData) continue;
    const perRowBaseSeed = raw.baseSeed ?? baseSeed;
    const uiData = toUiReadyPokemon(resolvedData, { locale, version, baseSeed: perRowBaseSeed });

    const shinyType = uiData.shinyType ?? DomainShinyType.Normal;
    if (shinyMode === 'shiny' && shinyType === DomainShinyType.Normal) continue;
    if (shinyMode === 'non-shiny' && shinyType !== DomainShinyType.Normal) continue;
    if (shinyMode === 'star' && shinyType !== DomainShinyType.Star) continue;
    if (shinyMode === 'square' && shinyType !== DomainShinyType.Square) continue;
    if (natureSet && !natureSet.has(uiData.natureId)) continue;

    if (speciesSet) {
      const speciesId = uiData.speciesId;
      if (!speciesId || !speciesSet.has(speciesId)) continue;
    }

    if (abilitySet) {
      const abilityIndex = uiData.abilityIndex;
      if (abilityIndex == null || !abilitySet.has(abilityIndex)) continue;
    }

    if (genderSet) {
      const gender = uiData.genderCode;
      if (!gender || !genderSet.has(gender)) continue;
    }

    if (hasLevelFilter) {
      const level = uiData.level;
      if (level == null) continue;
      if (levelRange?.min != null && level < levelRange.min) continue;
      if (levelRange?.max != null && level > levelRange.max) continue;
    }

    if (hasStatFilters) {
      const stats = uiData.stats;
      if (!stats) continue;
      let ok = true;
      for (const key of statKeys) {
        const range = filters.statRanges[key];
        if (!range) continue;
        const value = stats[key as keyof typeof stats];
        if (range.min != null && value < range.min) {
          ok = false;
          break;
        }
        if (range.max != null && value > range.max) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
    }

    if (raw.seedSourceMode) {
      uiData.seedSourceMode = raw.seedSourceMode;
      uiData.derivedSeedIndex = raw.derivedSeedIndex;
      uiData.seedSourceSeedHex = raw.seedSourceSeedHex;
    } else {
      uiData.seedSourceMode = undefined;
      uiData.derivedSeedIndex = undefined;
      uiData.seedSourceSeedHex = undefined;
    }
    uiData.timer0 = raw.timer0;
    uiData.vcount = raw.vcount;
    uiData.bootTimestampIso = raw.bootTimestampIso;
    if (raw.keyInputNames && raw.keyInputNames.length) {
      uiData.keyInputNames = raw.keyInputNames;
      uiData.keyInputDisplay = formatKeyInputDisplay(raw.keyInputNames, locale);
    } else {
      uiData.keyInputNames = undefined;
      uiData.keyInputDisplay = undefined;
    }

    entries.push({ raw, ui: uiData });
  }

  const field = filters.sortField ?? 'advance';
  const order = filters.sortOrder === 'desc' ? -1 : 1;

  entries.sort((a, b) => {
    const au = a.ui;
    const bu = b.ui;
    let av: number;
    let bv: number;
    switch (field) {
      case 'pid':
        av = au.pid >>> 0;
        bv = bu.pid >>> 0;
        break;
      case 'nature':
        av = au.natureId;
        bv = bu.natureId;
        break;
      case 'shiny':
        av = au.shinyType;
        bv = bu.shinyType;
        break;
      case 'species':
        av = au.speciesId ?? Number.MAX_SAFE_INTEGER;
        bv = bu.speciesId ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'ability':
        av = au.abilityIndex ?? Number.MAX_SAFE_INTEGER;
        bv = bu.abilityIndex ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'level':
        av = au.level ?? Number.MAX_SAFE_INTEGER;
        bv = bu.level ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'advance':
      default:
        av = au.advance;
        bv = bu.advance;
        break;
    }
    if (av < bv) return -1 * order;
    if (av > bv) return 1 * order;
    return 0;
  });

  const ui = entries.map(entry => entry.ui);
  const raw = entries.map(entry => entry.raw);
  const nextCache: NonNullable<FilteredRowsCache> = {
    resultsRef: results,
    resolvedRef: resolved,
    filtersRef: filters,
    encounterTableRef: encounterTable,
    genderRatiosRef: genderRatios,
    abilityCatalogRef: abilityCatalog,
    locale,
    versionRef: version,
    baseSeedRef: baseSeed,
    ui,
    raw,
  };
  _filteredRowsCache = nextCache;
  return nextCache;
}

export const selectFilteredDisplayRows = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja'): UiReadyPokemonData[] => {
  return computeFilteredRowsCache(s, locale).ui;
};

export const selectFilteredSortedResults = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja') => {
  return computeFilteredRowsCache(s, locale).raw;
};

export const selectResolvedResults = (s: GenerationSlice): ResolvedPokemonData[] => {
  return s.resolvedResults ?? [];
};
