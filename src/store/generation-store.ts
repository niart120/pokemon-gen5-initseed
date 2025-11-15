import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationProgress, GenerationCompletion, GenerationResultBatch, GenerationResult, GenerationParamsHex } from '@/types/generation';
import { validateGenerationParams, hexParamsToGenerationParams, generationParamsToHex, requiresStaticSelection } from '@/types/generation';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import { isLocationBasedEncounter, listEncounterLocations, listEncounterSpeciesOptions } from '@/data/encounters/helpers';
import type { DomainEncounterType } from '@/types/domain';
import { resolveBatch, toUiReadyPokemon, type ResolutionContext, type ResolvedPokemonData, type UiReadyPokemonData } from '@/lib/generation/pokemon-resolver';

export type GenerationStatus = 'idle' | 'starting' | 'running' | 'paused' | 'stopping' | 'completed' | 'error';

export type ShinyFilterMode = 'all' | 'shiny' | 'non-shiny';

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
  progress: GenerationProgress | null;
  results: GenerationResult[]; // GenerationResult 型 (UnresolvedPokemonData + advance)
  lastCompletion: GenerationCompletion | null;
  error: string | null;
  filters: GenerationFilters;
  metrics: { startTime?: number; lastUpdateTime?: number; shinyCount?: number };
  internalFlags: { receivedAnyBatch: boolean };
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
  setDraftParams: (partial: Partial<GenerationParamsHex>) => void;
  setEncounterField: (field: string | undefined) => void;
  setEncounterSpeciesId: (speciesId: number | undefined) => void;
  setStaticEncounterId: (staticId: string | null | undefined) => void;
  validateDraft: () => void;
  commitParams: () => boolean;
  startGeneration: () => Promise<boolean>;
  pauseGeneration: () => void;
  resumeGeneration: () => void;
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
  _onWorkerProgress: (p: GenerationProgress) => void;
  _onWorkerBatch: (b: GenerationResultBatch) => void;
  _onWorkerComplete: (c: GenerationCompletion) => void;
  _onWorkerStopped: (reason: string) => void;
  _onWorkerError: (err: string) => void;
}

export type GenerationSlice = GenerationSliceState & GenerationSliceActions;

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
  version: 'B',
  encounterType: 0,
  tid: 1,
  sid: 2,
  syncEnabled: false,
  syncNatureId: 0,
  stopAtFirstShiny: false,
  stopOnCap: true,
  batchSize: 10000,
  abilityMode: 'none',
  shinyCharm: false,
  isShinyLocked: false,
  memoryLink: false,
  newGame: false,
  withSave: true,
};

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

export const createGenerationSlice = (set: SetFn, get: GetFn<GenerationSlice>): GenerationSlice => ({
  params: null,
  draftParams: {
    ...DEFAULT_GENERATION_DRAFT_PARAMS,
  },
  // 動的Encounter UI用追加状態（WASMパラメータ未連動のため GenerationParamsHex 外）
  encounterField: undefined,
  encounterSpeciesId: undefined,
  staticEncounterId: null,
  validationErrors: [],
  status: 'idle',
  progress: null,
  results: [],
  lastCompletion: null,
  error: null,
  filters: createDefaultGenerationFilters(),
  metrics: {},
  internalFlags: { receivedAnyBatch: false },
  encounterTable: undefined,
  genderRatios: undefined,
  abilityCatalog: undefined,

  setDraftParams: (partial) => {
    set((state: GenerationSlice) => {
      const prevDraft = state.draftParams;
      const nextDraft = { ...prevDraft, ...partial } as GenerationParamsHex;
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
    const combinedErrors = needsStaticSelection && !staticEncounterId
      ? [...baseErrors, 'static encounter selection required']
      : baseErrors;
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
    const combinedErrors = needsStaticSelection && !staticEncounterId
      ? [...baseErrors, 'static encounter selection required']
      : baseErrors;
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
    if (status === 'running' || status === 'paused' || status === 'starting') return false;
    if (!get().commitParams()) return false;
    const { params } = get();
    if (!params) return false;
    set({ status: 'starting', progress: null, results: [], lastCompletion: null, error: null, metrics: { startTime: performance.now() } });
    try {
      await manager.start(params);
      set({ status: 'running' });
      return true;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ status: 'error', error: message || 'start-failed' });
      return false;
    }
  },
  pauseGeneration: () => {
    if (get().status !== 'running') return;
    manager.pause();
    set({ status: 'paused' });
  },
  resumeGeneration: () => {
    if (get().status !== 'paused') return;
    manager.resume();
    set({ status: 'running' });
  },
  stopGeneration: () => {
    const st = get().status;
    if (st === 'running' || st === 'paused') {
      set({ status: 'stopping' });
      manager.stop();
    }
  },
  clearResults: () => set({ results: [] }),
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
    progress: null,
    results: [],
    lastCompletion: null,
    error: null,
    metrics: {},
    internalFlags: { receivedAnyBatch: false },
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

  _onWorkerProgress: (p) => {
    set({ progress: p, metrics: { ...get().metrics, lastUpdateTime: performance.now() } });
  },
  _onWorkerBatch: (b) => {
    set((state: GenerationSlice) => {
      if (state.results.length >= (state.params?.maxResults || Infinity)) return state; // 変更なし
      const capacityLeft = (state.params?.maxResults || Infinity) - state.results.length;
      const slice = b.results.slice(0, capacityLeft);
      let shinyAdd = 0;
      for (let i = 0; i < slice.length; i++) if (slice[i].shiny_type !== 0) shinyAdd++;
      const shinyCount = (state.metrics.shinyCount || 0) + shinyAdd;
      return {
        ...state,
        results: state.results.concat(slice),
        internalFlags: { receivedAnyBatch: true },
        metrics: { ...state.metrics, shinyCount },
      };
    });
  },
  _onWorkerComplete: (c) => {
    set({ status: 'completed', lastCompletion: c });
  },
  _onWorkerStopped: (_reason) => {
    set({ status: 'idle' });
  },
  _onWorkerError: (err) => {
    set({ status: 'error', error: err });
  },
});

// マネージャーのイベントを slice にバインド（store 作成後に呼ばれる想定）
export const bindGenerationManager = (get: () => GenerationSlice) => {
  manager.onProgress(p => get()._onWorkerProgress(p));
  manager.onResultBatch(b => get()._onWorkerBatch(b));
  manager.onComplete(c => get()._onWorkerComplete(c));
  manager.onStopped(r => get()._onWorkerStopped(r.reason));
  manager.onError(e => get()._onWorkerError(e));
};

export const getGenerationManager = () => manager;

// --- Selectors (B1) ---
export const selectThroughputEma = (s: GenerationSlice): number | null => {
  const t = s.progress?.throughputEma ?? s.progress?.throughputRaw ?? s.progress?.throughput;
  return typeof t === 'number' && isFinite(t) && t > 0 ? t : null;
};

export const selectEtaFormatted = (s: GenerationSlice): string | null => {
  const p = s.progress;
  if (!p) return null;
  const ema = selectThroughputEma(s);
  if (!ema) return null;
  const remaining = (p.totalAdvances - p.processedAdvances);
  if (!(remaining > 0)) return '00:00';
  const sec = remaining / ema;
  if (!isFinite(sec) || sec <= 0) return null;
  const hrs = Math.floor(sec / 3600);
  const mins = Math.floor((sec % 3600) / 60);
  const secs = Math.floor(sec % 60);
  if (hrs > 0) return `${hrs}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
  return `${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
};

export const selectShinyCount = (s: GenerationSlice): number => s.metrics.shinyCount || 0;

function canBuildFullHex(d: Partial<GenerationParamsHex>): d is GenerationParamsHex {
  const required: (keyof GenerationParamsHex)[] = ['baseSeedHex','offsetHex','maxAdvances','maxResults','version','encounterType','tid','sid','syncEnabled','syncNatureId','shinyCharm','isShinyLocked','stopAtFirstShiny','stopOnCap','batchSize','memoryLink','newGame','withSave'];
  return required.every(k => (d as Record<string, unknown>)[k] !== undefined);
}

export function getCurrentHexParams(state: GenerationSlice): GenerationParamsHex | null {
  return state.params ? generationParamsToHex(state.params) : null;
}

export interface FilteredGenerationDisplayRow {
  raw: GenerationResult;
  resolved?: ResolvedPokemonData;
  ui?: UiReadyPokemonData;
}

type FilteredRowsCache = {
  resultsRef: GenerationResult[];
  filtersRef: GenerationFilters;
  encounterTableRef?: EncounterTable;
  genderRatiosRef?: Map<number, GenderRatio>;
  abilityCatalogRef?: Map<number, string[]>;
  locale: 'ja' | 'en';
  rows: FilteredGenerationDisplayRow[];
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
  if (
    cache &&
    cache.resultsRef === results &&
    cache.filtersRef === filters &&
    cache.encounterTableRef === encounterTable &&
    cache.genderRatiosRef === genderRatios &&
    cache.abilityCatalogRef === abilityCatalog &&
    cache.locale === locale
  ) {
    return cache;
  }

  const resolved = selectResolvedResults(s);
  const uiReady = selectUiReadyResults(s, locale);

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

  const rows: FilteredGenerationDisplayRow[] = [];

  for (let i = 0; i < results.length; i++) {
    const raw = results[i];
    const resolvedData = resolved[i];
    const uiData = uiReady[i];

    if (shinyMode === 'shiny' && raw.shiny_type === 0) continue;
    if (shinyMode === 'non-shiny' && raw.shiny_type !== 0) continue;
    if (natureSet && !natureSet.has(raw.nature)) continue;

    if (speciesSet) {
      const speciesId = resolvedData?.speciesId;
      if (!speciesId || !speciesSet.has(speciesId)) continue;
    }

    if (abilitySet) {
      const abilityIndex = resolvedData?.abilityIndex;
      if (abilityIndex == null || !abilitySet.has(abilityIndex)) continue;
    }

    if (genderSet) {
      const gender = resolvedData?.gender;
      if (!gender || !genderSet.has(gender)) continue;
    }

    if (hasLevelFilter) {
      const level = resolvedData?.level;
      if (level == null) continue;
      if (levelRange?.min != null && level < levelRange.min) continue;
      if (levelRange?.max != null && level > levelRange.max) continue;
    }

    if (hasStatFilters) {
      const stats = uiData?.stats;
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

    rows.push({ raw, resolved: resolvedData, ui: uiData });
  }

  const field = filters.sortField ?? 'advance';
  const order = filters.sortOrder === 'desc' ? -1 : 1;

  rows.sort((a, b) => {
    const ar = a.raw;
    const br = b.raw;
    const aResolved = a.resolved;
    const bResolved = b.resolved;
    let av: number;
    let bv: number;
    switch (field) {
      case 'pid':
        av = ar.pid >>> 0;
        bv = br.pid >>> 0;
        break;
      case 'nature':
        av = ar.nature;
        bv = br.nature;
        break;
      case 'shiny':
        av = ar.shiny_type;
        bv = br.shiny_type;
        break;
      case 'species':
        av = aResolved?.speciesId ?? Number.MAX_SAFE_INTEGER;
        bv = bResolved?.speciesId ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'ability':
        av = aResolved?.abilityIndex ?? Number.MAX_SAFE_INTEGER;
        bv = bResolved?.abilityIndex ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'level':
        av = aResolved?.level ?? Number.MAX_SAFE_INTEGER;
        bv = bResolved?.level ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'advance':
      default:
        av = ar.advance;
        bv = br.advance;
        break;
    }
    if (av < bv) return -1 * order;
    if (av > bv) return 1 * order;
    return 0;
  });

  const raw = rows.map((entry) => entry.raw);
  const nextCache: NonNullable<FilteredRowsCache> = {
    resultsRef: results,
    filtersRef: filters,
    encounterTableRef: encounterTable,
    genderRatiosRef: genderRatios,
    abilityCatalogRef: abilityCatalog,
    locale,
    rows,
    raw,
  };
  _filteredRowsCache = nextCache;
  return nextCache;
}

export const selectFilteredDisplayRows = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja'): FilteredGenerationDisplayRow[] => {
  return computeFilteredRowsCache(s, locale).rows;
};

export const selectFilteredSortedResults = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja') => {
  return computeFilteredRowsCache(s, locale).raw;
};

// ===== Resolution / UI adapters =====

let _resolvedCache: {
  resultsRef: GenerationResult[];
  encounterTableRef?: EncounterTable;
  genderRatiosRef?: Map<number, GenderRatio>;
  abilityCatalogRef?: Map<number, string[]>;
  output: ResolvedPokemonData[];
} | null = null;

export const selectResolvedResults = (s: GenerationSlice): ResolvedPokemonData[] => {
  const { results, encounterTable, genderRatios, abilityCatalog } = s as GenerationSlice & { encounterTable?: EncounterTable; genderRatios?: Map<number, GenderRatio>; abilityCatalog?: Map<number, string[]> };
  const cache = _resolvedCache;
  if (cache && cache.resultsRef === results && cache.encounterTableRef === encounterTable && cache.genderRatiosRef === genderRatios && cache.abilityCatalogRef === abilityCatalog) {
    return cache.output;
  }
  if (!results.length) {
    _resolvedCache = { resultsRef: results, encounterTableRef: encounterTable, genderRatiosRef: genderRatios, abilityCatalogRef: abilityCatalog, output: [] };
    return _resolvedCache.output;
  }
  const ctx: ResolutionContext = { encounterTable, genderRatios, abilityCatalog };
  const resolved = resolveBatch(results, ctx);
  _resolvedCache = { resultsRef: results, encounterTableRef: encounterTable, genderRatiosRef: genderRatios, abilityCatalogRef: abilityCatalog, output: resolved };
  return resolved;
};

let _uiReadyCache: {
  resolvedRef: ResolvedPokemonData[];
  locale: string;
  version: 'B' | 'W' | 'B2' | 'W2';
  baseSeed: bigint | undefined;
  output: UiReadyPokemonData[];
} | null = null;

export const selectUiReadyResults = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja'): UiReadyPokemonData[] => {
  const resolved = selectResolvedResults(s);
  const version = (s.params?.version ?? s.draftParams.version ?? 'B') as 'B' | 'W' | 'B2' | 'W2';
  const baseSeed = s.params?.baseSeed;
  if (
    _uiReadyCache &&
    _uiReadyCache.resolvedRef === resolved &&
    _uiReadyCache.locale === locale &&
    _uiReadyCache.version === version &&
    _uiReadyCache.baseSeed === baseSeed
  ) {
    return _uiReadyCache.output;
  }
  const out = resolved.map(r => toUiReadyPokemon(r, { locale, version, baseSeed }));
  _uiReadyCache = { resolvedRef: resolved, locale, version, baseSeed, output: out };
  return out;
};
