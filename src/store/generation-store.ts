import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationProgress, GenerationCompletion, GenerationResultBatch, GenerationResult, GenerationParamsHex } from '@/types/generation';
import { validateGenerationParams, hexParamsToGenerationParams, generationParamsToHex } from '@/types/generation';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';

// 単一インスタンスマネージャ（UI からは slice 経由で操作）
const manager = new GenerationWorkerManager();

export type GenerationStatus = 'idle' | 'starting' | 'running' | 'paused' | 'stopping' | 'completed' | 'error';

export interface GenerationFilters {
  shinyOnly: boolean;
  natureIds: number[]; // 追加フィルタ用プレースホルダ
  sortField?: 'advance' | 'pid' | 'nature' | 'shiny' | 'species' | 'ability' | 'level';
  sortOrder?: 'asc' | 'desc';
  advanceRange?: { min?: number; max?: number };
  shinyTypes?: number[]; // 0/1/2 指定。空 or undefined は全許可
  // --- New advanced filters (Phase3/4 UI) ---
  speciesIds?: number[]; // EncounterTable から解決された nationalId (複数選択)
  abilityIndices?: (0 | 1 | 2)[]; // 0:通常1,1:通常2,2:隠れ（speciesIds 選択時のみ有効）
  levelRange?: { min?: number; max?: number };
  genders?: ('M' | 'F' | 'N')[]; // 種族選択時のみ有効（N=性別不明種）
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
}

export interface GenerationSliceActions {
  setDraftParams: (partial: Partial<GenerationParamsHex>) => void;
  setEncounterField: (field: string | undefined) => void;
  setEncounterSpeciesId: (speciesId: number | undefined) => void;
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

// Zustand set/get 最小シグネチャ (型安全対象: GenerationSlice の部分更新)
type PartialState<T> = Partial<T> | ((state: T) => Partial<T>);
type SetFn = (partial: PartialState<GenerationSlice>, replace?: boolean) => void;
type GetFn<T> = () => T;

export const createGenerationSlice = (set: SetFn, get: GetFn<GenerationSlice>): GenerationSlice => ({
  params: null,
  draftParams: {
    baseSeedHex: '1',
    offsetHex: '0',
  // Defaults adjusted (Phase3/4 tuning): expand search window & batch throughput
  maxAdvances: 50000,
  maxResults: 15000,
    version: 'B',
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
  batchSize: 5000, // 固定 (UI 非表示, tuned from 1000 -> 5000 for reduced worker->UI message overhead)
  abilityMode: 'none',
  shinyCharm: false,
  memoryLink: false,
  newGame: true,
  noSave: false,
  },
  // 動的Encounter UI用追加状態（WASMパラメータ未連動のため GenerationParamsHex 外）
  encounterField: undefined,
  encounterSpeciesId: undefined,
  validationErrors: [],
  status: 'idle',
  progress: null,
  results: [],
  lastCompletion: null,
  error: null,
  filters: { shinyOnly: false, natureIds: [], sortField: 'advance', sortOrder: 'asc', advanceRange: undefined, shinyTypes: undefined, speciesIds: undefined, abilityIndices: undefined, levelRange: undefined, genders: undefined },
  metrics: {},
  internalFlags: { receivedAnyBatch: false },
  encounterTable: undefined,
  genderRatios: undefined,
  abilityCatalog: undefined,

  setDraftParams: (partial) => {
    set((state: GenerationSlice) => {
      const nextDraft = { ...state.draftParams, ...partial };
      // encounterType 変更検出で動的フィールドリセット
      if (partial.encounterType !== undefined && partial.encounterType !== state.draftParams.encounterType) {
        return { draftParams: nextDraft, encounterField: undefined, encounterSpeciesId: undefined } as Partial<GenerationSlice>;
      }
      return { draftParams: nextDraft } as Partial<GenerationSlice>;
    });
  },
  setEncounterField: (field) => set({ encounterField: field, encounterSpeciesId: undefined }),
  setEncounterSpeciesId: (speciesId) => set({ encounterSpeciesId: speciesId }),
  validateDraft: () => {
    const { draftParams } = get();
    // hex → bigint へ一時変換
    const maybe: GenerationParams | null = canBuildFullHex(draftParams) ? hexParamsToGenerationParams(draftParams as GenerationParamsHex) : null;
    const errors = maybe ? validateGenerationParams(maybe) : ['incomplete params'];
    set({ validationErrors: errors });
  },
  commitParams: () => {
    const { draftParams } = get();
    if (!canBuildFullHex(draftParams)) {
      set({ validationErrors: ['incomplete params'] });
      return false;
    }
    const full = hexParamsToGenerationParams(draftParams as GenerationParamsHex);
    const errors = validateGenerationParams(full);
    set({ validationErrors: errors });
    if (errors.length) return false;
    set({ params: full });
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
    const params = get().params!;
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
  applyFilters: (partial) => set((state: GenerationSlice) => ({ filters: { ...state.filters, ...partial } })),
  // 追加: リセット
  resetGenerationFilters: () => set({ filters: { shinyOnly: false, natureIds: [], sortField: 'advance', sortOrder: 'asc', advanceRange: undefined, shinyTypes: undefined, speciesIds: undefined, abilityIndices: undefined, levelRange: undefined, genders: undefined } }),
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
  const required: (keyof GenerationParamsHex)[] = ['baseSeedHex','offsetHex','maxAdvances','maxResults','version','encounterType','tid','sid','syncEnabled','syncNatureId','stopAtFirstShiny','stopOnCap','batchSize','memoryLink','newGame','noSave'];
  return required.every(k => (d as Record<string, unknown>)[k] !== undefined);
}

export function getCurrentHexParams(state: GenerationSlice): GenerationParamsHex | null {
  return state.params ? generationParamsToHex(state.params) : null;
}

// 結果フィルタ+ソート用セレクタ（簡易版）
// メモ化キャッシュ（単純参照比較）
let _filteredSortedCache: {
  resultsRef: GenerationResult[];
  filtersRef: GenerationFilters;
  output: GenerationResult[];
} | null = null;

export const selectFilteredSortedResults = (s: GenerationSlice) => {
  const { results, filters } = s;
  if (_filteredSortedCache && _filteredSortedCache.resultsRef === results && _filteredSortedCache.filtersRef === filters) {
    return _filteredSortedCache.output;
  }
  let arr: GenerationResult[] = results;
  if (filters.shinyOnly) arr = arr.filter(r => r.shiny_type !== 0);
  if (filters.shinyTypes && filters.shinyTypes.length > 0) {
    const set = new Set(filters.shinyTypes);
    arr = arr.filter(r => set.has(r.shiny_type));
  }
  if (filters.natureIds && filters.natureIds.length > 0) {
    const nset = new Set(filters.natureIds);
    arr = arr.filter(r => nset.has(r.nature));
  }
  if (filters.advanceRange) {
    const { min, max } = filters.advanceRange;
    if (min != null) arr = arr.filter(r => r.advance >= min);
    if (max != null) arr = arr.filter(r => r.advance <= max);
  }
  // --- Conditional resolved-based filters ---
  const needsResolved = (
    (filters.speciesIds && filters.speciesIds.length > 0) ||
    (filters.levelRange && (filters.levelRange.min != null || filters.levelRange.max != null)) ||
    (filters.abilityIndices && filters.abilityIndices.length > 0) ||
    (filters.genders && filters.genders.length > 0)
  );
  if (needsResolved) {
    // 解決結果を一度だけ取得
    const resolved = selectResolvedResults(s);
    // map: pid+advance で紐付け（advance は一意性高い）
    // 直接 resolved 配列でフィルタ後に元 raw 参照を返却
    let resolvedArr = resolved;
    if (filters.speciesIds && filters.speciesIds.length > 0) {
      const sp = new Set(filters.speciesIds);
      resolvedArr = resolvedArr.filter(r => r.speciesId != null && sp.has(r.speciesId));
    }
    if (filters.levelRange) {
      const { min, max } = filters.levelRange;
      if (min != null) resolvedArr = resolvedArr.filter(r => r.level != null && r.level >= min);
      if (max != null) resolvedArr = resolvedArr.filter(r => r.level != null && r.level <= max);
    }
    // ability/gender は species 選択時のみ有効
    const speciesSelected = filters.speciesIds && filters.speciesIds.length > 0;
    if (speciesSelected && filters.abilityIndices && filters.abilityIndices.length > 0) {
      const aset = new Set(filters.abilityIndices);
      resolvedArr = resolvedArr.filter(r => r.abilityIndex != null && aset.has(r.abilityIndex));
    }
    if (speciesSelected && filters.genders && filters.genders.length > 0) {
      const gset = new Set(filters.genders);
      resolvedArr = resolvedArr.filter(r => r.gender && gset.has(r.gender));
    }
    // raw 配列へ復元: seed/pid/advance マッチ (advance で検索)
    const advanceSet = new Set(resolvedArr.map(r => r.pid.toString() + ':' + r.seed.toString()));
    arr = arr.filter(r => advanceSet.has((r.pid >>> 0).toString() + ':' + r.seed.toString()));
  }
  const field = filters.sortField || 'advance';
  const order = filters.sortOrder === 'desc' ? -1 : 1;
  const cmp = (a: GenerationResult, b: GenerationResult) => {
    let av:number, bv:number;
    switch(field) {
      case 'pid': av = a.pid >>> 0; bv = b.pid >>> 0; break;
      case 'nature': av = a.nature; bv = b.nature; break;
      case 'shiny': av = a.shiny_type; bv = b.shiny_type; break;
      // species / ability / level は未解決 raw のため一旦 advance 安全フォールバック（UI側で並び替え予定）
      case 'species':
      case 'ability':
      case 'level':
        av = a.advance; bv = b.advance; break;
      case 'advance':
      default: av = a.advance; bv = b.advance; break;
    }
    if (av < bv) return -1 * order;
    if (av > bv) return 1 * order;
    return 0;
  };
  const output = [...arr].sort(cmp);
  _filteredSortedCache = { resultsRef: results, filtersRef: filters, output };
  return output;
};

// ===== Resolution / UI adapters =====
import { resolveBatch, toUiReadyPokemon, type ResolutionContext, type ResolvedPokemonData, type UiReadyPokemonData } from '@/lib/generation/pokemon-resolver';

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
  output: UiReadyPokemonData[];
} | null = null;

export const selectUiReadyResults = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja'): UiReadyPokemonData[] => {
  const resolved = selectResolvedResults(s);
  if (_uiReadyCache && _uiReadyCache.resolvedRef === resolved && _uiReadyCache.locale === locale) {
    return _uiReadyCache.output;
  }
  const out = resolved.map(r => toUiReadyPokemon(r, { locale }));
  _uiReadyCache = { resolvedRef: resolved, locale, output: out };
  return out;
};
