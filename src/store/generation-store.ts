import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationProgress, GenerationCompletion, GenerationResultBatch, GenerationResult, GenerationParamsHex } from '@/types/generation';
import { validateGenerationParams, hexParamsToGenerationParams, generationParamsToHex } from '@/types/generation';

// 単一インスタンスマネージャ（UI からは slice 経由で操作）
const manager = new GenerationWorkerManager();

export type GenerationStatus = 'idle' | 'starting' | 'running' | 'paused' | 'stopping' | 'completed' | 'error';

export interface GenerationFilters {
  shinyOnly: boolean;
  natureIds: number[]; // 追加フィルタ用プレースホルダ
  sortField?: 'advance' | 'pid' | 'nature' | 'shiny';
  sortOrder?: 'asc' | 'desc';
  advanceRange?: { min?: number; max?: number };
  shinyTypes?: number[]; // 0/1/2 指定。空 or undefined は全許可
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
}

export interface GenerationSliceActions {
  setDraftParams: (partial: Partial<GenerationParamsHex>) => void;
  validateDraft: () => void;
  commitParams: () => boolean;
  startGeneration: () => Promise<boolean>;
  pauseGeneration: () => void;
  resumeGeneration: () => void;
  stopGeneration: () => void;
  clearResults: () => void;
  applyFilters: (partial: Partial<GenerationFilters>) => void;
  resetGenerationState: () => void;
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
    maxAdvances: 10000,
    maxResults: 1000,
    version: 'B',
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
  batchSize: 1000, // 固定 (UI 非表示)
  abilityMode: 'none',
  shinyCharm: false,
  memoryLink: false,
  },
  validationErrors: [],
  status: 'idle',
  progress: null,
  results: [],
  lastCompletion: null,
  error: null,
  filters: { shinyOnly: false, natureIds: [], sortField: 'advance', sortOrder: 'asc', advanceRange: undefined, shinyTypes: undefined },
  metrics: {},
  internalFlags: { receivedAnyBatch: false },

  setDraftParams: (partial) => {
    set((state: GenerationSlice) => ({ draftParams: { ...state.draftParams, ...partial } }));
  },
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
  resetGenerationFilters: () => set({ filters: { shinyOnly: false, natureIds: [], sortField: 'advance', sortOrder: 'asc', advanceRange: undefined, shinyTypes: undefined } }),
  resetGenerationState: () => set({
    status: 'idle',
    progress: null,
    results: [],
    lastCompletion: null,
    error: null,
    metrics: {},
    internalFlags: { receivedAnyBatch: false },
  }),

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
  const required: (keyof GenerationParamsHex)[] = ['baseSeedHex','offsetHex','maxAdvances','maxResults','version','encounterType','tid','sid','syncEnabled','syncNatureId','stopAtFirstShiny','stopOnCap','batchSize'];
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
  const field = filters.sortField || 'advance';
  const order = filters.sortOrder === 'desc' ? -1 : 1;
  const cmp = (a: GenerationResult, b: GenerationResult) => {
    let av:number, bv:number;
    switch(field) {
      case 'pid': av = a.pid >>> 0; bv = b.pid >>> 0; break;
      case 'nature': av = a.nature; bv = b.nature; break;
      case 'shiny': av = a.shiny_type; bv = b.shiny_type; break;
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
