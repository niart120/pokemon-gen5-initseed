import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationProgress, GenerationCompletion, GenerationResultBatch, GenerationResult } from '@/types/generation';
import { validateGenerationParams } from '@/types/generation';

// 単一インスタンスマネージャ（UI からは slice 経由で操作）
const manager = new GenerationWorkerManager();

export type GenerationStatus = 'idle' | 'starting' | 'running' | 'paused' | 'stopping' | 'completed' | 'error';

export interface GenerationFilters {
  shinyOnly: boolean;
  natureIds: number[]; // 追加フィルタ用プレースホルダ
}

export interface GenerationSliceState {
  params: GenerationParams | null;
  draftParams: Partial<GenerationParams>;
  validationErrors: string[];
  status: GenerationStatus;
  progress: GenerationProgress | null;
  results: GenerationResult[]; // GenerationResult 型 (UnresolvedPokemonData + advance)
  lastCompletion: GenerationCompletion | null;
  error: string | null;
  filters: GenerationFilters;
  metrics: { startTime?: number; lastUpdateTime?: number };
  internalFlags: { receivedAnyBatch: boolean };
}

export interface GenerationSliceActions {
  setDraftParams: (partial: Partial<GenerationParams>) => void;
  validateDraft: () => void;
  commitParams: () => boolean;
  startGeneration: () => Promise<boolean>;
  pauseGeneration: () => void;
  resumeGeneration: () => void;
  stopGeneration: () => void;
  clearResults: () => void;
  applyFilters: (partial: Partial<GenerationFilters>) => void;
  resetGenerationState: () => void;
  // 内部コールバック（manager から）
  _onWorkerProgress: (p: GenerationProgress) => void;
  _onWorkerBatch: (b: GenerationResultBatch) => void;
  _onWorkerComplete: (c: GenerationCompletion) => void;
  _onWorkerStopped: (reason: string) => void;
  _onWorkerError: (err: string) => void;
}

export type GenerationSlice = GenerationSliceState & GenerationSliceActions;

export const createGenerationSlice = (set: any, get: any): GenerationSlice => ({
  params: null,
  draftParams: {
    baseSeed: 1n,
    offset: 0n,
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
    batchSize: 1000,
  },
  validationErrors: [],
  status: 'idle',
  progress: null,
  results: [],
  lastCompletion: null,
  error: null,
  filters: { shinyOnly: false, natureIds: [] },
  metrics: {},
  internalFlags: { receivedAnyBatch: false },

  setDraftParams: (partial) => {
    set((state: any) => ({ draftParams: { ...state.draftParams, ...partial } }));
  },
  validateDraft: () => {
    const { draftParams } = get();
    const errors = validateGenerationParams(draftParams as any);
    set({ validationErrors: errors });
  },
  commitParams: () => {
    const { draftParams } = get();
    const errors = validateGenerationParams(draftParams as any);
    set({ validationErrors: errors });
    if (errors.length) return false;
    set({ params: draftParams });
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
    } catch (e: any) {
      set({ status: 'error', error: e?.message || 'start-failed' });
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
  applyFilters: (partial) => set((state: any) => ({ filters: { ...state.filters, ...partial } })),
  resetGenerationState: () => set({
    status: 'idle', progress: null, results: [], lastCompletion: null, error: null, metrics: {}, internalFlags: { receivedAnyBatch: false }
  }),

  _onWorkerProgress: (p) => {
    set({ progress: p, metrics: { ...get().metrics, lastUpdateTime: performance.now() } });
  },
  _onWorkerBatch: (b) => {
    set((state: any) => {
      if (state.results.length >= (state.params?.maxResults || Infinity)) return {};
      const capacityLeft = (state.params?.maxResults || Infinity) - state.results.length;
      const slice = b.results.slice(0, capacityLeft);
      return { results: state.results.concat(slice), internalFlags: { receivedAnyBatch: true } };
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
