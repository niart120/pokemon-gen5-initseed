import { create } from 'zustand';
import {
  type EggGenerationParams,
  type EggGenerationParamsHex,
  type EnumeratedEggData,
  type EggCompletion,
  hexParamsToEggParams,
  validateEggParams,
  createDefaultEggParamsHex,
  deriveEggGameMode,
} from '@/types/egg';
import { EggWorkerManager } from '@/lib/egg/egg-worker-manager';
import type { DeviceProfile } from '@/types/profile';

export type EggStatus = 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'error';

const MAX_DISPLAY_RESULTS = 10000;

interface EggStore {
  // パラメータ
  draftParams: EggGenerationParamsHex;
  params: EggGenerationParams | null;
  validationErrors: string[];

  // 実行状態
  status: EggStatus;
  workerManager: EggWorkerManager | null;

  // 結果
  results: EnumeratedEggData[];
  lastCompletion: EggCompletion | null;
  errorMessage: string | null;

  // アクション
  updateDraftParams: (updates: Partial<EggGenerationParamsHex>) => void;
  updateDraftConditions: (updates: Partial<EggGenerationParamsHex['conditions']>) => void;
  updateDraftParentsMale: (ivs: EggGenerationParamsHex['parents']['male']) => void;
  updateDraftParentsFemale: (ivs: EggGenerationParamsHex['parents']['female']) => void;
  applyProfile: (profile: DeviceProfile) => void;
  validateDraft: () => boolean;
  startGeneration: () => Promise<void>;
  stopGeneration: () => void;
  clearResults: () => void;
  reset: () => void;
}

export const useEggStore = create<EggStore>((set, get) => ({
  draftParams: createDefaultEggParamsHex(),
  params: null,
  validationErrors: [],
  status: 'idle',
  workerManager: null,
  results: [],
  lastCompletion: null,
  errorMessage: null,

  updateDraftParams: (updates) => {
    set((state) => ({
      draftParams: { ...state.draftParams, ...updates },
    }));
  },

  updateDraftConditions: (updates) => {
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        conditions: { ...state.draftParams.conditions, ...updates },
      },
    }));
  },

  updateDraftParentsMale: (ivs) => {
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        parents: { ...state.draftParams.parents, male: ivs },
      },
    }));
  },

  updateDraftParentsFemale: (ivs) => {
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        parents: { ...state.draftParams.parents, female: ivs },
      },
    }));
  },

  applyProfile: (profile) => {
    const gameMode = deriveEggGameMode(profile.romVersion, profile.newGame);
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        gameMode,
        conditions: {
          ...state.draftParams.conditions,
          tid: profile.tid,
          sid: profile.sid,
        },
      },
    }));
  },

  validateDraft: () => {
    const draft = get().draftParams;
    try {
      const params = hexParamsToEggParams(draft);
      const errors = validateEggParams(params);
      set({ validationErrors: errors, params: errors.length === 0 ? params : null });
      return errors.length === 0;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ validationErrors: [message], params: null });
      return false;
    }
  },

  startGeneration: async () => {
    const { params, workerManager: existingManager } = get();
    if (!params) {
      set({ errorMessage: 'Invalid parameters' });
      return;
    }

    // Worker初期化
    const manager = existingManager || new EggWorkerManager();

    manager.clearCallbacks();

    manager
      .onResults((payload) => {
        set((state) => {
          const newResults = [...state.results, ...payload.results];
          // 上限を超えたら古い結果を削除
          if (newResults.length > MAX_DISPLAY_RESULTS) {
            return {
              results: newResults.slice(-MAX_DISPLAY_RESULTS),
            };
          }
          return { results: newResults };
        });
      })
      .onComplete((completion) => {
        set({
          status: 'completed',
          lastCompletion: completion,
        });
      })
      .onError((message, _category, fatal) => {
        set({
          status: fatal ? 'error' : get().status,
          errorMessage: message,
        });
      });

    set({
      workerManager: manager,
      status: 'starting',
      results: [],
      lastCompletion: null,
      errorMessage: null,
    });

    try {
      await manager.start(params);
      set({ status: 'running' });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ status: 'error', errorMessage: message });
    }
  },

  stopGeneration: () => {
    const { workerManager } = get();
    if (workerManager) {
      set({ status: 'stopping' });
      workerManager.stop();
    }
  },

  clearResults: () => {
    set({ results: [], lastCompletion: null, errorMessage: null });
  },

  reset: () => {
    const { workerManager } = get();
    if (workerManager) {
      workerManager.terminate();
    }
    set({
      draftParams: createDefaultEggParamsHex(),
      params: null,
      validationErrors: [],
      status: 'idle',
      workerManager: null,
      results: [],
      lastCompletion: null,
      errorMessage: null,
    });
  },
}));
