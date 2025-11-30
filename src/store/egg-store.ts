import { create } from 'zustand';
import {
  type EggGenerationParams,
  type EggGenerationParamsHex,
  type EnumeratedEggDataWithBootTiming,
  type EggCompletion,
  type DerivedEggSeedRunState,
  type DerivedEggSeedJob,
  type EggBootTimingDraft,
  hexParamsToEggParams,
  validateEggParams,
  createDefaultEggParamsHex,
  createDefaultEggFilter,
  deriveEggGameMode,
} from '@/types/egg';
import {
  deriveBootTimingEggSeedJobs,
  validateEggBootTimingInputs,
} from '@/lib/egg/boot-timing-egg-derivation';
import {
  createDerivedEggSeedState,
  advanceDerivedEggSeedState,
  markDerivedEggSeedAbort,
  shouldAppendDerivedEggResults,
} from '@/store/modules/egg-boot-timing-runner';
import {
  type EggBootTimingFilters,
  applyBootTimingFilters,
} from '@/lib/egg/egg-result-filter';
import { EggWorkerManager } from '@/lib/egg/egg-worker-manager';
import { keyMaskToNames } from '@/lib/utils/key-input';
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

  // Boot-Timing 状態
  derivedSeedRunState: DerivedEggSeedRunState | null;

  // Boot-Timing フィルター
  bootTimingFilters: EggBootTimingFilters;

  // 結果
  results: EnumeratedEggDataWithBootTiming[];
  lastCompletion: EggCompletion | null;
  errorMessage: string | null;

  // アクション
  updateDraftParams: (updates: Partial<EggGenerationParamsHex>) => void;
  updateDraftConditions: (updates: Partial<EggGenerationParamsHex['conditions']>) => void;
  updateDraftParentsMale: (ivs: EggGenerationParamsHex['parents']['male']) => void;
  updateDraftParentsFemale: (ivs: EggGenerationParamsHex['parents']['female']) => void;
  updateDraftBootTiming: (updates: Partial<EggBootTimingDraft>) => void;
  updateBootTimingFilters: (updates: Partial<EggBootTimingFilters>) => void;
  getFilteredResults: () => EnumeratedEggDataWithBootTiming[];
  applyProfile: (profile: DeviceProfile) => void;
  validateDraft: () => boolean;
  startGeneration: () => Promise<void>;
  stopGeneration: () => void;
  clearResults: () => void;
  resetFilters: () => void;
  reset: () => void;
}

export const useEggStore = create<EggStore>((set, get) => ({
  draftParams: createDefaultEggParamsHex(),
  params: null,
  validationErrors: [],
  status: 'idle',
  workerManager: null,
  derivedSeedRunState: null,
  bootTimingFilters: {},
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

  updateDraftBootTiming: (updates) => {
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        bootTiming: { ...state.draftParams.bootTiming, ...updates },
      },
    }));
  },

  updateBootTimingFilters: (updates) => {
    set((state) => ({
      bootTimingFilters: { ...state.bootTimingFilters, ...updates },
    }));
  },

  getFilteredResults: () => {
    const state = get();
    return applyBootTimingFilters(
      state.results,
      state.bootTimingFilters,
      state.draftParams.seedSourceMode,
    );
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
        // Boot-Timing パラメータをprofileから同期
        bootTiming: {
          ...state.draftParams.bootTiming,
          romVersion: profile.romVersion,
          romRegion: profile.romRegion,
          hardware: profile.hardware,
          timer0Range: { ...profile.timer0Range },
          vcountRange: { ...profile.vcountRange },
          macAddress: [...profile.macAddress] as EggBootTimingDraft['macAddress'],
        },
      },
    }));
  },

  validateDraft: () => {
    const draft = get().draftParams;

    // Boot-Timing モードの場合は追加バリデーション
    if (draft.seedSourceMode === 'boot-timing') {
      const bootTimingErrors = validateEggBootTimingInputs(draft.bootTiming);
      if (bootTimingErrors.length > 0) {
        set({ validationErrors: bootTimingErrors, params: null });
        return false;
      }
    }

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
    const { draftParams, workerManager: existingManager } = get();

    // Boot-Timing モードの場合は導出ジョブを生成
    if (draftParams.seedSourceMode === 'boot-timing') {
      const result = deriveBootTimingEggSeedJobs(draftParams);
      if (!result.ok) {
        set({ errorMessage: result.error, status: 'error' });
        return;
      }
      if (result.jobs.length === 0) {
        set({ errorMessage: 'No boot-timing jobs derived', status: 'error' });
        return;
      }
      const runState = createDerivedEggSeedState(result.jobs);
      set({
        derivedSeedRunState: runState,
        results: [],
        lastCompletion: null,
        errorMessage: null,
        status: 'starting',
      });
      await executeBootTimingJob(get, set, existingManager, result.jobs[0]);
      return;
    }

    // LCGモード (従来通り)
    const params = get().params;
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
          // LCGモード: seedSourceModeを付与
          const enrichedResults: EnumeratedEggDataWithBootTiming[] = payload.results.map(r => ({
            ...r,
            seedSourceMode: 'lcg' as const,
          }));
          const newResults = [...state.results, ...enrichedResults];
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
    const { workerManager, derivedSeedRunState } = get();
    if (derivedSeedRunState) {
      set({ derivedSeedRunState: markDerivedEggSeedAbort(derivedSeedRunState), status: 'stopping' });
    }
    if (workerManager) {
      set({ status: 'stopping' });
      workerManager.stop();
    }
  },

  clearResults: () => {
    set({ results: [], lastCompletion: null, errorMessage: null, derivedSeedRunState: null, bootTimingFilters: {} });
  },

  resetFilters: () => {
    const defaultFilter = createDefaultEggFilter();
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        filter: defaultFilter,
        filterDisabled: false,
      },
      bootTimingFilters: {},
    }));
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
      derivedSeedRunState: null,
      bootTimingFilters: {},
      results: [],
      lastCompletion: null,
      errorMessage: null,
    });
  },
}));

// === Boot-Timing Job Execution Helper ===

type EggStoreGet = () => EggStore;
type EggStoreSet = (partial: Partial<EggStore> | ((state: EggStore) => Partial<EggStore>)) => void;

async function executeBootTimingJob(
  get: EggStoreGet,
  set: EggStoreSet,
  existingManager: EggWorkerManager | null,
  job: DerivedEggSeedJob,
): Promise<void> {
  const manager = existingManager || new EggWorkerManager();

  manager.clearCallbacks();

  const currentJob = job;
  const metadata = currentJob.metadata;

  manager
    .onResults((payload) => {
      set((state) => {
        // Boot-Timing メタデータを各結果に付与
        const enrichedResults: EnumeratedEggDataWithBootTiming[] = payload.results.map(r => ({
          ...r,
          seedSourceMode: 'boot-timing' as const,
          derivedSeedIndex: metadata.derivedSeedIndex,
          seedSourceSeedHex: metadata.seedSourceSeedHex,
          timer0: metadata.timer0,
          vcount: metadata.vcount,
          bootTimestampIso: metadata.bootTimestampIso,
          keyInputNames: keyMaskToNames(metadata.keyMask),
          macAddress: metadata.macAddress,
        }));
        const newResults = shouldAppendDerivedEggResults(state.derivedSeedRunState)
          ? [...state.results, ...enrichedResults]
          : enrichedResults;
        if (newResults.length > MAX_DISPLAY_RESULTS) {
          return { results: newResults.slice(-MAX_DISPLAY_RESULTS) };
        }
        return { results: newResults };
      });
    })
    .onComplete((completion) => {
      const state = get();
      const runState = state.derivedSeedRunState;
      if (!runState) {
        set({ status: 'completed', lastCompletion: completion });
        return;
      }
      const advanceResult = advanceDerivedEggSeedState(runState, completion);
      if (advanceResult.finalCompletion) {
        // 全ジョブ完了
        set({
          status: 'completed',
          lastCompletion: advanceResult.finalCompletion,
          derivedSeedRunState: null,
        });
        return;
      }
      // 次のジョブへ
      if (advanceResult.nextState && advanceResult.nextJob) {
        set({ derivedSeedRunState: advanceResult.nextState });
        void executeBootTimingJob(get, set, manager, advanceResult.nextJob);
      }
    })
    .onError((message, _category, fatal) => {
      set({
        status: fatal ? 'error' : get().status,
        errorMessage: message,
      });
    });

  set({ workerManager: manager, status: 'running' });

  try {
    await manager.start(job.params);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    set({ status: 'error', errorMessage: message });
  }
}
