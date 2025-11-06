import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SearchConditions, InitialSeedResult, TargetSeedList, SearchProgress, SearchPreset } from '../types/search';
import type { GenerationParamsHex } from '@/types/generation';
import type { ROMVersion, ROMRegion, Hardware } from '../types/rom';
import type { ParallelSearchSettings, AggregatedProgress } from '../types/parallel';
import { DEMO_TARGET_SEEDS } from '../data/default-seeds';

import type { GenerationSlice } from './generation-store';
import { createGenerationSlice, bindGenerationManager, DEFAULT_GENERATION_DRAFT_PARAMS } from './generation-store';

interface AppStore extends GenerationSlice {
  // Search conditions
  searchConditions: SearchConditions;
  setSearchConditions: (conditions: Partial<SearchConditions>) => void;
  resetSearchConditions: () => void;

  // Target seeds
  targetSeeds: TargetSeedList;
  setTargetSeeds: (seeds: number[]) => void;
  addTargetSeed: (seed: number) => void;
  removeTargetSeed: (seed: number) => void;
  clearTargetSeeds: () => void;

  // Search results
  searchResults: InitialSeedResult[];
  setSearchResults: (results: InitialSeedResult[]) => void;
  addSearchResult: (result: InitialSeedResult) => void;
  clearSearchResults: () => void;

  // Search progress
  searchProgress: SearchProgress;
  setSearchProgress: (progress: Partial<SearchProgress>) => void;
  startSearch: () => void;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;

  // Last search duration
  lastSearchDuration: number | null;
  setLastSearchDuration: (duration: number) => void;

  // Parallel search settings
  parallelSearchSettings: ParallelSearchSettings;
  setParallelSearchEnabled: (enabled: boolean) => void;
  setMaxWorkers: (count: number) => void;
  setChunkStrategy: (strategy: ParallelSearchSettings['chunkStrategy']) => void;

  // Parallel search progress
  parallelProgress: AggregatedProgress | null;
  setParallelProgress: (progress: AggregatedProgress | null) => void;

  // UI state
  activeTab: string;
  setActiveTab: (tab: string) => void;
  
  // Wake Lock settings for preventing screen sleep on mobile devices
  wakeLockEnabled: boolean;
  setWakeLockEnabled: (enabled: boolean) => void;
  
  // Raw target seed input
  targetSeedInput: string;
  setTargetSeedInput: (input: string) => void;

  // Presets
  presets: SearchPreset[];
  setPresets: (presets: SearchPreset[]) => void;
  addPreset: (preset: SearchPreset) => void;
  removePreset: (presetId: string) => void;
  loadPreset: (presetId: string) => void;
}

const defaultSearchConditions: SearchConditions = {
  romVersion: 'B' as ROMVersion,
  romRegion: 'JPN' as ROMRegion,
  hardware: 'DS' as Hardware,
  
  timer0VCountConfig: {
    useAutoConfiguration: true,
    timer0Range: {
      min: 3193,
      max: 3194,
    },
    vcountRange: {
      min: 95,
      max: 95,
    },
  },
  
  dateRange: {
    startYear: 2000,
    endYear: 2099,
    startMonth: 1,
    endMonth: 12,
    startDay: 1,
    endDay: 31,
    startHour: 0,
    endHour: 23,
    startMinute: 0,
    endMinute: 59,
    startSecond: 0,
    endSecond: 59,
  },
  
  keyInput: 0x2FFF, // Default: no keys pressed
  macAddress: [0x00, 0x1B, 0x2C, 0x3D, 0x4E, 0x5F],
};

const defaultSearchProgress: SearchProgress = {
  isRunning: false,
  currentStep: 0,
  totalSteps: 0,
  currentDateTime: null,
  elapsedTime: 0,
  estimatedTimeRemaining: 0,
  matchesFound: 0,
  canPause: false,
  isPaused: false,
};

const defaultParallelSearchSettings: ParallelSearchSettings = {
  enabled: true,
  maxWorkers: navigator.hardwareConcurrency || 4,
  chunkStrategy: 'time-based',
};

// Use demo seeds for initial setup (development only)
if (import.meta.env.DEV) {
  console.warn('Using demo target seeds:', DEMO_TARGET_SEEDS.map(s => '0x' + s.toString(16).padStart(8, '0')));
}

// 以前の BigInt 変換ロジックは撤去。persist 対象を必要最低限に制限。
interface PersistedGenerationMinimal {
  draftParams: AppStore['draftParams'];
  // NOTE: params には bigint を含むため永続化しない（JSON.stringify 失敗回避）
  validationErrors: string[];
  status: AppStore['status'];
  lastCompletion: AppStore['lastCompletion'];
  error: string | null;
  filters: AppStore['filters'];
  metrics: AppStore['metrics'];
  internalFlags: AppStore['internalFlags'];
}
function extractGenerationForPersist(state: AppStore): PersistedGenerationMinimal {
  return {
    draftParams: state.draftParams,
    validationErrors: state.validationErrors,
    status: state.status,
    lastCompletion: state.lastCompletion,
    error: state.error,
    filters: state.filters,
    metrics: state.metrics,
    internalFlags: state.internalFlags,
  };
}

function mergeDraftParams(restored: Partial<GenerationSlice['draftParams']> | undefined): GenerationSlice['draftParams'] {
  const source = (restored ?? {}) as Partial<GenerationParamsHex>;
  const merged: GenerationParamsHex = { ...DEFAULT_GENERATION_DRAFT_PARAMS };
  type DraftKey = keyof GenerationParamsHex;
  for (const key of Object.keys(DEFAULT_GENERATION_DRAFT_PARAMS) as DraftKey[]) {
    const value = source[key];
    if (value !== undefined) {
      (merged as Record<DraftKey, unknown>)[key] = value;
    }
  }
  return merged;
}

function reviveGenerationMinimal(obj: unknown): Partial<GenerationSlice> {
  if (!obj || typeof obj !== 'object') return {};
  const o = obj as Partial<PersistedGenerationMinimal>;
  const normalizedStatus = normalizeRestoredStatus(o.status);
  return {
    draftParams: mergeDraftParams(o.draftParams),
    // params は非永続化（draft から再生成する設計）
    params: null,
    validationErrors: o.validationErrors ?? [],
    status: normalizedStatus,
    lastCompletion: o.lastCompletion ?? null,
    error: o.error ?? null,
    filters: o.filters ?? { shinyOnly: false, natureIds: [] },
    metrics: normalizedStatus === 'idle' ? {} : (o.metrics ?? {}),
    internalFlags: normalizedStatus === 'idle'
      ? { receivedAnyBatch: false }
      : (o.internalFlags ?? { receivedAnyBatch: false }),
  };
}

function normalizeRestoredStatus(status: AppStore['status'] | undefined): AppStore['status'] {
  if (!status) return 'idle';
  if (status === 'running' || status === 'starting' || status === 'paused' || status === 'stopping') {
    return 'idle';
  }
  return status;
}

export const useAppStore = create<AppStore>()(
  persist<AppStore>(
    (set, get) => ({
      // Generation slice 注入
      ...createGenerationSlice(set, get),
      // 元々の AppStore フィールド
      // Search conditions
      searchConditions: defaultSearchConditions,
      setSearchConditions: (conditions) =>
        set((state) => ({
          searchConditions: { ...state.searchConditions, ...conditions },
        })),
      resetSearchConditions: () =>
        set({ searchConditions: defaultSearchConditions }),

      // Target seeds
      targetSeeds: { seeds: DEMO_TARGET_SEEDS },
      setTargetSeeds: (seeds) => set({ targetSeeds: { seeds } }),
      addTargetSeed: (seed) =>
        set((state) => ({
          targetSeeds: {
            seeds: [...new Set([...state.targetSeeds.seeds, seed])],
          },
        })),
      removeTargetSeed: (seed) =>
        set((state) => ({
          targetSeeds: {
            seeds: state.targetSeeds.seeds.filter((s) => s !== seed),
          },
        })),
      clearTargetSeeds: () => set({ targetSeeds: { seeds: [] } }),

      // Search results
      searchResults: [],
      setSearchResults: (results) => set({ searchResults: results }),
      addSearchResult: (result) =>
        set((state) => {
          // 効率的な配列追加：スプレッド演算子による新配列作成を避ける
          const newResults = state.searchResults.slice();
          newResults.push(result);
          return { searchResults: newResults };
        }),
      clearSearchResults: () => set({ searchResults: [] }),

      // Search progress
      searchProgress: defaultSearchProgress,
      setSearchProgress: (progress) =>
        set((state) => ({
          searchProgress: { ...state.searchProgress, ...progress },
        })),
      startSearch: () =>
        set((state) => ({
          searchProgress: {
            ...state.searchProgress,
            isRunning: true,
            isPaused: false,
            currentStep: 0,
            elapsedTime: 0,
            matchesFound: 0,
          },
        })),
      pauseSearch: () =>
        set((state) => ({
          searchProgress: { ...state.searchProgress, isPaused: true, canPause: true },
        })),
      resumeSearch: () =>
        set((state) => ({
          searchProgress: { ...state.searchProgress, isPaused: false },
        })),
      stopSearch: () =>
        set((state) => ({
          searchProgress: {
            ...defaultSearchProgress,
            matchesFound: state.searchProgress.matchesFound,
          },
        })),

      // Last search duration
      lastSearchDuration: null,
      setLastSearchDuration: (duration) => set({ lastSearchDuration: duration }),

      // Parallel search settings
      parallelSearchSettings: defaultParallelSearchSettings,
      setParallelSearchEnabled: (enabled) =>
        set((state) => ({
          parallelSearchSettings: { ...state.parallelSearchSettings, enabled },
        })),
      setMaxWorkers: (count) =>
        set((state) => ({
          parallelSearchSettings: { ...state.parallelSearchSettings, maxWorkers: count },
        })),
      setChunkStrategy: (strategy) =>
        set((state) => ({
          parallelSearchSettings: { ...state.parallelSearchSettings, chunkStrategy: strategy },
        })),

      // Parallel search progress
      parallelProgress: null,
      setParallelProgress: (progress) => set({ parallelProgress: progress }),

      // UI state
      activeTab: 'search',
      setActiveTab: (tab) => set({ activeTab: tab }),
      
      // Wake Lock settings for preventing screen sleep on mobile devices
      wakeLockEnabled: false,
      setWakeLockEnabled: (enabled) => set({ wakeLockEnabled: enabled }),
      
      // Raw target seed input
      targetSeedInput: DEMO_TARGET_SEEDS.map(s => '0x' + s.toString(16).padStart(8, '0')).join('\n'),
      setTargetSeedInput: (input) => set({ targetSeedInput: input }),

      // Presets
      presets: [],
      setPresets: (presets) => set({ presets }),
      addPreset: (preset) =>
        set((state) => ({
          presets: [...state.presets, preset],
        })),
      removePreset: (presetId) =>
        set((state) => ({
          presets: state.presets.filter((p) => p.id !== presetId),
        })),
      loadPreset: (presetId) =>
        set((state) => {
          const preset = state.presets.find((p) => p.id === presetId);
          if (preset) {
            return {
              searchConditions: preset.conditions,
              presets: state.presets.map((p) =>
                p.id === presetId ? { ...p, lastUsed: new Date() } : p
              ),
            };
          }
          return state;
        }),
  }),
    {
      name: 'app-store',
      version: 1,
      // BigInt を含む値が万一混入しても JSON.stringify で落ちないように、
      // シリアライズ/デシリアライズをカスタマイズ（型は起動時に復元）。
      storage: (() => {
        const storage = {
          getItem: (name: string) => {
            const raw = localStorage.getItem(name);
            if (raw == null) return null;
            try {
              const parsed = JSON.parse(raw, (_k, v) => {
                if (typeof v === 'string' && /^__bigint__:.+/.test(v)) {
                  // 復元は必要箇所（types 側）で行うため、ここでは文字列のまま返す
                  // 例: "__bigint__:0x1234" -> そのまま文字列
                  return v;
                }
                return v;
              });
              return parsed as unknown as string;
            } catch {
              return raw as unknown as string;
            }
          },
          setItem: (name: string, value: unknown) => {
            const json = JSON.stringify(value, (_k, v) => {
              if (typeof v === 'bigint') {
                // 文字列タグ化して stringify 可能に（復元は型側の変換で対応）
                return `__bigint__:${'0x' + v.toString(16)}`;
              }
              return v;
            });
            localStorage.setItem(name, json);
          },
          removeItem: (name: string) => localStorage.removeItem(name),
        } as const;
        // 型が合うように adapter を返す
        return {
          getItem: async (name: string) => storage.getItem(name),
          setItem: async (name: string, value: unknown) => storage.setItem(name, value),
          removeItem: async (name: string) => storage.removeItem(name),
        } as unknown as Parameters<typeof persist<AppStore>>[1]['storage'];
      })(),
      partialize: (state: AppStore) => ({
        searchConditions: state.searchConditions,
        targetSeeds: state.targetSeeds,
        parallelSearchSettings: state.parallelSearchSettings,
        activeTab: state.activeTab,
        wakeLockEnabled: state.wakeLockEnabled,
        targetSeedInput: state.targetSeedInput,
        presets: state.presets,
        __generation: extractGenerationForPersist(state),
      }) as unknown as AppStore,
      merge: (persisted: unknown, current: AppStore) => {
        if (!persisted || typeof persisted !== 'object') return current;
        const { __generation, ...rest } = persisted as Partial<AppStore> & { __generation?: unknown };
        const revived = __generation ? reviveGenerationMinimal(__generation) : {};
        return {
          ...current,
          ...rest,
          ...revived,
          searchResults: [],
          searchProgress: { ...defaultSearchProgress },
          parallelProgress: null,
          lastSearchDuration: null,
          results: current.results,
          progress: current.progress,
        } as AppStore;
      },
      // migrate 不要（新キーで旧スキーマ非対応）
      migrate: (s) => s as AppStore,
    },
   ),
);

// バインド（store インスタンス生成後）
bindGenerationManager(() => useAppStore.getState());