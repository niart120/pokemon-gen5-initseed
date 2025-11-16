import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SearchConditions, InitialSeedResult, TargetSeedList, SearchProgress, SearchPreset } from '../types/search';
import type { GenerationParamsHex } from '@/types/generation';
import type { ParallelSearchSettings, AggregatedProgress } from '../types/parallel';
import type { DeviceProfile, DeviceProfileDraft } from '../types/profile';
import { createDefaultDeviceProfile, createDeviceProfile, applyDeviceProfileDraft } from '../types/profile';
import { DEMO_TARGET_SEEDS } from '../data/default-seeds';

import type { GenerationSlice, GenerationFilters } from './generation-store';
import { createGenerationSlice, bindGenerationManager, DEFAULT_GENERATION_DRAFT_PARAMS, createDefaultGenerationFilters } from './generation-store';
import { DEFAULT_LOCALE } from '@/types/i18n';

export type SearchExecutionMode = 'gpu' | 'cpu-parallel';

interface AppStore extends GenerationSlice {
  profiles: DeviceProfile[];
  activeProfileId: string | null;
  setActiveProfile: (profileId: string | null) => void;
  createProfile: (draft: DeviceProfileDraft) => DeviceProfile;
  updateProfile: (profileId: string, draft: Partial<DeviceProfileDraft>) => void;
  deleteProfile: (profileId: string) => void;
  applyProfileToSearch: (profileId?: string) => void;
  applyProfileToGeneration: (profileId?: string) => void;
  locale: 'ja' | 'en';
  setLocale: (locale: 'ja' | 'en') => void;
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
  completeSearch: () => void;

  // Last search duration
  lastSearchDuration: number | null;
  setLastSearchDuration: (duration: number) => void;

  // Parallel search settings
  parallelSearchSettings: ParallelSearchSettings;
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

  // Search execution mode
  searchExecutionMode: SearchExecutionMode;
  setSearchExecutionMode: (mode: SearchExecutionMode) => void;
  
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

const defaultDeviceProfile = createDefaultDeviceProfile();

const defaultSearchConditions: SearchConditions = {
  romVersion: defaultDeviceProfile.romVersion,
  romRegion: defaultDeviceProfile.romRegion,
  hardware: defaultDeviceProfile.hardware,

  timer0VCountConfig: {
    useAutoConfiguration: defaultDeviceProfile.timer0Auto,
    timer0Range: {
      min: defaultDeviceProfile.timer0Range.min,
      max: defaultDeviceProfile.timer0Range.max,
    },
    vcountRange: {
      min: defaultDeviceProfile.vcountRange.min,
      max: defaultDeviceProfile.vcountRange.max,
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
  
  keyInput: 0x0000, // Default: no key input
  macAddress: Array.from(defaultDeviceProfile.macAddress),
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

const detectedHardwareConcurrency = typeof navigator !== 'undefined'
  ? navigator.hardwareConcurrency || 1
  : 1;

const isWebGpuAvailable = typeof navigator !== 'undefined'
  && typeof (navigator as Navigator & { gpu?: unknown }).gpu !== 'undefined';

const defaultSearchExecutionMode: SearchExecutionMode = isWebGpuAvailable
  ? 'gpu'
  : 'cpu-parallel';

const defaultParallelSearchSettings: ParallelSearchSettings = {
  maxWorkers: navigator.hardwareConcurrency || 4,
  chunkStrategy: 'time-based',
};

function resolveProfile(state: Pick<AppStore, 'profiles' | 'activeProfileId'>, profileId?: string | null): DeviceProfile | undefined {
  if (!state.profiles.length) return undefined;
  const targetId = profileId ?? state.activeProfileId;
  if (!targetId) {
    return state.profiles[0];
  }
  return state.profiles.find((profile) => profile.id === targetId) ?? state.profiles[0];
}

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
  staticEncounterId: AppStore['staticEncounterId'];
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
    staticEncounterId: state.staticEncounterId ?? null,
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

function normalizeRestoredFilters(input: unknown): GenerationFilters {
  const defaults = createDefaultGenerationFilters();
  if (!input || typeof input !== 'object') {
    return defaults;
  }

  const candidate = input as Partial<GenerationFilters> & Record<string, unknown>;
  if (typeof candidate.shinyMode === 'string') {
    return {
      sortField: typeof candidate.sortField === 'string' ? candidate.sortField : defaults.sortField,
      sortOrder: candidate.sortOrder === 'desc' ? 'desc' : 'asc',
      shinyMode: candidate.shinyMode === 'shiny' || candidate.shinyMode === 'non-shiny' ? candidate.shinyMode : 'all',
      speciesIds: Array.isArray(candidate.speciesIds) ? [...candidate.speciesIds] : [],
      natureIds: Array.isArray(candidate.natureIds) ? [...candidate.natureIds] : [],
      abilityIndices: Array.isArray(candidate.abilityIndices) ? [...candidate.abilityIndices] as (0 | 1 | 2)[] : [],
      genders: Array.isArray(candidate.genders) ? [...candidate.genders] as ('M' | 'F' | 'N')[] : [],
      levelRange: candidate.levelRange ? { ...candidate.levelRange } : undefined,
      statRanges: candidate.statRanges ? { ...candidate.statRanges } : {},
    };
  }

  const legacy = input as Record<string, unknown>;
  const shinyOnly = Boolean(legacy.shinyOnly);
  let shinyMode: 'all' | 'shiny' | 'non-shiny' = shinyOnly ? 'shiny' : 'all';
  const shinyTypes = Array.isArray(legacy.shinyTypes) ? legacy.shinyTypes as number[] : [];
  if (shinyTypes.length === 1) {
    if (shinyTypes[0] === 0) shinyMode = 'non-shiny';
    else shinyMode = 'shiny';
  }

  return {
    sortField: typeof legacy.sortField === 'string' ? legacy.sortField as GenerationFilters['sortField'] : defaults.sortField,
    sortOrder: legacy.sortOrder === 'desc' ? 'desc' : 'asc',
    shinyMode,
    speciesIds: Array.isArray(legacy.speciesIds) ? [...legacy.speciesIds as number[]] : [],
    natureIds: Array.isArray(legacy.natureIds) ? [...legacy.natureIds as number[]] : [],
    abilityIndices: Array.isArray(legacy.abilityIndices) ? [...legacy.abilityIndices as (0 | 1 | 2)[]] : [],
    genders: Array.isArray(legacy.genders) ? [...legacy.genders as ('M' | 'F' | 'N')[]] : [],
    levelRange: undefined,
    statRanges: {},
  };
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
    filters: normalizeRestoredFilters(o.filters),
    metrics: normalizedStatus === 'idle' ? {} : (o.metrics ?? {}),
    internalFlags: normalizedStatus === 'idle'
      ? { receivedAnyBatch: false }
      : (o.internalFlags ?? { receivedAnyBatch: false }),
    staticEncounterId: o.staticEncounterId ?? null,
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
      profiles: [defaultDeviceProfile],
      activeProfileId: defaultDeviceProfile.id,
      setActiveProfile: (profileId) => {
        set({ activeProfileId: profileId });
        const profile = resolveProfile(get(), profileId);
        if (profile) {
          get().applyProfileToSearch(profile.id);
          get().applyProfileToGeneration(profile.id);
        }
      },
      createProfile: (draft) => {
        const profile = createDeviceProfile(draft);
        set((state) => ({
          profiles: [...state.profiles, profile],
          activeProfileId: profile.id,
        }));
        get().applyProfileToSearch(profile.id);
        get().applyProfileToGeneration(profile.id);
        return profile;
      },
      updateProfile: (profileId, draft) => {
        set((state) => {
          const existing = state.profiles.find((p) => p.id === profileId);
          if (!existing) return {};
          const updated = applyDeviceProfileDraft(existing, draft);
          const profiles = state.profiles.map((p) => (p.id === profileId ? updated : p));
          return { profiles };
        });
        const profile = resolveProfile(get(), profileId);
        if (profile) {
          get().applyProfileToSearch(profile.id);
          get().applyProfileToGeneration(profile.id);
        }
      },
      deleteProfile: (profileId) => {
        set((state) => {
          const profiles = state.profiles.filter((p) => p.id !== profileId);
          let activeProfileId = state.activeProfileId;
          if (activeProfileId === profileId) {
            activeProfileId = profiles[0]?.id ?? null;
          }
          return { profiles, activeProfileId };
        });
        const profile = resolveProfile(get());
        if (profile) {
          get().applyProfileToSearch(profile.id);
          get().applyProfileToGeneration(profile.id);
        }
      },
      applyProfileToSearch: (profileId) => {
        const state = get();
        const profile = resolveProfile(state, profileId);
        if (!profile) return;
        set((current) => ({
          searchConditions: {
            ...current.searchConditions,
            romVersion: profile.romVersion,
            romRegion: profile.romRegion,
            hardware: profile.hardware,
            timer0VCountConfig: {
              ...current.searchConditions.timer0VCountConfig,
              useAutoConfiguration: profile.timer0Auto,
              timer0Range: {
                min: profile.timer0Range.min,
                max: profile.timer0Range.max,
              },
              vcountRange: {
                min: profile.vcountRange.min,
                max: profile.vcountRange.max,
              },
            },
            macAddress: Array.from(profile.macAddress),
          },
        }));
      },
      applyProfileToGeneration: (profileId) => {
        const state = get();
        const profile = resolveProfile(state, profileId);
        if (!profile) return;
        state.setDraftParams({
          version: profile.romVersion,
          tid: profile.tid,
          sid: profile.sid,
          shinyCharm: profile.shinyCharm,
          newGame: profile.newGame,
          withSave: profile.withSave,
          memoryLink: profile.memoryLink,
        });
      },
      locale: DEFAULT_LOCALE,
      setLocale: (locale) => set({ locale }),
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
      completeSearch: () =>
        set((state) => ({
          searchProgress: {
            ...state.searchProgress,
            isRunning: false,
            isPaused: false,
            canPause: false,
          },
        })),

      // Last search duration
      lastSearchDuration: null,
      setLastSearchDuration: (duration) => set({ lastSearchDuration: duration }),

      // Parallel search settings
      parallelSearchSettings: defaultParallelSearchSettings,
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
      setActiveTab: (tab) => {
        set({ activeTab: tab });
      },
      
      // Wake Lock settings for preventing screen sleep on mobile devices
      wakeLockEnabled: false,
      setWakeLockEnabled: (enabled) => set({ wakeLockEnabled: enabled }),

      // Search execution mode
      searchExecutionMode: defaultSearchExecutionMode,
      setSearchExecutionMode: (mode) =>
        set({
          searchExecutionMode: mode,
        }),
      
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
        locale: state.locale,
        profiles: state.profiles,
        activeProfileId: state.activeProfileId,
        searchConditions: state.searchConditions,
        targetSeeds: state.targetSeeds,
        parallelSearchSettings: state.parallelSearchSettings,
    searchExecutionMode: state.searchExecutionMode,
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
        const merged = {
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
        if (!merged.profiles || merged.profiles.length === 0) {
          merged.profiles = current.profiles;
          merged.activeProfileId = current.activeProfileId;
        } else if (!merged.activeProfileId || !merged.profiles.some((p) => p.id === merged.activeProfileId)) {
          merged.activeProfileId = merged.profiles[0]?.id ?? null;
        }
        return merged;
      },
      // migrate 不要（新キーで旧スキーマ非対応）
      migrate: (s) => s as AppStore,
    },
   ),
);

// バインド（store インスタンス生成後）
bindGenerationManager(() => useAppStore.getState());