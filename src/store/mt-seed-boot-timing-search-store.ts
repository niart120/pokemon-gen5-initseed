/**
 * mt-seed-boot-timing-search-store.ts
 * MT Seed 起動時間検索パネル向けのZustand Store
 * EggBootTimingSearchStoreのパターンを流用
 */

import { create } from 'zustand';
import type {
  MtSeedBootTimingSearchParams,
  MtSeedBootTimingSearchResult,
  DateRange,
} from '@/types/mt-seed-boot-timing-search';
import {
  createDefaultMtSeedBootTimingSearchParams,
  validateMtSeedBootTimingSearchParams,
} from '@/types/mt-seed-boot-timing-search';
import type { DeviceProfile } from '@/types/profile';
import type { DailyTimeRange } from '@/types/search';
import {
  getSearchWorkerManager,
  type MtSeedBootTimingSearchCallbacks,
} from '@/lib/search/search-worker-manager';

/**
 * 検索実行状態
 */
export type MtSeedBootTimingSearchStatus =
  | 'idle' // 初期状態・完了後
  | 'starting' // 検索開始中
  | 'running' // 検索実行中
  | 'stopping' // 停止処理中
  | 'paused' // 一時停止中
  | 'completed' // 完了
  | 'error'; // エラー

/**
 * 進捗情報
 */
export interface MtSeedBootTimingSearchProgress {
  /** 処理済み組み合わせ数 */
  processedCombinations: number;
  /** 総組み合わせ数 */
  totalCombinations: number;
  /** 見つかった結果数 */
  foundCount: number;
  /** 進捗率 (0-100) */
  progressPercent: number;
  /** 経過時間 (ms) */
  elapsedMs: number;
  /** 推定残り時間 (ms) */
  estimatedRemainingMs: number;
}

/**
 * 結果フィルター条件
 */
export interface MtSeedBootTimingResultFilters {
  // Timer0フィルター (hex文字列)
  timer0Filter?: string;
  // VCountフィルター (hex文字列)
  vcountFilter?: string;
}

/**
 * 完了情報
 */
export interface MtSeedBootTimingCompletion {
  reason: 'completed' | 'stopped' | 'max_results' | 'error';
  processedCombinations: number;
  totalCombinations: number;
  resultsCount: number;
  elapsedMs: number;
}

const MAX_RESULTS = 1000;

interface MtSeedBootTimingSearchState {
  // --- パラメータ ---
  /** UI入力用ドラフトパラメータ */
  draftParams: MtSeedBootTimingSearchParams;
  /** バリデーション済みパラメータ */
  params: MtSeedBootTimingSearchParams | null;
  /** バリデーションエラー */
  validationErrors: string[];

  // --- 実行状態 ---
  /** 検索状態 */
  status: MtSeedBootTimingSearchStatus;

  // --- 進捗 ---
  /** 進捗情報 */
  progress: MtSeedBootTimingSearchProgress | null;

  // --- 結果 ---
  /** 検索中の内部バッファ（UIには反映しない） */
  _pendingResults: MtSeedBootTimingSearchResult[];
  /** 検索結果配列（完了/停止時に一括反映） */
  results: MtSeedBootTimingSearchResult[];
  /** 結果フィルター条件 */
  resultFilters: MtSeedBootTimingResultFilters;

  // --- エラー ---
  /** 最終エラーメッセージ */
  errorMessage: string | null;

  // --- 完了情報 ---
  /** 最終実行時間 (ms) */
  lastElapsedMs: number | null;
  /** 完了情報 */
  lastCompletion: MtSeedBootTimingCompletion | null;
}

interface MtSeedBootTimingSearchActions {
  // --- パラメータ更新 ---
  updateDraftParams: (updates: Partial<MtSeedBootTimingSearchParams>) => void;
  updateDateRange: (updates: Partial<DateRange>) => void;
  updateTimeRange: (updates: Partial<DailyTimeRange>) => void;
  updateTargetSeeds: (seeds: number[]) => void;
  applyProfile: (profile: DeviceProfile) => void;

  // --- バリデーション ---
  validateDraft: () => boolean;

  // --- 検索制御 ---
  startSearch: () => Promise<void>;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;

  // --- 進捗更新（Worker Manager からの呼び出し用） ---
  _updateProgress: (progress: MtSeedBootTimingSearchProgress) => void;
  _addPendingResult: (result: MtSeedBootTimingSearchResult) => boolean;
  _onComplete: (completion: MtSeedBootTimingCompletion) => void;
  _onError: (error: string) => void;
  _onPaused: () => void;
  _onResumed: () => void;
  _onStopped: () => void;

  // --- 結果操作 ---
  updateResultFilters: (filters: Partial<MtSeedBootTimingResultFilters>) => void;
  getFilteredResults: () => MtSeedBootTimingSearchResult[];
  clearResults: () => void;

  // --- リセット ---
  reset: () => void;
}

export interface MtSeedBootTimingSearchStore
  extends MtSeedBootTimingSearchState,
    MtSeedBootTimingSearchActions {}

export const useMtSeedBootTimingSearchStore = create<MtSeedBootTimingSearchStore>(
  (set, get) => ({
    // === Initial State ===
    draftParams: createDefaultMtSeedBootTimingSearchParams(),
    params: null,
    validationErrors: [],
    status: 'idle',
    progress: null,
    _pendingResults: [],
    results: [],
    resultFilters: {},
    errorMessage: null,
    lastElapsedMs: null,
    lastCompletion: null,

    // === パラメータ更新 ===
    updateDraftParams: (updates) => {
      set((state) => ({
        draftParams: { ...state.draftParams, ...updates },
        validationErrors: [], // パラメータ変更時にエラーをクリア
      }));
    },

    updateDateRange: (updates) => {
      set((state) => ({
        draftParams: {
          ...state.draftParams,
          dateRange: { ...state.draftParams.dateRange, ...updates },
        },
      }));
    },

    updateTimeRange: (updates) => {
      set((state) => ({
        draftParams: {
          ...state.draftParams,
          timeRange: {
            hour: updates.hour ?? state.draftParams.timeRange.hour,
            minute: updates.minute ?? state.draftParams.timeRange.minute,
            second: updates.second ?? state.draftParams.timeRange.second,
          },
        },
      }));
    },

    updateTargetSeeds: (seeds) => {
      set((state) => ({
        draftParams: { ...state.draftParams, targetSeeds: seeds },
      }));
    },

    applyProfile: (profile) => {
      set((state) => ({
        draftParams: {
          ...state.draftParams,
          romVersion: profile.romVersion,
          romRegion: profile.romRegion,
          hardware: profile.hardware,
          macAddress: profile.macAddress,
          timer0Range: {
            min: profile.timer0Range.min,
            max: profile.timer0Range.max,
          },
          vcountRange: {
            min: profile.vcountRange.min,
            max: profile.vcountRange.max,
          },
        },
      }));
    },

    // === バリデーション ===
    validateDraft: () => {
      const { draftParams } = get();
      const errors = validateMtSeedBootTimingSearchParams(draftParams);
      set({
        validationErrors: errors,
        params: errors.length === 0 ? draftParams : null,
      });
      return errors.length === 0;
    },

    // === 検索制御 ===
    startSearch: async () => {
      const { validateDraft, draftParams } = get();

      if (!validateDraft()) {
        return;
      }

      set({
        status: 'starting',
        _pendingResults: [],
        results: [],
        progress: null,
        errorMessage: null,
        lastElapsedMs: null,
        lastCompletion: null,
        params: draftParams,
      });

      const manager = getSearchWorkerManager();

      const callbacks: MtSeedBootTimingSearchCallbacks = {
        onProgress: (progress) => {
          const progressPercent =
            progress.totalSteps > 0
              ? (progress.currentStep / progress.totalSteps) * 100
              : 0;
          get()._updateProgress({
            processedCombinations: progress.currentStep,
            totalCombinations: progress.totalSteps,
            foundCount: progress.matchesFound,
            progressPercent,
            elapsedMs: progress.elapsedTime,
            estimatedRemainingMs: progress.estimatedTimeRemaining,
          });
        },
        onResult: (result) => {
          const canContinue = get()._addPendingResult(result);
          if (!canContinue) {
            console.warn(
              `[MtSeedSearch] MAX_RESULTS (${MAX_RESULTS}) reached, stopping search`
            );
            get().stopSearch();
          }
        },
        onComplete: (_message) => {
          get()._onComplete({
            reason: 'completed',
            processedCombinations: get().progress?.processedCombinations ?? 0,
            totalCombinations: get().progress?.totalCombinations ?? 0,
            resultsCount: get()._pendingResults.length,
            elapsedMs: get().progress?.elapsedMs ?? 0,
          });
        },
        onError: (error) => {
          get()._onError(error);
        },
        onPaused: () => {
          get()._onPaused();
        },
        onResumed: () => {
          get()._onResumed();
        },
        onStopped: () => {
          get()._onStopped();
        },
      };

      try {
        set({ status: 'running' });
        manager.startMtSeedBootTimingSearch(draftParams, callbacks);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        set({ status: 'error', errorMessage: message });
      }
    },

    pauseSearch: () => {
      const manager = getSearchWorkerManager();
      manager.pauseSearch();
    },

    resumeSearch: () => {
      const manager = getSearchWorkerManager();
      manager.resumeSearch();
    },

    stopSearch: () => {
      set({ status: 'stopping' });
      const manager = getSearchWorkerManager();
      manager.stopSearch();
    },

    // === 進捗更新 ===
    _updateProgress: (progress) => {
      set({ progress });
    },

    _addPendingResult: (result) => {
      const pending = get()._pendingResults;
      if (pending.length >= MAX_RESULTS) {
        return false;
      }
      pending.push(result);
      set({ _pendingResults: pending });
      return pending.length < MAX_RESULTS;
    },

    _onComplete: (completion) => {
      const { _pendingResults, progress } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'completed',
        lastElapsedMs: progress?.elapsedMs ?? null,
        lastCompletion: completion,
      });
    },

    _onError: (error) => {
      const { _pendingResults } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'error',
        errorMessage: error,
      });
    },

    _onPaused: () => {
      set({ status: 'paused' });
    },

    _onResumed: () => {
      set({ status: 'running' });
    },

    _onStopped: () => {
      const { _pendingResults } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'idle',
      });
    },

    // === 結果操作 ===
    updateResultFilters: (filters) => {
      set((state) => ({
        resultFilters: { ...state.resultFilters, ...filters },
      }));
    },

    getFilteredResults: () => {
      const { results, resultFilters } = get();
      let filtered = results;

      if (resultFilters.timer0Filter) {
        const filterValue = parseInt(resultFilters.timer0Filter, 16);
        if (!isNaN(filterValue)) {
          filtered = filtered.filter((r) => r.boot.timer0 === filterValue);
        }
      }

      if (resultFilters.vcountFilter) {
        const filterValue = parseInt(resultFilters.vcountFilter, 16);
        if (!isNaN(filterValue)) {
          filtered = filtered.filter((r) => r.boot.vcount === filterValue);
        }
      }

      return filtered;
    },

    clearResults: () => {
      set({
        results: [],
        _pendingResults: [],
        lastCompletion: null,
        lastElapsedMs: null,
      });
    },

    // === リセット ===
    reset: () => {
      const manager = getSearchWorkerManager();
      manager.stopSearch();

      set({
        draftParams: createDefaultMtSeedBootTimingSearchParams(),
        params: null,
        validationErrors: [],
        status: 'idle',
        progress: null,
        _pendingResults: [],
        results: [],
        resultFilters: {},
        errorMessage: null,
        lastElapsedMs: null,
        lastCompletion: null,
      });
    },
  })
);
