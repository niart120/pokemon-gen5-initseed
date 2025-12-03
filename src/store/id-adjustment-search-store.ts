/**
 * id-adjustment-search-store.ts
 * ID調整検索パネル向けのZustand Store
 * MtSeedBootTimingSearchStoreのパターンを流用
 */

import { create } from 'zustand';
import type {
  IdAdjustmentSearchParams,
  IdAdjustmentSearchResult,
} from '@/types/id-adjustment-search';
import {
  createDefaultIdAdjustmentSearchParams,
  validateIdAdjustmentSearchParams,
} from '@/types/id-adjustment-search';
import type { DeviceProfile } from '@/types/profile';
import type { DailyTimeRange, DateRange } from '@/types/search';
import {
  IdAdjustmentMultiWorkerManager,
  type AggregatedIdAdjustmentProgress,
} from '@/lib/id-adjustment';
import { DomainShinyType } from '@/types/domain';

/**
 * 検索実行状態
 */
export type IdAdjustmentSearchStatus =
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
export interface IdAdjustmentSearchProgress {
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
  /** アクティブなWorker数 */
  activeWorkers: number;
}

/**
 * 結果フィルター条件
 */
export interface IdAdjustmentResultFilters {
  /** Timer0フィルター (hex文字列) */
  timer0Filter?: string;
  /** VCountフィルター (hex文字列) */
  vcountFilter?: string;
  /** 色違いのみ表示 */
  shinyOnly?: boolean;
}

/**
 * 完了情報
 */
export interface IdAdjustmentCompletion {
  reason: 'completed' | 'stopped' | 'max_results' | 'error';
  processedCombinations: number;
  totalCombinations: number;
  resultsCount: number;
  elapsedMs: number;
}

const MAX_RESULTS = 1000;

interface IdAdjustmentSearchState {
  // --- パラメータ ---
  /** UI入力用ドラフトパラメータ */
  draftParams: IdAdjustmentSearchParams;
  /** バリデーション済みパラメータ */
  params: IdAdjustmentSearchParams | null;
  /** バリデーションエラー */
  validationErrors: string[];

  // --- 実行状態 ---
  /** 検索状態 */
  status: IdAdjustmentSearchStatus;

  // --- 進捗 ---
  /** 進捗情報 */
  progress: IdAdjustmentSearchProgress | null;

  // --- 結果 ---
  /** 検索中の内部バッファ（UIには反映しない） */
  _pendingResults: IdAdjustmentSearchResult[];
  /** 検索結果配列（完了/停止時に一括反映） */
  results: IdAdjustmentSearchResult[];
  /** 結果フィルター条件 */
  resultFilters: IdAdjustmentResultFilters;

  // --- エラー ---
  /** 最終エラーメッセージ */
  errorMessage: string | null;

  // --- 完了情報 ---
  /** 最終実行時間 (ms) */
  lastElapsedMs: number | null;
  /** 完了情報 */
  lastCompletion: IdAdjustmentCompletion | null;

  // --- Worker Manager ---
  /** WorkerManager インスタンス */
  _workerManager: IdAdjustmentMultiWorkerManager | null;
}

interface IdAdjustmentSearchActions {
  // --- パラメータ更新 ---
  updateDraftParams: (updates: Partial<IdAdjustmentSearchParams>) => void;
  updateDateRange: (updates: Partial<DateRange>) => void;
  updateTimeRange: (updates: Partial<DailyTimeRange>) => void;
  applyProfile: (profile: DeviceProfile) => void;

  // --- バリデーション ---
  validateDraft: () => boolean;

  // --- 検索制御 ---
  startSearch: () => Promise<void>;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;

  // --- 進捗更新（内部用） ---
  _updateProgress: (progress: AggregatedIdAdjustmentProgress) => void;
  _addPendingResult: (result: IdAdjustmentSearchResult) => boolean;
  _onComplete: (completion: IdAdjustmentCompletion) => void;
  _onError: (error: string) => void;
  _onPaused: () => void;
  _onResumed: () => void;
  _onStopped: () => void;

  // --- 結果操作 ---
  updateResultFilters: (filters: Partial<IdAdjustmentResultFilters>) => void;
  getFilteredResults: () => IdAdjustmentSearchResult[];
  clearResults: () => void;

  // --- リセット ---
  reset: () => void;
}

export interface IdAdjustmentSearchStore
  extends IdAdjustmentSearchState,
    IdAdjustmentSearchActions {}

export const useIdAdjustmentSearchStore = create<IdAdjustmentSearchStore>(
  (set, get) => ({
    // === Initial State ===
    draftParams: createDefaultIdAdjustmentSearchParams(),
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
    _workerManager: null,

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
      const errors = validateIdAdjustmentSearchParams(draftParams);
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

      const manager = new IdAdjustmentMultiWorkerManager();
      set({ _workerManager: manager });

      try {
        set({ status: 'running' });

        await manager.startParallelSearch(draftParams, {
          onProgress: (progress) => {
            get()._updateProgress(progress);
          },
          onResult: (result) => {
            const canContinue = get()._addPendingResult(result);
            if (!canContinue) {
              console.warn(
                `[IdAdjustmentSearch] MAX_RESULTS (${MAX_RESULTS}) reached, stopping search`
              );
              get().stopSearch();
            }
          },
          onComplete: () => {
            const currentProgress = get().progress;
            get()._onComplete({
              reason: 'completed',
              processedCombinations: currentProgress?.processedCombinations ?? 0,
              totalCombinations: currentProgress?.totalCombinations ?? 0,
              resultsCount: get()._pendingResults.length,
              elapsedMs: currentProgress?.elapsedMs ?? 0,
            });
          },
          onError: (error) => {
            get()._onError(error);
          },
        });
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        set({ status: 'error', errorMessage: message });
      }
    },

    pauseSearch: () => {
      const manager = get()._workerManager;
      if (manager) {
        manager.pauseAll();
        get()._onPaused();
      }
    },

    resumeSearch: () => {
      const manager = get()._workerManager;
      if (manager) {
        manager.resumeAll();
        get()._onResumed();
      }
    },

    stopSearch: () => {
      set({ status: 'stopping' });
      const manager = get()._workerManager;
      if (manager) {
        manager.terminateAll();
        get()._onStopped();
      }
    },

    // === 進捗更新 ===
    _updateProgress: (progress) => {
      const progressPercent =
        progress.totalSteps > 0
          ? (progress.totalCurrentStep / progress.totalSteps) * 100
          : 0;

      set({
        progress: {
          processedCombinations: progress.totalCurrentStep,
          totalCombinations: progress.totalSteps,
          foundCount: progress.totalMatchesFound,
          progressPercent,
          elapsedMs: progress.totalElapsedTime,
          estimatedRemainingMs: progress.totalEstimatedTimeRemaining,
          activeWorkers: progress.activeWorkers,
        },
      });
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
        _workerManager: null,
      });
    },

    _onError: (error) => {
      const { _pendingResults } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'error',
        errorMessage: error,
        _workerManager: null,
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
        _workerManager: null,
      });
    },

    // === 結果操作 ===
    updateResultFilters: (filters) => {
      set((state) => ({
        resultFilters: { ...state.resultFilters, ...filters },
      }));
    },

    getFilteredResults: () => {
      const { results, _pendingResults, status, resultFilters } = get();
      
      // 検索中は _pendingResults を、完了後は results を使用
      const baseResults = (status === 'running' || status === 'paused' || status === 'starting')
        ? _pendingResults
        : results;
      
      let filtered = baseResults;

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

      if (resultFilters.shinyOnly) {
        filtered = filtered.filter(
          (r) => r.shinyType !== DomainShinyType.Normal
        );
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
      const manager = get()._workerManager;
      if (manager) {
        manager.terminateAll();
      }

      set({
        draftParams: createDefaultIdAdjustmentSearchParams(),
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
        _workerManager: null,
      });
    },
  })
);
