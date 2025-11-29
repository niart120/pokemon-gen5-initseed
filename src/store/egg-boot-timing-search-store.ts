/**
 * egg-boot-timing-search-store.ts
 * Search(Egg)パネル向けのZustand Store
 * 仕様: spec/agent/pr_egg_boot_timing_search/STATE_MANAGEMENT.md
 */

import { create } from 'zustand';
import type {
  EggBootTimingSearchParams,
  EggBootTimingSearchResult,
  EggBootTimingCompletion,
  DateRange,
} from '@/types/egg-boot-timing-search';
import {
  createDefaultEggBootTimingSearchParams,
  validateEggBootTimingSearchParams,
} from '@/types/egg-boot-timing-search';
import type { EggGenerationConditions, ParentsIVs, IvSet, EggIndividualFilter } from '@/types/egg';
import { createDefaultEggFilter } from '@/types/egg';
import type { DeviceProfile } from '@/types/profile';
import type { DailyTimeRange } from '@/types/search';
import { EggBootTimingMultiWorkerManager } from '@/lib/egg';

/**
 * 検索実行状態
 */
export type EggBootTimingSearchStatus =
  | 'idle'       // 初期状態・完了後
  | 'starting'   // 検索開始中
  | 'running'    // 検索実行中
  | 'stopping'   // 停止処理中
  | 'completed'  // 完了
  | 'error';     // エラー

/**
 * 進捗情報
 */
export interface EggBootTimingSearchProgress {
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
}

/**
 * 結果フィルター条件
 */
export interface EggBootTimingResultFilters {
  // 色違いフィルター
  shinyOnly?: boolean;
  // 性格フィルター
  natures?: number[];
  // Timer0フィルター (hex文字列)
  timer0Filter?: string;
  // VCountフィルター (hex文字列)
  vcountFilter?: string;
}

const MAX_RESULTS = 1000;

interface EggBootTimingSearchState {
  // --- パラメータ ---
  /** UI入力用ドラフトパラメータ */
  draftParams: EggBootTimingSearchParams;
  /** バリデーション済みパラメータ */
  params: EggBootTimingSearchParams | null;
  /** バリデーションエラー */
  validationErrors: string[];
  
  // --- 実行状態 ---
  /** 検索状態 */
  status: EggBootTimingSearchStatus;
  /** Worker Manager (並列版) */
  workerManager: EggBootTimingMultiWorkerManager | null;
  
  // --- 進捗 ---
  /** 進捗情報 */
  progress: EggBootTimingSearchProgress | null;
  
  // --- 結果 ---
  /** 検索中の内部バッファ（UIには反映しない） */
  _pendingResults: EggBootTimingSearchResult[];
  /** 検索結果配列（完了/停止時に一括反映） */
  results: EggBootTimingSearchResult[];
  /** 結果フィルター条件 */
  resultFilters: EggBootTimingResultFilters;
  
  // --- エラー ---
  /** 最終エラーメッセージ */
  errorMessage: string | null;
  
  // --- 完了情報 ---
  /** 最終実行時間 (ms) */
  lastElapsedMs: number | null;
  /** 完了情報 */
  lastCompletion: EggBootTimingCompletion | null;
}

interface EggBootTimingSearchActions {
  // --- パラメータ更新 ---
  updateDraftParams: (updates: Partial<EggBootTimingSearchParams>) => void;
  updateDraftConditions: (updates: Partial<EggGenerationConditions>) => void;
  updateDraftParents: (updates: Partial<ParentsIVs>) => void;
  updateDraftParentsMale: (ivs: IvSet) => void;
  updateDraftParentsFemale: (ivs: IvSet) => void;
  updateDateRange: (updates: Partial<DateRange>) => void;
  updateTimeRange: (updates: Partial<DailyTimeRange>) => void;
  updateFilter: (updates: Partial<EggIndividualFilter>) => void;
  applyProfile: (profile: DeviceProfile) => void;
  
  // --- バリデーション ---
  validateDraft: () => boolean;
  
  // --- 検索制御 ---
  startSearch: () => Promise<void>;
  stopSearch: () => void;
  
  // --- 進捗更新（Worker Manager からの呼び出し用） ---
  _updateProgress: (progress: EggBootTimingSearchProgress) => void;
  _addPendingResult: (result: EggBootTimingSearchResult) => boolean;
  _onComplete: (completion: EggBootTimingCompletion) => void;
  _onError: (error: string) => void;
  _onStopped: () => void;
  
  // --- 結果操作 ---
  updateResultFilters: (filters: Partial<EggBootTimingResultFilters>) => void;
  getFilteredResults: () => EggBootTimingSearchResult[];
  clearResults: () => void;
  
  // --- リセット ---
  reset: () => void;
}

export interface EggBootTimingSearchStore
  extends EggBootTimingSearchState,
    EggBootTimingSearchActions {}

export const useEggBootTimingSearchStore = create<EggBootTimingSearchStore>(
  (set, get) => ({
    // === Initial State ===
    draftParams: createDefaultEggBootTimingSearchParams(),
    params: null,
    validationErrors: [],
    status: 'idle',
    workerManager: null,
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

    updateDraftParents: (updates) => {
      set((state) => ({
        draftParams: {
          ...state.draftParams,
          parents: { ...state.draftParams.parents, ...updates },
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
          timeRange: { ...state.draftParams.timeRange, ...updates },
        },
      }));
    },

    updateFilter: (updates) => {
      set((state) => {
        const currentFilter = state.draftParams.filter || createDefaultEggFilter();
        return {
          draftParams: {
            ...state.draftParams,
            filter: { ...currentFilter, ...updates },
          },
        };
      });
    },

    applyProfile: (profile) => {
      set((state) => ({
        draftParams: {
          ...state.draftParams,
          romVersion: profile.romVersion,
          romRegion: profile.romRegion,
          hardware: profile.hardware,
          macAddress: [...profile.macAddress] as readonly [
            number, number, number, number, number, number,
          ],
          // Timer0/VCountはProfileから取得
          timer0Range: { ...profile.timer0Range },
          vcountRange: { ...profile.vcountRange },
          // TID/SIDもプロファイルから
          conditions: {
            ...state.draftParams.conditions,
            tid: profile.tid,
            sid: profile.sid,
          },
        },
      }));
    },

    // === バリデーション ===
    validateDraft: () => {
      const { draftParams } = get();
      const errors = validateEggBootTimingSearchParams(draftParams);
      set({
        validationErrors: errors,
        params: errors.length === 0 ? draftParams : null,
      });
      return errors.length === 0;
    },

    // === 検索制御 ===
    startSearch: async () => {
      const { validateDraft, draftParams, workerManager: existingManager } = get();
      
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

      // Worker Manager初期化またはリユース (並列版)
      const manager = existingManager || new EggBootTimingMultiWorkerManager();

      try {
        set({ workerManager: manager, status: 'running' });
        
        await manager.startParallelSearch(draftParams, {
          onProgress: (aggregatedProgress) => {
            // 並列版の進捗を共通形式に変換
            const progressPercent = aggregatedProgress.totalSteps > 0
              ? (aggregatedProgress.totalCurrentStep / aggregatedProgress.totalSteps) * 100
              : 0;
            get()._updateProgress({
              processedCombinations: aggregatedProgress.totalCurrentStep,
              totalCombinations: aggregatedProgress.totalSteps,
              foundCount: aggregatedProgress.totalMatchesFound,
              progressPercent,
              elapsedMs: aggregatedProgress.totalElapsedTime,
            });
          },
          onResult: (result) => {
            const canContinue = get()._addPendingResult(result);
            if (!canContinue) {
              // MAX_RESULTS到達: 検索を停止
              console.warn(`[EggSearch] MAX_RESULTS (${MAX_RESULTS}) reached, stopping search`);
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
          onStopped: () => {
            get()._onStopped();
          },
        });
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        set({ status: 'error', errorMessage: message });
      }
    },

    stopSearch: () => {
      const { workerManager } = get();
      set({ status: 'stopping' });
      if (workerManager) {
        workerManager.terminateAll();
      }
    },

    // === 進捗更新 ===
    _updateProgress: (progress) => {
      set({ progress });
    },

    _addPendingResult: (result) => {
      // ミュータブルにpushして参照更新（O(n²)スプレッド問題回避）
      const pending = get()._pendingResults;
      if (pending.length >= MAX_RESULTS) {
        // 上限到達
        return false;
      }
      pending.push(result);
      // 配列自体は同じ参照だが、Zustandに変更を通知するためsetを呼ぶ
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
      
      // Timer0フィルター値をパース
      const timer0FilterValue = resultFilters.timer0Filter
        ? parseInt(resultFilters.timer0Filter, 16)
        : null;
      const hasTimer0Filter = timer0FilterValue !== null && !isNaN(timer0FilterValue);
      
      // VCountフィルター値をパース
      const vcountFilterValue = resultFilters.vcountFilter
        ? parseInt(resultFilters.vcountFilter, 16)
        : null;
      const hasVcountFilter = vcountFilterValue !== null && !isNaN(vcountFilterValue);
      
      return results.filter((result) => {
        // 色違いフィルター
        if (resultFilters.shinyOnly && result.egg.egg.shiny === 0) {
          return false;
        }
        // 性格フィルター
        if (resultFilters.natures && resultFilters.natures.length > 0) {
          if (!resultFilters.natures.includes(result.egg.egg.nature)) {
            return false;
          }
        }
        // Timer0フィルター
        if (hasTimer0Filter && result.boot.timer0 !== timer0FilterValue) {
          return false;
        }
        // VCountフィルター
        if (hasVcountFilter && result.boot.vcount !== vcountFilterValue) {
          return false;
        }
        return true;
      });
    },

    clearResults: () => {
      set({
        results: [],
        _pendingResults: [],
        progress: null,
        errorMessage: null,
        lastElapsedMs: null,
        lastCompletion: null,
      });
    },

    // === リセット ===
    reset: () => {
      const { workerManager } = get();
      if (workerManager) {
        workerManager.terminateAll();
      }
      set({
        draftParams: createDefaultEggBootTimingSearchParams(),
        params: null,
        validationErrors: [],
        status: 'idle',
        workerManager: null,
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
