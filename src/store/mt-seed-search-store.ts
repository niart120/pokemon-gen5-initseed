/**
 * mt-seed-search-store.ts
 * MT Seed 32bit全探索パネル向けのZustand Store
 */

import { create } from 'zustand';
import type {
  IvSearchFilter,
  StatRange,
  MtSeedMatch,
} from '@/types/mt-seed-search';
import {
  createMtSeedSearchManager,
  type MtSeedSearchManager,
  type AggregatedProgress,
  type SearchCompletion,
  type SearchError,
} from '@/lib/mt-seed-search/mt-seed-search-manager';

// === 型定義 ===

/**
 * 検索実行状態
 */
export type MtSeedSearchStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'paused'
  | 'completed'
  | 'error';

/**
 * 検索パラメータ（ドラフト）
 */
export interface MtSeedSearchDraftParams {
  /** MT消費数 */
  mtAdvances: number;
  /** 徘徊ポケモンモード（IV順序: HABCDS → HABDSC） */
  isRoamer: boolean;
  /** 個体値フィルター */
  filter: IvSearchFilter;
}

/**
 * 進捗情報
 */
export interface MtSeedSearchProgressInfo {
  processedCount: number;
  totalCount: number;
  matchesFound: number;
  progressPercent: number;
  elapsedMs: number;
  estimatedRemainingMs: number;
  mode: 'gpu' | 'cpu';
  activeWorkers: number;
}

/**
 * 完了情報
 */
export interface MtSeedSearchCompletionInfo {
  reason: 'finished' | 'stopped' | 'error';
  totalProcessed: number;
  totalMatches: number;
  elapsedMs: number;
  mode: 'gpu' | 'cpu';
}

// === デフォルト値 ===

export function createDefaultIvSearchFilter(): IvSearchFilter {
  return {
    ivRanges: [
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
    ],
    hiddenPowerType: undefined,
    hiddenPowerPower: undefined,
  };
}

export function createDefaultDraftParams(): MtSeedSearchDraftParams {
  return {
    mtAdvances: 0,
    isRoamer: false,
    filter: createDefaultIvSearchFilter(),
  };
}

// === Store定義 ===

interface MtSeedSearchState {
  // --- パラメータ ---
  draftParams: MtSeedSearchDraftParams;
  validationErrors: string[];

  // --- 実行状態 ---
  status: MtSeedSearchStatus;

  // --- 進捗 ---
  progress: MtSeedSearchProgressInfo | null;

  // --- 結果 ---
  results: MtSeedMatch[];

  // --- エラー ---
  errorMessage: string | null;

  // --- 完了情報 ---
  lastCompletion: MtSeedSearchCompletionInfo | null;
}

interface MtSeedSearchActions {
  // --- パラメータ更新 ---
  updateDraftParams: (updates: Partial<MtSeedSearchDraftParams>) => void;
  updateIvRange: (statIndex: number, minMax: 'min' | 'max', value: number) => void;
  updateFilter: (updates: Partial<IvSearchFilter>) => void;

  // --- バリデーション ---
  validateDraft: () => boolean;

  // --- 検索制御 ---
  startSearch: () => Promise<void>;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;

  // --- 結果操作 ---
  clearResults: () => void;

  // --- リセット ---
  reset: () => void;

  // --- Manager管理 ---
  _getManager: () => MtSeedSearchManager | null;
  _disposeManager: () => void;
}

export interface MtSeedSearchStore extends MtSeedSearchState, MtSeedSearchActions {}

// Manager インスタンス（Store外で管理）
let managerInstance: MtSeedSearchManager | null = null;

export const useMtSeedSearchStore = create<MtSeedSearchStore>((set, get) => ({
  // === Initial State ===
  draftParams: createDefaultDraftParams(),
  validationErrors: [],
  status: 'idle',
  progress: null,
  results: [],
  errorMessage: null,
  lastCompletion: null,

  // === パラメータ更新 ===
  updateDraftParams: (updates) => {
    set((state) => ({
      draftParams: { ...state.draftParams, ...updates },
      validationErrors: [],
    }));
  },

  updateIvRange: (statIndex, minMax, value) => {
    set((state) => {
      const newRanges = [...state.draftParams.filter.ivRanges] as [
        StatRange, StatRange, StatRange, StatRange, StatRange, StatRange
      ];
      newRanges[statIndex] = {
        ...newRanges[statIndex],
        [minMax]: value,
      };
      return {
        draftParams: {
          ...state.draftParams,
          filter: {
            ...state.draftParams.filter,
            ivRanges: newRanges,
          },
        },
        validationErrors: [],
      };
    });
  },

  updateFilter: (updates) => {
    set((state) => ({
      draftParams: {
        ...state.draftParams,
        filter: { ...state.draftParams.filter, ...updates },
      },
      validationErrors: [],
    }));
  },

  // === バリデーション ===
  validateDraft: () => {
    const { draftParams } = get();
    const errors: string[] = [];

    // MT消費数の検証
    if (draftParams.mtAdvances < 0 || !Number.isInteger(draftParams.mtAdvances)) {
      errors.push('MT消費数は0以上の整数を指定してください');
    }

    // IV範囲の検証
    for (let i = 0; i < 6; i++) {
      const range = draftParams.filter.ivRanges[i];
      if (range.min < 0 || range.min > 31) {
        errors.push(`IV${i + 1}のmin値は0-31の範囲で指定してください`);
      }
      if (range.max < 0 || range.max > 31) {
        errors.push(`IV${i + 1}のmax値は0-31の範囲で指定してください`);
      }
      if (range.min > range.max) {
        errors.push(`IV${i + 1}のmin値はmax値以下にしてください`);
      }
    }

    // めざパ威力の検証
    if (draftParams.filter.hiddenPowerPower !== undefined) {
      if (draftParams.filter.hiddenPowerPower < 30 || draftParams.filter.hiddenPowerPower > 70) {
        errors.push('めざパ威力は30-70の範囲で指定してください');
      }
    }

    set({ validationErrors: errors });
    return errors.length === 0;
  },

  // === 検索制御 ===
  startSearch: async () => {
    const { validateDraft, draftParams, _disposeManager } = get();

    if (!validateDraft()) {
      return;
    }

    // 既存のManagerを破棄
    _disposeManager();

    set({
      status: 'starting',
      results: [],
      progress: null,
      errorMessage: null,
      lastCompletion: null,
    });

    // MT消費数を取得
    const mtAdvances = draftParams.mtAdvances;

    // コールバック付きでManagerを作成
    const manager = createMtSeedSearchManager({
      mode: 'auto',
      onProgress: (progress: AggregatedProgress) => {
        set({
          progress: {
            processedCount: progress.processedCount,
            totalCount: progress.totalCount,
            matchesFound: progress.matchesFound,
            progressPercent: progress.progressPercent,
            elapsedMs: progress.elapsedMs,
            estimatedRemainingMs: progress.estimatedRemainingMs,
            mode: progress.mode,
            activeWorkers: progress.activeWorkers,
          },
        });
      },
      onResult: (result) => {
        set((state) => ({
          results: [...state.results, ...result.matches],
        }));
      },
      onComplete: (completion: SearchCompletion) => {
        set({
          status: completion.reason === 'error' ? 'error' : 'completed',
          lastCompletion: {
            reason: completion.reason,
            totalProcessed: completion.totalProcessed,
            totalMatches: completion.totalMatches,
            elapsedMs: completion.elapsedMs,
            mode: completion.mode,
          },
          // 最終結果を反映
          results: completion.matches,
        });
      },
      onError: (error: SearchError) => {
        set({
          status: 'error',
          errorMessage: error.message,
        });
      },
    });

    managerInstance = manager;

    try {
      set({ status: 'running' });
      await manager.start({
        mtAdvances,
        filter: draftParams.filter,
        isRoamer: draftParams.isRoamer,
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({
        status: 'error',
        errorMessage: message,
      });
    }
  },

  pauseSearch: () => {
    if (managerInstance) {
      managerInstance.pause();
      set({ status: 'paused' });
    }
  },

  resumeSearch: () => {
    if (managerInstance) {
      managerInstance.resume();
      set({ status: 'running' });
    }
  },

  stopSearch: () => {
    set({ status: 'stopping' });
    if (managerInstance) {
      managerInstance.stop();
    }
  },

  // === 結果操作 ===
  clearResults: () => {
    set({
      results: [],
      lastCompletion: null,
      progress: null,
    });
  },

  // === リセット ===
  reset: () => {
    const { _disposeManager } = get();
    _disposeManager();

    set({
      draftParams: createDefaultDraftParams(),
      validationErrors: [],
      status: 'idle',
      progress: null,
      results: [],
      errorMessage: null,
      lastCompletion: null,
    });
  },

  // === Manager管理 ===
  _getManager: () => managerInstance,

  _disposeManager: () => {
    if (managerInstance) {
      managerInstance.dispose();
      managerInstance = null;
    }
  },
}));
