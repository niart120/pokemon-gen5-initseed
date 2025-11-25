# Zustand 状態管理設計

## 1. 概要

Search(Egg) パネル向けの Zustand Store 設計を定義する。既存の `app-store.ts` および `egg-store.ts` のパターンを踏襲し、孵化乱数起動時間検索専用の状態管理を実装する。

## 2. Store 設計方針

### 2.1 既存パターン分析

#### app-store.ts (初期Seed検索)

```typescript
// 状態カテゴリ
- 検索条件: searchConditions
- ターゲットSeeds: targetSeeds
- 検索結果: searchResults
- 進捗: searchProgress, parallelProgress
- 並列設定: parallelSearchSettings
- UI状態: activeTab, wakeLockEnabled
- プロファイル: profiles, activeProfileId

// 永続化
- persist() ミドルウェア使用
- partialize で選択的永続化
- merge で復元時の正規化
```

#### egg-store.ts (タマゴ列挙)

```typescript
// 状態カテゴリ
- パラメータ: draftParams, params
- 実行状態: status, workerManager
- Boot-Timing状態: derivedSeedRunState
- フィルター: bootTimingFilters
- 結果: results, lastCompletion, errorMessage

// 特徴
- EggWorkerManager をインスタンスとして保持
- Boot-Timing モードと LCG モードの分岐
- 永続化なし（セッション内のみ）
```

### 2.2 Search(Egg) パネル向け設計

**方針**: `egg-store.ts` のパターンをベースに、`EggBootTimingMultiWorkerManager` との連携を追加

## 3. Store 構造定義

### 3.1 型定義

```typescript
// src/store/egg-boot-timing-search-store.ts

import { create } from 'zustand';
import type {
  EggBootTimingSearchParams,
  EggBootTimingSearchResult,
} from '@/types/egg-boot-timing-search';
import type { AggregatedEggBootTimingProgress } from '@/lib/egg/boot-timing-egg-multi-worker-manager';

/**
 * 検索実行状態
 */
export type EggBootTimingSearchStatus =
  | 'idle'       // 初期状態・完了後
  | 'starting'   // 検索開始中
  | 'running'    // 検索実行中
  | 'paused'     // 一時停止中
  | 'stopping'   // 停止処理中
  | 'completed'  // 完了
  | 'error';     // エラー

/**
 * Store State
 */
interface EggBootTimingSearchState {
  // --- パラメータ ---
  /** UI入力用ドラフトパラメータ */
  draftParams: EggBootTimingSearchParams;
  /** バリデーション済みパラメータ (null = 未検証/エラー) */
  params: EggBootTimingSearchParams | null;
  /** バリデーションエラー */
  validationErrors: string[];
  
  // --- 実行状態 ---
  /** 検索状態 */
  status: EggBootTimingSearchStatus;
  /** WorkerManager インスタンス */
  workerManager: EggBootTimingMultiWorkerManager | null;
  
  // --- 並列検索設定 ---
  /** 最大Worker数 */
  maxWorkers: number;
  
  // --- 進捗 ---
  /** 集約進捗 */
  progress: AggregatedEggBootTimingProgress | null;
  
  // --- 結果 ---
  /** 検索結果配列 */
  results: EggBootTimingSearchResult[];
  /** フィルター済み結果のキャッシュ (遅延計算) */
  filteredResults: EggBootTimingSearchResult[];
  /** 結果フィルター条件 */
  resultFilters: EggBootTimingResultFilters;
  
  // --- エラー ---
  /** 最終エラーメッセージ */
  errorMessage: string | null;
  
  // --- 完了情報 ---
  /** 最終実行時間 (ms) */
  lastElapsedMs: number | null;
}

/**
 * 結果フィルター条件
 */
interface EggBootTimingResultFilters {
  // 個体値フィルター
  ivRanges?: {
    min: [number, number, number, number, number, number];
    max: [number, number, number, number, number, number];
  };
  // 性格フィルター
  natures?: number[];
  // 色違いフィルター
  shinyOnly?: boolean;
  // 起動時間範囲
  bootTimeRange?: { start: Date; end: Date };
}
```

### 3.2 アクション定義

```typescript
/**
 * Store Actions
 */
interface EggBootTimingSearchActions {
  // --- パラメータ更新 ---
  /** ドラフトパラメータ更新 */
  updateDraftParams: (updates: Partial<EggBootTimingSearchParams>) => void;
  /** 条件部分更新 */
  updateDraftConditions: (updates: Partial<EggGenerationConditions>) => void;
  /** 親個体値更新 */
  updateDraftParents: (updates: Partial<ParentsIVs>) => void;
  /** Boot-Timing パラメータ更新 */
  updateBootTimingParams: (updates: Partial<BootTimingParams>) => void;
  /** プロファイル適用 */
  applyProfile: (profile: DeviceProfile) => void;
  
  // --- バリデーション ---
  /** ドラフト検証 */
  validateDraft: () => boolean;
  
  // --- 検索制御 ---
  /** 検索開始 */
  startSearch: () => Promise<void>;
  /** 検索停止 */
  stopSearch: () => void;
  /** 一時停止 */
  pauseSearch: () => void;
  /** 再開 */
  resumeSearch: () => void;
  
  // --- 並列設定 ---
  /** Worker数設定 */
  setMaxWorkers: (count: number) => void;
  
  // --- 結果操作 ---
  /** 結果クリア */
  clearResults: () => void;
  /** フィルター更新 */
  updateResultFilters: (filters: Partial<EggBootTimingResultFilters>) => void;
  /** フィルター済み結果取得 */
  getFilteredResults: () => EggBootTimingSearchResult[];
  
  // --- リセット ---
  /** Store リセット */
  reset: () => void;
}
```

### 3.3 完全な Store 定義

```typescript
// src/store/egg-boot-timing-search-store.ts

import { create } from 'zustand';
import {
  type EggBootTimingSearchParams,
  type EggBootTimingSearchResult,
  createDefaultEggBootTimingSearchParams,
  validateEggBootTimingSearchParams,
} from '@/types/egg-boot-timing-search';
import {
  EggBootTimingMultiWorkerManager,
  type AggregatedEggBootTimingProgress,
  type EggBootTimingMultiWorkerCallbacks,
} from '@/lib/egg/boot-timing-egg-multi-worker-manager';
import type { DeviceProfile } from '@/types/profile';

const MAX_RESULTS = 10000;

interface EggBootTimingSearchStore
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
    maxWorkers: navigator.hardwareConcurrency || 4,
    progress: null,
    results: [],
    filteredResults: [],
    resultFilters: {},
    errorMessage: null,
    lastElapsedMs: null,

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

    updateBootTimingParams: (updates) => {
      set((state) => ({
        draftParams: {
          ...state.draftParams,
          ...updates,
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
          macAddress: [...profile.macAddress] as readonly [
            number,
            number,
            number,
            number,
            number,
            number,
          ],
          timer0Range: { ...profile.timer0Range },
          vcountRange: { ...profile.vcountRange },
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
      const { draftParams, workerManager: existingManager, maxWorkers } = get();

      // バリデーション
      if (!get().validateDraft()) {
        return;
      }

      // WorkerManager 初期化
      const manager = existingManager || new EggBootTimingMultiWorkerManager(maxWorkers);

      set({
        workerManager: manager,
        status: 'starting',
        results: [],
        filteredResults: [],
        progress: null,
        errorMessage: null,
        lastElapsedMs: null,
      });

      const callbacks: EggBootTimingMultiWorkerCallbacks = {
        onProgress: (progress) => {
          set({ progress });
        },
        onResult: (result) => {
          set((state) => {
            const newResults = [...state.results, result];
            // 上限を超えた場合は古い結果を削除
            if (newResults.length > MAX_RESULTS) {
              return { results: newResults.slice(-MAX_RESULTS) };
            }
            return { results: newResults };
          });
        },
        onComplete: (message) => {
          console.log('Search completed:', message);
          const { progress } = get();
          set({
            status: 'completed',
            lastElapsedMs: progress?.totalElapsedTime ?? null,
          });
        },
        onError: (error) => {
          console.error('Search error:', error);
          set({ status: 'error', errorMessage: error });
        },
        onPaused: () => {
          set({ status: 'paused' });
        },
        onResumed: () => {
          set({ status: 'running' });
        },
        onStopped: () => {
          set({ status: 'idle' });
        },
      };

      try {
        await manager.startParallelSearch(draftParams, callbacks);
        set({ status: 'running' });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set({ status: 'error', errorMessage: message });
      }
    },

    stopSearch: () => {
      const { workerManager } = get();
      if (workerManager) {
        set({ status: 'stopping' });
        workerManager.terminateAll();
      }
    },

    pauseSearch: () => {
      const { workerManager } = get();
      if (workerManager) {
        workerManager.pauseAll();
      }
    },

    resumeSearch: () => {
      const { workerManager } = get();
      if (workerManager) {
        workerManager.resumeAll();
      }
    },

    // === 並列設定 ===
    setMaxWorkers: (count) => {
      const { status, workerManager } = get();
      if (status === 'running' || status === 'paused') {
        console.warn('Cannot change worker count during active search');
        return;
      }
      set({ maxWorkers: count });
      if (workerManager) {
        workerManager.setMaxWorkers(count);
      }
    },

    // === 結果操作 ===
    clearResults: () => {
      set({
        results: [],
        filteredResults: [],
        progress: null,
        errorMessage: null,
        lastElapsedMs: null,
      });
    },

    updateResultFilters: (filters) => {
      set((state) => ({
        resultFilters: { ...state.resultFilters, ...filters },
      }));
    },

    getFilteredResults: () => {
      const { results, resultFilters } = get();
      // フィルター適用ロジック (簡略化)
      return results.filter((result) => {
        if (resultFilters.shinyOnly && result.egg.egg.shiny === 0) {
          return false;
        }
        // その他のフィルター条件...
        return true;
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
        results: [],
        filteredResults: [],
        resultFilters: {},
        errorMessage: null,
        lastElapsedMs: null,
      });
    },
  })
);
```

## 4. プロファイル連携

### 4.1 app-store との連携

既存の `app-store.ts` の `activeProfileId` を監視し、プロファイル変更時に Search(Egg) Store に反映する。

```typescript
// src/hooks/use-sync-profile-to-egg-boot-timing.ts

import { useEffect } from 'react';
import { useAppStore } from '@/store/app-store';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';

export function useSyncProfileToEggBootTiming() {
  const activeProfileId = useAppStore((s) => s.activeProfileId);
  const profiles = useAppStore((s) => s.profiles);
  const applyProfile = useEggBootTimingSearchStore((s) => s.applyProfile);

  useEffect(() => {
    if (!activeProfileId) return;
    const profile = profiles.find((p) => p.id === activeProfileId);
    if (profile) {
      applyProfile(profile);
    }
  }, [activeProfileId, profiles, applyProfile]);
}
```

### 4.2 パネルでの使用

```tsx
// src/components/egg-boot-timing-search/EggBootTimingSearchPanel.tsx

export const EggBootTimingSearchPanel: React.FC = () => {
  // プロファイル同期フック
  useSyncProfileToEggBootTiming();

  // Store から状態取得
  const {
    draftParams,
    status,
    progress,
    results,
    validationErrors,
    startSearch,
    stopSearch,
    updateDraftParams,
  } = useEggBootTimingSearchStore();

  // ... コンポーネント実装
};
```

## 5. 永続化設計

### 5.1 永続化方針

**Search(Egg) Store では永続化しない**

理由:
1. 検索パラメータはプロファイルから復元可能
2. 検索結果はセッション内のみ有効（起動時間に依存）
3. 実行状態の永続化は不要

### 5.2 プロファイルベースの復元

ページリロード時は:
1. `app-store` から `activeProfileId` を復元
2. プロファイルの設定を Search(Egg) Store に適用
3. デフォルト値 + プロファイル値で初期化

## 6. Selector パターン

### 6.1 派生状態の計算

```typescript
// src/store/selectors/egg-boot-timing-search-selectors.ts

import { useEggBootTimingSearchStore } from '../egg-boot-timing-search-store';

/**
 * 検索可能かどうか
 */
export function useCanStartSearch(): boolean {
  const status = useEggBootTimingSearchStore((s) => s.status);
  const validationErrors = useEggBootTimingSearchStore((s) => s.validationErrors);
  
  return (
    (status === 'idle' || status === 'completed' || status === 'error') &&
    validationErrors.length === 0
  );
}

/**
 * 進捗パーセント
 */
export function useProgressPercent(): number {
  const progress = useEggBootTimingSearchStore((s) => s.progress);
  if (!progress || progress.totalSteps === 0) return 0;
  return (progress.totalCurrentStep / progress.totalSteps) * 100;
}

/**
 * 残り時間表示文字列
 */
export function useEstimatedTimeRemaining(): string {
  const progress = useEggBootTimingSearchStore((s) => s.progress);
  if (!progress) return '--';
  
  const seconds = Math.ceil(progress.totalEstimatedTimeRemaining / 1000);
  if (seconds < 60) return `${seconds}s`;
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}
```

## 7. 既存 Store との比較

| 項目 | app-store | egg-store | egg-boot-timing-search-store |
|------|-----------|-----------|------------------------------|
| 永続化 | あり | なし | なし |
| プロファイル連携 | 直接保持 | 手動適用 | フック経由で同期 |
| Worker管理 | 外部関数 | インスタンス保持 | インスタンス保持 |
| 進捗管理 | parallelProgress | derivedSeedRunState | progress (集約) |
| 結果上限 | なし | 10000件 | 10000件 |

## 8. 参考ドキュメント

- `src/store/app-store.ts` - メインアプリStore
- `src/store/egg-store.ts` - タマゴ列挙Store
- `src/lib/egg/boot-timing-egg-multi-worker-manager.ts` - 並列Worker管理
- `/spec/agent/pr_egg_boot_timing_search/PROCESSING_FLOW.md` - 処理フロー
