# UI → WorkerManager 処理フロー設計

## 1. 概要

Search(Egg) パネルにおける UI からバックエンドまでの処理フローを定義する。既存の `SearchControlCard` および `EggRunCard` のパターンを踏襲し、`EggBootTimingMultiWorkerManager` を通じて並列検索を実行する。

## 2. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              React UI Layer                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │               EggBootTimingSearchPanel (新規)                          │ │
│  │  ┌──────────────┬──────────────────┬───────────────┬────────────────┐ │ │
│  │  │ ProfileCard  │ BootTimingParams │ EggParamsCard │ EggFilterCard  │ │ │
│  │  │  (既存流用)  │   (新規)         │  (既存流用)   │  (既存流用)    │ │ │
│  │  └──────────────┴──────────────────┴───────────────┴────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │  │              EggBootTimingRunCard (新規)                           │ │ │
│  │  │  - 検索開始/停止ボタン                                            │ │ │
│  │  │  - 進捗表示                                                        │ │ │
│  │  │  - Worker数設定                                                    │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │  │              EggBootTimingResultsCard (新規)                       │ │ │
│  │  │  - 結果テーブル                                                    │ │ │
│  │  │  - フィルター                                                      │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    │ (1) Zustand Store                      │
│                                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              useEggBootTimingSearchStore (新規)                        │ │
│  │  - draftParams: EggBootTimingSearchParams                             │ │
│  │  - status: 'idle' | 'running' | 'completed' | ...                     │ │
│  │  - results: EggBootTimingSearchResult[]                               │ │
│  │  - progress: AggregatedEggBootTimingProgress                          │ │
│  │  - actions: startSearch, stopSearch, updateParams, ...                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ (2) WorkerManager 呼び出し
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EggBootTimingMultiWorkerManager                        │
│  - チャンク分割 (日時範囲を Worker 数で分割)                                │
│  - Worker プール管理                                                        │
│  - 進捗集約                                                                 │
│  - 一時停止/再開                                                            │
│  - エラーハンドリング                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  Worker 0    │ │  Worker 1    │ │  Worker N    │
           │  (Chunk 0)   │ │  (Chunk 1)   │ │  (Chunk N)   │
           └──────────────┘ └──────────────┘ └──────────────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         egg-boot-timing-worker.ts                           │
│  - WASM 初期化                                                              │
│  - EggBootTimingSearcher 構築                                               │
│  - search_eggs_integrated_simd 呼び出し                                     │
│  - 結果変換・送信                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WASM (Rust)                                      │
│  - EggBootTimingSearcher                                                    │
│  - SHA-1 SIMD 計算                                                          │
│  - EggSeedEnumerator                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 処理シーケンス

### 3.1 検索開始フロー

```
User Click [Generate]
        │
        ▼
EggBootTimingRunCard.handleStart()
        │
        │  (1) パラメータ検証
        ▼
useEggBootTimingSearchStore.validateDraft()
        │
        │  エラーあり → エラー表示して終了
        │  エラーなし → 続行
        ▼
useEggBootTimingSearchStore.startSearch()
        │
        │  (2) Store 状態更新
        │      - status: 'idle' → 'starting'
        │      - results: []
        │      - progress: null
        ▼
EggBootTimingMultiWorkerManager.startParallelSearch(params, callbacks)
        │
        │  (3) チャンク分割
        │      calculateEggBootTimingChunks(params, maxWorkers)
        ▼
        ├─── Worker 0 初期化
        │       └── egg-boot-timing-worker.ts
        │               └── WASM init()
        │               └── EggBootTimingSearcher.new(...)
        │               └── search_eggs_integrated_simd(...)
        ├─── Worker 1 初期化
        │       └── (同上)
        └─── Worker N 初期化
                └── (同上)
        │
        │  (4) Store 状態更新
        │      - status: 'starting' → 'running'
        ▼
[検索実行中]
        │
        │  (5) Worker からのメッセージ処理
        │      - PROGRESS: 進捗更新
        │      - RESULTS: 結果追加
        │      - COMPLETE: 完了処理
        │      - ERROR: エラー処理
        ▼
callbacks.onProgress(aggregatedProgress)
callbacks.onResult(result)
        │
        │  (6) Store 状態更新
        │      - progress: AggregatedEggBootTimingProgress
        │      - results: [..., newResult]
        ▼
[全Worker完了]
        │
callbacks.onComplete(message)
        │
        │  (7) Store 状態更新
        │      - status: 'running' → 'completed'
        ▼
[検索完了]
```

### 3.2 検索停止フロー

```
User Click [Stop]
        │
        ▼
EggBootTimingRunCard.handleStop()
        │
        ▼
useEggBootTimingSearchStore.stopSearch()
        │
        │  (1) Store 状態更新
        │      - status: 'running' → 'stopping'
        ▼
EggBootTimingMultiWorkerManager.terminateAll()
        │
        │  (2) 全 Worker 停止
        │      - Worker.terminate()
        │      - タイマー停止
        ▼
callbacks.onStopped()
        │
        │  (3) Store 状態更新
        │      - status: 'stopping' → 'idle'
        ▼
[停止完了]
```

## 4. コールバック設計

### 4.1 EggBootTimingMultiWorkerCallbacks

既存の `MultiWorkerSearchCallbacks` パターンを踏襲:

```typescript
export interface EggBootTimingMultiWorkerCallbacks {
  /** 集約進捗更新 */
  onProgress: (progress: AggregatedEggBootTimingProgress) => void;
  
  /** 結果1件通知 */
  onResult: (result: EggBootTimingSearchResult) => void;
  
  /** 全Worker完了 */
  onComplete: (message: string) => void;
  
  /** エラー発生 */
  onError: (error: string) => void;
  
  /** 一時停止完了 (オプション) */
  onPaused?: () => void;
  
  /** 再開完了 (オプション) */
  onResumed?: () => void;
  
  /** 停止完了 (オプション) */
  onStopped?: () => void;
}
```

### 4.2 Store でのコールバック登録

```typescript
// store/egg-boot-timing-search-store.ts

const MAX_RESULTS = 1000;

startSearch: async () => {
  const { draftParams, workerManager } = get();
  
  set({ status: 'starting', _pendingResults: [], results: [], progress: null });
  
  const callbacks: EggBootTimingMultiWorkerCallbacks = {
    onProgress: (progress) => {
      set({ progress });
    },
    onResult: (result) => {
      // 内部バッファに追加（UIには反映しない）
      const pendingCount = get()._pendingResults.length;
      if (pendingCount >= MAX_RESULTS) {
        // 上限到達時は検索を停止
        workerManager.terminateAll();
        return;
      }
      set((state) => ({
        _pendingResults: [...state._pendingResults, result],
      }));
    },
    onComplete: (message) => {
      // 完了時に一括でUIに反映
      console.log('Search completed:', message);
      const { _pendingResults } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'completed',
      });
    },
    onError: (error) => {
      console.error('Search error:', error);
      const { _pendingResults } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'error',
        errorMessage: error,
      });
    },
    onStopped: () => {
      // 停止時に一括でUIに反映
      const { _pendingResults } = get();
      set({
        results: _pendingResults,
        _pendingResults: [],
        status: 'idle',
      });
    },
  };
  
  try {
    await workerManager.startParallelSearch(draftParams, callbacks);
    set({ status: 'running' });
  } catch (error) {
    set({ status: 'error', errorMessage: String(error) });
  }
},
```

## 5. Worker 通信プロトコル

### 5.1 リクエスト型

```typescript
export type EggBootTimingWorkerRequest =
  | { type: 'START_SEARCH'; params: EggBootTimingSearchParams; requestId?: string }
  | { type: 'STOP'; requestId?: string };
```

### 5.2 レスポンス型

```typescript
export type EggBootTimingWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: EggBootTimingProgress }
  | { type: 'RESULTS'; payload: EggBootTimingResultsPayload }
  | { type: 'COMPLETE'; payload: EggBootTimingCompletion }
  | { type: 'ERROR'; message: string; category: EggBootTimingErrorCategory; fatal: boolean };
```

## 6. エラーハンドリング

### 6.1 エラーカテゴリ

| カテゴリ | 説明 | リカバリ可能性 |
|----------|------|----------------|
| `VALIDATION` | パラメータ検証エラー | 可能 (パラメータ修正) |
| `WASM_INIT` | WASM 初期化エラー | 一部可能 (リロード) |
| `RUNTIME` | 実行時エラー | 限定的 |
| `ABORTED` | ユーザー中断 | N/A |

### 6.2 Worker エラー時の動作

1. エラーした Worker を終了
2. 残りの Worker で継続
3. 全 Worker 失敗時は検索終了
4. Store にエラー状態を反映

## 7. 既存パターンとの統合

### 7.1 参照する既存コンポーネント

| 既存コンポーネント | 流用内容 |
|-------------------|----------|
| `SearchControlCard` | 検索制御 UI パターン、Worker 管理呼び出し |
| `EggRunCard` | タマゴ生成 UI パターン、Store 連携 |
| `MultiWorkerSearchManager` | 並列 Worker 管理パターン |
| `EggBootTimingMultiWorkerManager` | 孵化乱数起動時間検索特化の並列管理 |

### 7.2 新規実装コンポーネント

| コンポーネント | 責務 |
|---------------|------|
| `EggBootTimingSearchPanel` | パネル全体のレイアウト |
| `EggBootTimingParamsCard` | 起動時間パラメータ入力 |
| `EggBootTimingRunCard` | 検索制御・進捗表示 |
| `EggBootTimingResultsCard` | 結果表示・フィルタリング |

## 8. パフォーマンス考慮

### 8.1 バッチ処理

- 結果はバッチ単位で送信（メモリ効率）
- バッチサイズは組み合わせ数に応じて動的調整
- `calculateBatchSize()` による最適化

### 8.2 進捗更新頻度

- 500ms 間隔で集約進捗を報告
- UI 描画負荷を抑制しつつリアルタイム性を確保

### 8.3 結果件数制限

- Store 側で `MAX_RESULTS` (10000件) を超える場合は古い結果を削除
- メモリ使用量を制限

## 9. 参考ドキュメント

- `/spec/agent/pr_egg_boot_timing_search/SPECIFICATION.md` - 全体仕様
- `/spec/agent/pr_egg_boot_timing_search/TYPESCRIPT_DESIGN.md` - TypeScript設計
- `src/components/search/control/SearchControlCard.tsx` - 既存検索制御UI
- `src/lib/search/multi-worker-manager.ts` - 既存並列Worker管理
- `src/lib/egg/boot-timing-egg-multi-worker-manager.ts` - 孵化乱数並列管理
