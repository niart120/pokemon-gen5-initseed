# ID調整機能 仕様書

## 1. 概要

本仕様書は、ポケモン BW/BW2 における ID調整機能の実装要件を定義する。ユーザーが指定した表ID/裏IDを持つ初期Seedを検索する機能を提供する。

### 1.1 目的

- 任意の表ID（TID）/裏ID（SID）を持つゲーム開始条件を探索
- オプションで色違いにしたい個体のPIDとの組み合わせ検索に対応
- 複数CPU Worker による並列処理で高速検索を実現

### 1.2 配置位置

MiscPanel上の一機能として実装する。

## 2. ユーザー入力仕様

### 2.1 検索パラメータ（IdAdjustmentSearchParams）

検索パラメータは既存の `MtSeedBootTimingSearchParams` / `EggBootTimingSearchParams` と同様の命名規則・構造に従う。
共通の型（`DateRange`, `DailyTimeRange`, `BootCondition` 等）は `@/types/search` から再利用する。

```typescript
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import type { DailyTimeRange, DateRange, BootCondition } from '@/types/search';

// Re-export shared types for convenience
export type { DateRange, BootCondition };

/**
 * ID調整検索パラメータ
 * 既存の MtSeedBootTimingSearchParams と同様の構造
 */
export interface IdAdjustmentSearchParams {
  // === 起動時間パラメータ（boot-timing-search共通） ===

  /** 日付範囲 */
  dateRange: DateRange;

  /**
   * 検索範囲（秒）
   * チャンク分割時にManagerが設定。
   * 指定されている場合、dateRangeからの再計算をスキップする。
   */
  rangeSeconds?: number;

  /** Timer0範囲 */
  timer0Range: {
    min: number; // 0x0000-0xFFFF
    max: number;
  };

  /** VCount範囲 */
  vcountRange: {
    min: number; // 0x00-0xFF
    max: number;
  };

  /** キー入力マスク (ビットマスク) - IdAdjustmentCardから入力 */
  keyInputMask: number;

  /** MACアドレス (6バイト) */
  macAddress: readonly [number, number, number, number, number, number];

  /** ハードウェア */
  hardware: Hardware;

  /** ROMバージョン */
  romVersion: ROMVersion;

  /** ROM地域 */
  romRegion: ROMRegion;

  /** 時刻範囲フィルター（1日の中で検索する時間帯） */
  timeRange: DailyTimeRange;

  // === ID調整固有パラメータ ===

  /** 検索対象の表ID（必須、0〜65535） */
  targetTid: number;

  /** 検索対象の裏ID（任意、0〜65535 または null） */
  targetSid: number | null;

  /** 色違いにしたい個体のPID（任意、0〜0xFFFFFFFF または null） */
  shinyPid: number | null;

  /** ゲームモード */
  gameMode: GameMode;

  // === 制限 ===

  /** 結果上限数 (全体) */
  maxResults: number;
}
```

**注**: `keyInputMask` はProfileCardからではなくIdAdjustmentCardから入力する。

### 2.2 暗黙的パラメータ（ProfileCardから取得）

以下のパラメータは ProfileCard から取得し、`IdAdjustmentSearchParams` に統合する:

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `romVersion` | `ROMVersion` | ROMバージョン（B/W/B2/W2） |
| `romRegion` | `ROMRegion` | ROMリージョン |
| `hardware` | `Hardware` | ハードウェア（DS/DS Lite/3DS） |
| `macAddress` | `[number, number, number, number, number, number]` | MACアドレス |
| `timer0Range` | `{ min: number, max: number }` | Timer0範囲 |
| `vcountRange` | `{ min: number, max: number }` | VCount範囲 |
| `newGame` | `boolean` | 始めからかどうか（GameMode導出に使用） |
| `withSave` | `boolean` | セーブデータがあるか（GameMode導出に使用） |
| `memoryLink` | `boolean` | 思い出リンク済みか（BW2のみ、GameMode導出に使用） |

## 3. 検索結果仕様

### 3.1 結果データ構造

既存の `MtSeedBootTimingSearchResult` と同様に、共通の `BootCondition` 型を使用する。

```typescript
import type { BootCondition } from '@/types/search';
import type { DomainShinyType } from '@/types/domain';

/**
 * ID調整検索結果1件
 */
export interface IdAdjustmentSearchResult {
  /** 起動条件（boot-timing-search共通） */
  boot: BootCondition;

  /** LCG Seed (16進文字列) */
  lcgSeedHex: string;

  /** 算出された表ID */
  tid: number;

  /** 算出された裏ID */
  sid: number;

  /** 
   * 色違いタイプ（shinyPid指定時のみ有効）
   * - 0: Normal（色違いではない）
   * - 1: Square（四角い色違い、最レア）
   * - 2: Star（星形色違い）
   */
  shinyType?: DomainShinyType;
}
```

### 3.2 表示制限

- バッチ検索あたりの結果上限: 32件
- 結果は仮想テーブル（Virtual Table）を用いて表示
- 全体の結果上限は `maxResults` パラメータで制御（デフォルト: 1000件）

### 3.3 結果テーブルカラム

| カラム名 | 説明 |
|---------|------|
| 日時 | 起動日時 |
| 初期Seed | LCG Seed（16進数） |
| 表ID | TID |
| 裏ID | SID |
| 色違い | 色違いタイプ表示（shinyPid指定時のみ、◇=Square/☆=Star/-=Normal） |
| Timer0 | Timer0値（16進表示） |
| VCount | VCount値（16進表示） |
| キー入力 | キー入力名の一覧 |

## 4. アーキテクチャ設計

### 4.1 全体構成

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MiscPanel                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  IdAdjustmentCard                             │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐    │  │
│  │  │ ID入力フォーム │  │ 期間設定      │  │ 検索コントロール│    │  │
│  │  └───────────────┘  └───────────────┘  └────────────────┘    │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │              検索結果テーブル（仮想スクロール）          │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
        │                                            ▲
        ▼                                            │
┌─────────────────────────────────────────────────────────────────────┐
│                    IdAdjustmentWorkerManager                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │   ...      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       └─────────────┴─────────────┴─────────────┘                   │
│                            │                                        │
│                      WASM Interface                                 │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Rust/WASM Layer                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              IdAdjustmentSearchIterator                       │  │
│  │  - HashValuesEnumerator 経由で LCG Seed 計算                  │  │
│  │  - GameMode に応じた TID/SID 算出                             │  │
│  │  - フィルタ条件による結果選別                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 モジュール構成

```
src/
├── components/
│   └── misc/
│       ├── MiscPanel.tsx                    # 既存（IdAdjustmentCardを追加）
│       ├── IdAdjustmentCard.tsx             # 新規: ID調整機能メインコンポーネント
│       ├── IdAdjustmentSearchForm.tsx       # 新規: 検索条件入力フォーム
│       ├── IdAdjustmentResultsTable.tsx     # 新規: 結果表示テーブル
│       └── index.ts                         # 既存（export追加）
├── workers/
│   ├── id-adjustment-worker.ts              # 新規: 検索Worker
│   └── id-adjustment-worker-manager.ts      # 新規: Worker管理
├── types/
│   └── id-adjustment-search.ts              # 新規: 型定義
├── store/
│   └── id-adjustment-store.ts               # 新規: Zustand状態管理
└── hooks/
    └── use-id-adjustment-search.ts          # 新規: 検索操作hook

wasm-pkg/
└── src/
    ├── id_adjustment_search.rs              # 新規: Rust検索実装
    └── lib.rs                               # 既存（export追加）
```

## 5. コンポーネント設計

### 5.1 IdAdjustmentCard

ID調整機能のメインカードコンポーネント。

```typescript
interface IdAdjustmentCardProps {}

const IdAdjustmentCard: React.FC<IdAdjustmentCardProps> = () => {
  // Zustand storeから状態を取得
  const { 
    searchParams, 
    results, 
    isSearching, 
    progress,
    startSearch,
    stopSearch,
    updateSearchParams 
  } = useIdAdjustmentStore();
  
  // ProfileCardから暗黙的パラメータを取得（keyInputMask以外）
  const profile = useActiveProfile();
  
  return (
    <PanelCard title="ID調整">
      <IdAdjustmentSearchForm 
        params={searchParams} 
        onParamsChange={updateSearchParams} 
      />
      <SearchControls 
        isSearching={isSearching}
        progress={progress}
        onStart={() => startSearch(profile)}
        onStop={stopSearch}
      />
      <IdAdjustmentResultsTable results={results} />
    </PanelCard>
  );
};
```

### 5.2 IdAdjustmentSearchForm

検索条件入力フォーム。

```typescript
interface IdAdjustmentSearchFormProps {
  params: IdAdjustmentSearchParams;
  onParamsChange: (params: Partial<IdAdjustmentSearchParams>) => void;
}
```

入力フィールド:
- 表ID（必須、0〜65535）
- 裏ID（任意、0〜65535）
- 色違いPID（任意、16進数入力）
- 検索日付範囲（開始日〜終了日）
- 検索時刻範囲（時分秒それぞれのstart〜end）
- キー入力設定（許可するキー入力マスク）

### 5.3 IdAdjustmentResultsTable

仮想スクロール対応の結果テーブル。

```typescript
interface IdAdjustmentResultsTableProps {
  results: IdAdjustmentSearchResult[];
}
```

既存の `@tanstack/react-virtual` または同等のライブラリを使用。

## 6. Worker設計

### 6.1 WorkerManager

CPUコア数に応じてWorkerを生成・管理する。

```typescript
// id-adjustment-worker-manager.ts
// 既存の MtSeedBootTimingMultiWorkerManager / EggBootTimingMultiWorkerManager と同様の構造

interface IdAdjustmentWorkerManager {
  startSearch(params: IdAdjustmentSearchParams): Promise<void>;
  stopSearch(): void;
  onProgress(callback: (progress: SearchProgress) => void): void;
  onResults(callback: (results: IdAdjustmentSearchResult[]) => void): void;
  onComplete(callback: (completion: SearchCompletion) => void): void;
  onError(callback: (error: Error) => void): void;
}
```

検索空間の分割方式:
- 日時範囲を Worker 数で均等に分割（既存の `calculateTimeChunks` を流用）
- 各 Worker は割り当てられた日時範囲内で timer0 × vcount × keyCode のループを実行

### 6.2 Worker通信プロトコル

既存の `MtSeedBootTimingWorkerRequest` / `MtSeedBootTimingWorkerResponse` と同様の構造に従う。

```typescript
/**
 * Worker リクエスト
 */
export type IdAdjustmentWorkerRequest =
  | {
      type: 'START_SEARCH';
      params: IdAdjustmentSearchParams;
      requestId?: string;
    }
  | {
      type: 'PAUSE';
      requestId?: string;
    }
  | {
      type: 'RESUME';
      requestId?: string;
    }
  | {
      type: 'STOP';
      requestId?: string;
    };

/**
 * Worker レスポンス
 */
export type IdAdjustmentWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: IdAdjustmentProgress }
  | { type: 'RESULTS'; payload: IdAdjustmentResultsPayload }
  | { type: 'COMPLETE'; payload: IdAdjustmentCompletion }
  | {
      type: 'ERROR';
      message: string;
      category: IdAdjustmentErrorCategory;
      fatal: boolean;
    };

/**
 * 進捗情報
 */
export interface IdAdjustmentProgress {
  /** 処理済み起動条件の組み合わせ数（完了セグメント数） */
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

  /** 処理済み秒数（検索範囲の秒数単位） */
  processedSeconds?: number;
}

/**
 * 結果ペイロード（バッチ送信用）
 */
export interface IdAdjustmentResultsPayload {
  results: IdAdjustmentSearchResult[];
  batchIndex: number;
}

/**
 * 完了情報
 */
export interface IdAdjustmentCompletion {
  /** 完了理由 */
  reason: 'completed' | 'stopped' | 'max-results' | 'error';

  /** 処理した起動条件の組み合わせ数 */
  processedCombinations: number;

  /** 総組み合わせ数 */
  totalCombinations: number;

  /** 見つかった結果数 */
  resultsCount: number;

  /** 経過時間 (ms) */
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type IdAdjustmentErrorCategory =
  | 'VALIDATION' // パラメータ検証エラー
  | 'WASM_INIT' // WASM初期化エラー
  | 'RUNTIME' // 実行時エラー
  | 'ABORTED'; // 中断
```

### 6.3 Worker内部処理フロー

各Workerは割り当てられた日時範囲（TimeChunk）に対して検索を実行する。

```
1. WASM 初期化
2. 検索パラメータ解析
3. 割り当てられた日時チャンク（rangeSeconds）の処理開始
4. WASM IdAdjustmentSearchIterator 作成
5. 日時範囲ループ（秒単位）
   ├── 各秒に対して timer0 範囲をイテレート
   │   ├── vcount 範囲をイテレート
   │   │   ├── keyCode 一覧をイテレート
   │   │   │   ├── LCG Seed 計算
   │   │   │   ├── TID/SID 算出
   │   │   │   ├── フィルタ条件マッチ判定
   │   │   │   │   └── shinyPid指定時: ShinyChecker.check_shiny_type() で色違いタイプ判定
   │   │   │   └── マッチ時（Square または Star）: 結果をバッファに追加
   │   │   └── 定期的に結果をメインスレッドへ送信
6. 完了通知
```

## 7. Rust/WASM API設計

### 7.1 既存APIの活用

TID/SID計算には既存の `wasm-pkg/src/offset_calculator.rs` の以下を使用:

```rust
// 既存API
// モジュール: wasm-pkg/src/offset_calculator.rs
// TypeScript側からの使用: import { calculate_tid_sid_from_seed, GameMode, TidSidResult } from '@/lib/core/wasm-interface';

/// TID/SID決定処理統合API
/// @param initial_seed - 64bit LCG Seed
/// @param mode - ゲームモード（BwNewGameWithSave, BwNewGameNoSave等）
/// @returns TidSidResult { tid: u16, sid: u16, advances_used: u32 }
pub fn calculate_tid_sid_from_seed(initial_seed: u64, mode: GameMode) -> TidSidResult
```

参照: `wasm-pkg/src/lib.rs` にて re-export 済み

### 7.2 新規API: IdAdjustmentSearchIterator

既存の `MtSeedBootTimingSearchIterator` と同様の構造に従い、共通のJS用ラッパー型（`DSConfigJs`, `SegmentParamsJs`, `TimeRangeParamsJs`, `SearchRangeParamsJs`）を再利用する。

```rust
// wasm-pkg/src/id_adjustment_search.rs

use wasm_bindgen::prelude::*;
use crate::mt_seed_boot_timing_search::{DSConfigJs, SegmentParamsJs, TimeRangeParamsJs, SearchRangeParamsJs};

/// ID調整検索結果（WASM向け）
/// MtSeedBootTimingSearchResult と同様の起動条件フィールドを持つ
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct IdAdjustmentSearchResult {
    // 起動条件（boot-timing-search共通）
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub timer0: u16,
    pub vcount: u8,
    pub key_code: u16,
    
    // 計算結果
    pub lcg_seed_hex: String,
    pub tid: u16,
    pub sid: u16,
    /// 色違いタイプ（shinyPid指定時のみ有効）
    /// 0: Normal（色違いではない）、1: Square（四角い色違い）、2: Star（星形色違い）
    pub shiny_type: u8,
}

/// ID調整検索イテレータ
/// MtSeedBootTimingSearchIterator と同様のインターフェースを提供
#[wasm_bindgen]
pub struct IdAdjustmentSearchIterator {
    // 内部状態（boot-timing-search共通の構造体を再利用）
    ds_config: DSConfigJs,
    segment_params: SegmentParamsJs,
    time_range_params: TimeRangeParamsJs,
    search_range_params: SearchRangeParamsJs,
    
    // ID調整固有のフィルタ条件
    target_tid: u16,
    target_sid: Option<u16>,
    shiny_pid: Option<u32>,
    game_mode: GameMode,
    
    // イテレーション状態
    current_offset_seconds: u32,
    is_finished: bool,
}

#[wasm_bindgen]
impl IdAdjustmentSearchIterator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        ds_config: DSConfigJs,
        segment_params: SegmentParamsJs,
        time_range_params: TimeRangeParamsJs,
        search_range_params: SearchRangeParamsJs,
        target_tid: u16,
        target_sid: Option<u16>,
        shiny_pid: Option<u32>,
        game_mode: GameMode,
    ) -> IdAdjustmentSearchIterator;
    
    /// 次のバッチを取得
    /// - limit: 最大結果数
    /// - chunk_seconds: 処理する秒数
    pub fn next_batch(&mut self, limit: u32, chunk_seconds: u32) -> IdAdjustmentSearchResults;
    
    /// 検索完了判定
    #[wasm_bindgen(getter)]
    pub fn is_finished(&self) -> bool;
    
    /// 進捗率（0.0〜1.0）
    #[wasm_bindgen(getter)]
    pub fn progress(&self) -> f64;
}
```

### 7.3 検索アルゴリズム

既存の `ShinyChecker` (`wasm-pkg/src/pid_shiny_checker.rs`) を使用して色違い判定を行う。

```rust
use crate::pid_shiny_checker::{ShinyChecker, ShinyType};

impl IdAdjustmentSearchIterator {
    fn search_single_datetime(&self, datetime: DateTime) -> Option<IdAdjustmentSearchResult> {
        // 1. HashValues 計算
        let hash_values = self.calculate_hash_values(datetime);
        
        // 2. LCG Seed 導出
        let lcg_seed = derive_lcg_seed_from_hash(&hash_values);
        
        // 3. GameMode に基づいた TID/SID 計算
        let tid_sid = calculate_tid_sid_from_seed(lcg_seed, self.game_mode);
        
        // 4. TID フィルタ
        if tid_sid.tid != self.target_tid {
            return None;
        }
        
        // 5. SID フィルタ（指定時のみ）
        if let Some(target_sid) = self.target_sid {
            if tid_sid.sid != target_sid {
                return None;
            }
        }
        
        // 6. 色違いタイプ判定（shinyPid 指定時）
        // 既存の ShinyChecker を使用
        let shiny_type = if let Some(pid) = self.shiny_pid {
            ShinyChecker::check_shiny_type(tid_sid.tid, tid_sid.sid, pid)
        } else {
            ShinyType::Normal
        };
        
        // 7. 色違いフィルタ（shinyPid指定時はSquareまたはStarのみ結果に含める）
        if self.shiny_pid.is_some() && shiny_type == ShinyType::Normal {
            return None;
        }
        
        Some(IdAdjustmentSearchResult {
            // 結果構築
            shiny_type: shiny_type as u8, // 0=Normal, 1=Square, 2=Star
            ...
        })
    }
}

// ShinyChecker の既存実装を使用（pid_shiny_checker.rs より）:
// - ShinyChecker::check_shiny_type(tid, sid, pid) -> ShinyType
// - ShinyType::Normal = 0（色違いではない）
// - ShinyType::Square = 1（四角い色違い、shiny_value == 0）
// - ShinyType::Star = 2（星形色違い、shiny_value 1..=7）
```

## 8. 状態管理設計

### 8.1 Zustand Store

```typescript
// src/store/id-adjustment-store.ts
import { create } from 'zustand';
import type { IdAdjustmentSearchParams, IdAdjustmentSearchResult, IdAdjustmentProgress } from '@/types/id-adjustment-search';

interface IdAdjustmentState {
  // 検索パラメータ
  searchParams: IdAdjustmentSearchParams;
  
  // 検索状態
  isSearching: boolean;
  isPaused: boolean;
  progress: IdAdjustmentProgress;
  
  // 検索結果
  results: IdAdjustmentSearchResult[];
  
  // エラー
  error: Error | null;
}

interface IdAdjustmentActions {
  updateSearchParams: (params: Partial<IdAdjustmentSearchParams>) => void;
  startSearch: () => Promise<void>;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;
  clearResults: () => void;
  addResults: (results: IdAdjustmentSearchResult[]) => void;
  setError: (error: Error | null) => void;
}

const useIdAdjustmentStore = create<IdAdjustmentState & IdAdjustmentActions>((set, get) => ({
  // 初期状態（createDefaultIdAdjustmentSearchParams() で生成）
  searchParams: createDefaultIdAdjustmentSearchParams(),
  isSearching: false,
  isPaused: false,
  progress: { 
    processedCombinations: 0, 
    totalCombinations: 0, 
    foundCount: 0, 
    progressPercent: 0, 
    elapsedMs: 0, 
    estimatedRemainingMs: 0 
  },
  results: [],
  error: null,
  
  // アクション
  updateSearchParams: (params) => set((state) => ({ 
    searchParams: { ...state.searchParams, ...params } 
  })),
  
  startSearch: async () => {
    const { searchParams } = get();
    // WorkerManager を通じて検索開始
    // searchParams を渡す（ProfileData は既に統合済み）
  },
  
  pauseSearch: () => {
    // WorkerManager に一時停止を要求
  },
  
  resumeSearch: () => {
    // WorkerManager に再開を要求
  },
  
  stopSearch: () => {
    // WorkerManager に停止を要求
  },
  
  clearResults: () => set({ results: [] }),
  
  addResults: (newResults) => set((state) => ({
    results: [...state.results, ...newResults].slice(0, state.searchParams.maxResults)
  })),
  
  setError: (error) => set({ error }),
}));
```

## 9. GameMode 決定ロジック

ProfileCard の設定から GameMode を導出する。

```typescript
function deriveGameMode(profile: ProfileData): GameMode {
  const { romVersion, newGame, withSave, memoryLink } = profile;
  
  const isBW2 = romVersion === 'B2' || romVersion === 'W2';
  
  if (isBW2) {
    if (!newGame) {
      // 続きから
      return memoryLink ? 'Bw2ContinueWithMemoryLink' : 'Bw2ContinueNoMemoryLink';
    } else {
      // 始めから
      if (!withSave) {
        return 'Bw2NewGameNoSave';
      }
      return memoryLink ? 'Bw2NewGameWithMemoryLinkSave' : 'Bw2NewGameNoMemoryLinkSave';
    }
  } else {
    // BW
    if (!newGame) {
      return 'BwContinue';
    }
    return withSave ? 'BwNewGameWithSave' : 'BwNewGameNoSave';
  }
}
```

**重要**: 「続きから」モードではTID/SID決定処理は行われないため、ID調整検索の対象外となる。UIでは「始めから」モードのみを許可するか、警告を表示する。

## 10. バリデーション仕様

### 10.1 入力バリデーション

| 項目 | 条件 | エラーメッセージ |
|------|------|------------------|
| 表ID | 0〜65535の整数 | 表IDは0〜65535の範囲で入力してください |
| 裏ID | null または 0〜65535の整数 | 裏IDは0〜65535の範囲で入力してください |
| 色違いPID | null または 0〜0xFFFFFFFFの整数 | PIDは0〜FFFFFFFFの16進数で入力してください |
| 検索開始日 | 有効な日付 | 有効な日付を入力してください |
| 検索終了日 | 開始日以降 | 終了日は開始日以降を指定してください |
| 時刻範囲 | start <= end | 開始は終了以前を指定してください |
| GameMode | 「始めから」モード | ID調整には「始めから」モードを選択してください |

### 10.2 実行時バリデーション

- ProfileCardの必須項目が未設定の場合、検索を開始しない
- 検索中は入力フォームを無効化

## 11. UI/UXガイドライン

### 11.1 レイアウト

MiscPanel の3列グリッド内の1列目に配置:

```
┌────────────────┬────────────────┬────────────────┐
│  MT Seed検索   │   ID調整       │    (空き)      │
│  (既存)        │   (新規)       │                │
└────────────────┴────────────────┴────────────────┘
```

### 11.2 コンポーネント構成

```
IdAdjustmentCard
├── ヘッダー（タイトル、ヘルプアイコン）
├── ID入力セクション
│   ├── 表ID入力（数値）
│   ├── 裏ID入力（数値、任意）
│   └── 色違いPID入力（16進数、任意）
├── 期間設定セクション
│   ├── 日付範囲（DatePicker または 年月日入力）
│   └── 時刻範囲（時分秒の範囲入力）
├── キー入力設定セクション
│   └── 許可するキー入力マスク設定
├── 検索コントロール
│   ├── 検索開始/停止ボタン
│   └── 進捗バー
└── 結果テーブル（仮想スクロール）
```

### 11.3 レスポンシブ対応

- デスクトップ: 3列グリッド
- モバイル: スタック配置、折りたたみ可能

## 12. エラーハンドリング

### 12.1 エラーカテゴリ

| カテゴリ | 説明 | 対応 |
|---------|------|------|
| VALIDATION | 入力値エラー | フォームにエラーメッセージ表示 |
| WASM_INIT | WASM初期化失敗 | リトライボタン表示 |
| RUNTIME | 実行時エラー | トースト通知、検索停止 |
| WORKER | Worker通信エラー | 自動リトライ（3回まで） |

### 12.2 Worker異常終了時の復旧

- 異常終了したWorkerを検出し、残りのWorkerで継続
- 全Worker異常終了時は検索を中止し、エラー通知

## 13. パフォーマンス考慮事項

### 13.1 Worker数の決定

既存の `getDefaultWorkerCount()` 関数（`@/lib/search/chunk-calculator.ts`）を使用する。

```typescript
import { getDefaultWorkerCount } from '@/lib/search/chunk-calculator';

// 既存実装を流用
export function getDefaultWorkerCount(): number {
  return typeof navigator !== 'undefined'
    ? navigator.hardwareConcurrency || 4
    : 4;
}

// WorkerManager のコンストラクタで使用
class IdAdjustmentMultiWorkerManager {
  constructor(private maxWorkers: number = getDefaultWorkerCount()) {}
  
  setMaxWorkers(count: number): void {
    const maxHwConcurrency = getDefaultWorkerCount();
    this.maxWorkers = Math.max(1, Math.min(count, maxHwConcurrency));
  }
}
```

既存の `MtSeedBootTimingMultiWorkerManager` / `EggBootTimingMultiWorkerManager` と同じ方式を採用し、特別な上限設定は行わない。

### 13.2 バッチ処理

```typescript
const BATCH_CONFIG = {
  // 1チャンクあたりの処理秒数: 7日分
  CHUNK_SECONDS: 3600 * 24 * 7,  // 604,800秒
  
  // バッチ検索あたりの結果上限
  BATCH_RESULT_LIMIT: 32,
  
  // 進捗報告インターバル（ms）
  PROGRESS_INTERVAL_MS: 500,
};
```

パラメータ設計根拠:
- `CHUNK_SECONDS`: 7日分（604,800秒）を1チャンクとして処理
- `BATCH_RESULT_LIMIT`: 1回のバッチ検索で32件を上限とし、それ以降は次のバッチへ
- `PROGRESS_INTERVAL_MS`: UI更新頻度とのバランス（既存実装と同値）

### 13.3 メモリ管理

- バッチあたりの結果は32件に制限
- 全体の結果上限は `maxResults` パラメータで制御（デフォルト: 1000件）
- Worker終了時はリソースを解放

## 14. テスト仕様

### 14.1 ユニットテスト

| 対象 | テスト内容 |
|------|-----------|
| TID/SID計算 | 既知のSeedからのTID/SID計算が正確か |
| 色違い判定 | 各種TID/SID/PID組み合わせでの判定 |
| GameMode導出 | ProfileDataからのGameMode導出 |
| 入力バリデーション | 各フィールドのバリデーション |

### 14.2 統合テスト

| 対象 | テスト内容 |
|------|-----------|
| Worker通信 | メッセージ送受信の正確性 |
| 並列検索 | 複数Workerでの検索結果の一貫性 |
| 一時停止/再開 | 状態の正確な保持と復旧 |

### 14.3 E2Eテスト

| 対象 | テスト内容 |
|------|-----------|
| 基本フロー | ID入力 → 検索実行 → 結果表示 |
| エラーケース | 無効な入力でのエラー表示 |
| パフォーマンス | 大量結果でのスクロール性能 |

## 15. 実装優先順位

### Phase 1: 基盤実装

1. Rust API (`IdAdjustmentSearchIterator`) の実装
2. 型定義 (`id-adjustment-search.ts`) の作成
3. Worker (`id-adjustment-worker.ts`) の実装

### Phase 2: UI実装

4. Zustand Store の実装
5. `IdAdjustmentCard` コンポーネントの実装
6. `IdAdjustmentSearchForm` の実装
7. `IdAdjustmentResultsTable` の実装

### Phase 3: 統合・最適化

8. WorkerManager の実装
9. MiscPanel への統合
10. パフォーマンス最適化
11. テスト作成

## 16. 参考資料

### 16.1 既存実装（流用対象）

- Worker管理: `src/lib/mt-seed/mt-seed-boot-timing-multi-worker-manager.ts`
- Worker管理: `src/lib/egg/boot-timing-egg-multi-worker-manager.ts`
- チャンク分割: `src/lib/search/chunk-calculator.ts` (`getDefaultWorkerCount`, `calculateTimeChunks`)
- 色違い判定: `wasm-pkg/src/pid_shiny_checker.rs` (`ShinyChecker`, `ShinyType`)
- TID/SID計算: `wasm-pkg/src/offset_calculator.rs` (`calculate_tid_sid_from_seed`)
- ドメイン型: `src/types/domain.ts` (`DomainShinyType`, `DomainGameMode`)

### 16.2 仕様書

- 既存仕様書: `spec/implementation/02-algorithms.md`
- 既存仕様書: `spec/implementation/algorithms/offset-calculator.md`

---

**作成日**: 2025年12月3日  
**バージョン**: 1.1  
**作成者**: GitHub Copilot  
**関連PR**: #101

### 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| 1.1 | 2025-12-03 | Worker数決定ロジックを既存実装の流用に変更、色違い判定をShinyType (Square/Star)ベースに変更、検索空間分割を日時範囲のみに変更、結果テーブルカラム順序を変更 |
| 1.0 | 2025-12-03 | 初版作成 |
