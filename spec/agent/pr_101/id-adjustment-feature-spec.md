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

### 2.1 検索条件VO（IdAdjustmentSearchConditionVO）

検索条件はValue Objectとしてまとめて管理する。

```typescript
/**
 * ID調整検索条件VO
 * IdAdjustmentCardから入力される全ての検索パラメータをまとめた値オブジェクト
 */
interface IdAdjustmentSearchConditionVO {
  // --- ID検索パラメータ ---
  targetTid: number;           // 表ID（必須、0〜65535）
  targetSid: number | null;    // 裏ID（任意、0〜65535 または null）
  shinyPid: number | null;     // 色違いにしたい個体のPID（任意、0〜0xFFFFFFFF または null）
  
  // --- 検索期間パラメータ ---
  dateRange: {
    startYear: number;         // 検索開始年 (2000〜2099)
    startMonth: number;        // 検索開始月 (1〜12)
    startDay: number;          // 検索開始日 (1〜31)
    endYear: number;           // 検索終了年 (2000〜2099)
    endMonth: number;          // 検索終了月 (1〜12)
    endDay: number;            // 検索終了日 (1〜31)
  };
  timeRange: {
    hour: { start: number; end: number };     // 0〜23
    minute: { start: number; end: number };   // 0〜59
    second: { start: number; end: number };   // 0〜59
  };
  
  // --- キー入力パラメータ（IdAdjustmentCardから入力） ---
  keyInputMask: number;        // 許可するキー入力マスク
}
```

### 2.2 暗黙的パラメータ（ProfileCardから取得）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `romVersion` | `ROMVersion` | ROMバージョン（B/W/B2/W2） |
| `romRegion` | `ROMRegion` | ROMリージョン |
| `hardware` | `Hardware` | ハードウェア（DS/DS Lite/3DS） |
| `macAddress` | `[number, number, number, number, number, number]` | MACアドレス |
| `timer0Range` | `{ min: number, max: number }` | Timer0範囲 |
| `vcountRange` | `{ min: number, max: number }` | VCount範囲 |
| `newGame` | `boolean` | 始めからかどうか |
| `withSave` | `boolean` | セーブデータがあるか |
| `memoryLink` | `boolean` | 思い出リンク済みか（BW2のみ） |

**注**: `keyInputMask` はProfileCardからではなくIdAdjustmentCardから入力する。

## 3. 検索結果仕様

### 3.1 結果データ構造

```typescript
interface IdAdjustmentSearchResult {
  boot: {
    datetime: Date;         // 起動日時
    timer0: number;         // Timer0値
    vcount: number;         // VCount値
    keyCode: number;        // キー入力コード
    keyInputNames: string[]; // キー入力名
    macAddress: readonly [number, number, number, number, number, number];
  };
  lcgSeedHex: string;       // 初期Seed（16進数文字列）
  tid: number;              // 算出された表ID
  sid: number;              // 算出された裏ID
  isShiny?: boolean;        // 指定PIDが色違いになるか（shinyPid指定時）
}
```

### 3.2 表示制限

- 最大結果件数: 32件
- 結果は仮想テーブル（Virtual Table）を用いて表示

### 3.3 結果テーブルカラム

| カラム名 | 説明 |
|---------|------|
| 日時 | 起動日時 |
| Timer0 | Timer0値（16進表示） |
| VCount | VCount値（16進表示） |
| キー入力 | キー入力名の一覧 |
| 初期Seed | LCG Seed（16進数） |
| 表ID | TID |
| 裏ID | SID |
| 色違い | 色違い判定結果（shinyPid指定時のみ） |

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
    searchCondition, 
    results, 
    isSearching, 
    progress,
    startSearch,
    stopSearch,
    updateSearchCondition 
  } = useIdAdjustmentStore();
  
  // ProfileCardから暗黙的パラメータを取得（keyInputMask以外）
  const profile = useActiveProfile();
  
  return (
    <PanelCard title="ID調整">
      <IdAdjustmentSearchForm 
        condition={searchCondition} 
        onConditionChange={updateSearchCondition} 
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
  condition: IdAdjustmentSearchConditionVO;
  onConditionChange: (condition: Partial<IdAdjustmentSearchConditionVO>) => void;
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

interface IdAdjustmentWorkerManager {
  startSearch(condition: IdAdjustmentSearchConditionVO, profile: ProfileData): Promise<void>;
  stopSearch(): void;
  onProgress(callback: (progress: SearchProgress) => void): void;
  onResults(callback: (results: IdAdjustmentSearchResult[]) => void): void;
  onComplete(callback: (completion: SearchCompletion) => void): void;
  onError(callback: (error: Error) => void): void;
}
```

検索空間の分割方式:
- Timer0 × VCount × KeyCode のセグメント単位で分割
- 各Workerに均等にセグメントを割り当て

### 6.2 Worker通信プロトコル

```typescript
// Worker への要求
type IdAdjustmentWorkerRequest =
  | { type: 'START_SEARCH'; condition: IdAdjustmentSearchConditionVO; profile: ProfileData; workerIndex: number; totalWorkers: number }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'STOP' };

// Worker からの応答
type IdAdjustmentWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: IdAdjustmentProgress }
  | { type: 'RESULTS'; payload: { results: IdAdjustmentSearchResult[]; batchIndex: number } }
  | { type: 'COMPLETE'; payload: IdAdjustmentCompletion }
  | { type: 'ERROR'; message: string; category: string; fatal: boolean };
```

### 6.3 Worker内部処理フロー

```
1. WASM 初期化
2. 検索パラメータ解析
3. 自WorkerIDに基づくセグメント割り当て計算
4. セグメントループ開始
   ├── timer0 範囲をイテレート
   │   ├── vcount 範囲をイテレート
   │   │   ├── keyCode 一覧をイテレート
   │   │   │   ├── WASM IdAdjustmentSearchIterator 作成
   │   │   │   ├── 時刻範囲を走査
   │   │   │   │   ├── LCG Seed 計算
   │   │   │   │   ├── TID/SID 算出
   │   │   │   │   ├── フィルタ条件マッチ判定
   │   │   │   │   └── マッチ時: 結果をバッファに追加
   │   │   │   └── 定期的に結果をメインスレッドへ送信
5. 完了通知
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

```rust
// wasm-pkg/src/id_adjustment_search.rs

use wasm_bindgen::prelude::*;

/// ID調整検索結果
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct IdAdjustmentSearchResult {
    // 起動条件
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
    pub is_shiny: bool,  // shinyPid指定時のみ有効
}

/// ID調整検索イテレータ
#[wasm_bindgen]
pub struct IdAdjustmentSearchIterator {
    // 内部状態
    ds_config: DSConfigJs,
    segment_params: SegmentParamsJs,
    time_range_params: TimeRangeParamsJs,
    search_range_params: SearchRangeParamsJs,
    
    // フィルタ条件
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

```rust
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
        
        // 6. 色違い判定（shinyPid 指定時）
        let is_shiny = if let Some(pid) = self.shiny_pid {
            check_shiny(pid, tid_sid.tid, tid_sid.sid)
        } else {
            false
        };
        
        Some(IdAdjustmentSearchResult {
            // 結果構築
            ...
        })
    }
}

fn check_shiny(pid: u32, tid: u16, sid: u16) -> bool {
    // 色違い判定式: (pid_upper ^ pid_lower ^ tid ^ sid) < 8
    let pid_upper = (pid >> 16) as u16;
    let pid_lower = (pid & 0xFFFF) as u16;
    (pid_upper ^ pid_lower ^ tid ^ sid) < 8
}
```

## 8. 状態管理設計

### 8.1 Zustand Store

```typescript
// src/store/id-adjustment-store.ts
import { create } from 'zustand';

/**
 * ID調整検索条件VO
 */
interface IdAdjustmentSearchConditionVO {
  targetTid: number;
  targetSid: number | null;
  shinyPid: number | null;
  dateRange: DateRange;
  timeRange: TimeRange;
  keyInputMask: number;
}

interface IdAdjustmentState {
  // 検索条件VO
  searchCondition: IdAdjustmentSearchConditionVO;
  
  // 検索状態
  isSearching: boolean;
  isPaused: boolean;
  progress: SearchProgress;
  
  // 検索結果（最大32件）
  results: IdAdjustmentSearchResult[];
  
  // エラー
  error: Error | null;
}

interface IdAdjustmentActions {
  updateSearchCondition: (condition: Partial<IdAdjustmentSearchConditionVO>) => void;
  startSearch: (profile: ProfileData) => Promise<void>;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;
  clearResults: () => void;
  addResults: (results: IdAdjustmentSearchResult[]) => void;
  setError: (error: Error | null) => void;
}

const MAX_RESULTS = 32;

const useIdAdjustmentStore = create<IdAdjustmentState & IdAdjustmentActions>((set, get) => ({
  // 初期状態
  searchCondition: {
    targetTid: 0,
    targetSid: null,
    shinyPid: null,
    dateRange: { startYear: 2010, startMonth: 1, startDay: 1, endYear: 2010, endMonth: 12, endDay: 31 },
    timeRange: { hour: { start: 0, end: 23 }, minute: { start: 0, end: 59 }, second: { start: 0, end: 59 } },
    keyInputMask: 0,
  },
  isSearching: false,
  isPaused: false,
  progress: { processedCombinations: 0, totalCombinations: 0, foundCount: 0, progressPercent: 0, elapsedMs: 0, estimatedRemainingMs: 0 },
  results: [],
  error: null,
  
  // アクション
  updateSearchCondition: (condition) => set((state) => ({ 
    searchCondition: { ...state.searchCondition, ...condition } 
  })),
  
  startSearch: async (profile) => {
    // WorkerManager を通じて検索開始
    // 検索条件VOとProfileDataを渡す
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
    results: [...state.results, ...newResults].slice(0, MAX_RESULTS)
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
└── 結果テーブル（仮想スクロール、最大32件）
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

```typescript
// 設定可能な定数として定義
const ID_ADJUSTMENT_CONFIG = {
  MIN_WORKERS: 1,
  MAX_WORKERS: 8,  // メモリ使用量考慮: 1 Worker あたり約 50MB
  RESERVED_THREADS: 1,  // UIスレッド確保用
};

const workerCount = Math.max(
  ID_ADJUSTMENT_CONFIG.MIN_WORKERS, 
  Math.min(
    navigator.hardwareConcurrency - ID_ADJUSTMENT_CONFIG.RESERVED_THREADS, 
    ID_ADJUSTMENT_CONFIG.MAX_WORKERS
  )
);
```

Worker数上限の根拠:
- 各Workerは WASM インスタンスを保持するため、約 50MB のメモリを消費
- 8 Worker × 50MB = 400MB が上限目安
- 実装時にプロファイリングを行い、必要に応じて調整可能とする

### 13.2 バッチ処理

```typescript
const BATCH_CONFIG = {
  // 1チャンクあたりの処理秒数: 7日分
  CHUNK_SECONDS: 3600 * 24 * 7,  // 604,800秒
  
  // 検索結果の上限
  MAX_RESULTS: 32,
  
  // 進捗報告インターバル（ms）
  PROGRESS_INTERVAL_MS: 500,
};
```

パラメータ設計根拠:
- `CHUNK_SECONDS`: 7日分（604,800秒）を1チャンクとして処理
- `MAX_RESULTS`: 32件を上限とし、それ以降の検索結果は破棄
- `PROGRESS_INTERVAL_MS`: UI更新頻度とのバランス（既存実装と同値）

### 13.3 メモリ管理

- 結果は最大32件に制限
- 上限到達後は新規追加を停止
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

- 既存実装: `mt-seed-boot-timing-worker.ts`
- 既存実装: `offset_calculator.rs` (`calculate_tid_sid_from_seed`)
- 既存仕様書: `spec/implementation/02-algorithms.md`
- 既存仕様書: `spec/implementation/algorithms/offset-calculator.md`

---

**作成日**: 2025年12月3日  
**バージョン**: 1.0  
**作成者**: GitHub Copilot  
**関連PR**: #101
