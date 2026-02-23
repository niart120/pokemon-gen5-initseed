# Rust WASM 処理共通化リファクタリング方針

## 対象ファイル
- `wasm-pkg/src/integrated_search.rs` - 初期Seed検索 (32bit MT Seed) → 廃止予定
- `wasm-pkg/src/egg_boot_timing_search.rs` - 孵化乱数起動時間検索 (64bit LCG Seed)

## 現状の問題点

### 1. コード重複
以下の処理が両ファイルで重複している：

| 処理 | integrated_search.rs | egg_boot_timing_search.rs |
|------|---------------------|--------------------------|
| DailyTimeRangeConfig | 完全重複 | 完全重複 |
| build_allowed_second_mask | 完全重複 | 完全重複 |
| base_message構築ロジック | 約50行 | 約50行 |
| calculate_datetime_codes | ほぼ同一 | ほぼ同一 |
| generate_display_datetime | ほぼ同一 | ほぼ同一 |
| build_message | 類似 | 類似 |
| process_simd_batch | 構造類似 | 構造類似 |
| 定数 (EPOCH_2000_UNIX, SECONDS_PER_DAY) | 重複 | 重複 |
| hardwareバリデーション | 重複 | 重複 + frame導出 |

### 2. 命名問題
- `IntegratedSeedSearcher`: "Integrated" は抽象的で目的が不明確
- 本質は「IV確定用の起動時間検索」であり、命名を明確化する必要がある

---

## 新しい命名規則

### 命名方針
- 目的を明確に示す命名
- TS側とRust側の役割を明確に分離

| 旧名称 | 新名称 | 説明 |
|--------|--------|------|
| `IntegratedSeedSearcher` | `MtSeedBootTimingSearchIterator` | MT Seedから起動時間を検索 |
| `IntegratedSearchResult` | `MtSeedBootTimingSearchResult` | MT Seed検索結果 |
| - | `DSConfigJs` | DS設定パラメータ (MAC/Nazo/Hardware) |
| - | `SegmentParamsJs` | セグメントパラメータ (Timer0/VCount/KeyCode) |
| - | `TimeRangeParamsJs` | 時刻範囲パラメータ |
| - | `SearchRangeParamsJs` | 検索範囲パラメータ |

---

## アーキテクチャ設計

### 型の分類

#### 1. 公開型（wasm-bindgen経由でTSに公開）
TypeScriptとのインターフェースとなるValue Object。`#[wasm_bindgen]` を付与。

```rust
// パラメータ用Value Object
#[wasm_bindgen]
pub struct DSConfigJs { mac, nazo, hardware }

#[wasm_bindgen]
pub struct SegmentParamsJs { timer0, vcount, key_code }

#[wasm_bindgen]  
pub struct TimeRangeParamsJs { hour_start/end, minute_start/end, second_start/end }

#[wasm_bindgen]
pub struct SearchRangeParamsJs { start_year/month/day, range_seconds }

// 検索イテレータ・結果
#[wasm_bindgen]
pub struct MtSeedBootTimingSearchIterator { ... }

#[wasm_bindgen]
pub struct MtSeedBootTimingSearchResult { ... }
```

#### 2. 内部型（Rust内部のみ）
パフォーマンス最適化やロジック共通化のための型。`#[wasm_bindgen]` なし。

```rust
// ハードウェア種別 (String → enum変換)
pub enum HardwareType { DS, DSLite, ThreeDS }

// パラメータ内部型
pub struct DSConfig { mac, nazo, hardware }
pub struct SegmentParams { timer0, vcount, key_code }
pub struct TimeRangeParams { hour_start/end, minute_start/end, second_start/end }
pub struct SearchRangeParams { start_year/month/day, range_seconds }

// SHA-1メッセージ構築
pub struct BaseMessageBuilder { ... }

// 日時コード列挙
pub struct DateTimeCodeEnumerator { ... }
pub struct DateTimeCode { date_code, time_code }

// SHA-1ハッシュ値
pub struct HashValues { h0..h4 }
```

### TS-Rust責務分離

| 責務 | TS側 | Rust側 |
|------|------|--------|
| セグメントループ制御 | ✓ | - |
| KeyCode生成 | ✓ | - |
| 進捗報告・キャンセル | ✓ | - |
| パラメータ検証 | 基本検証 | 詳細検証 |
| SHA-1計算 | - | ✓ |
| SIMD最適化 | - | ✓ |
| 結果フィルタリング | 表示用 | 計算用 |

---

## モジュール分割設計（B案: 役割別分離） ✅ 実装済み

### ファイル構成

```
wasm-pkg/src/
├── search_common/
│   ├── mod.rs           # re-export, 定数
│   ├── params.rs        # DSConfig, SegmentParams, TimeRangeParams, SearchRangeParams
│   │                    # DSConfigJs, SegmentParamsJs, TimeRangeParamsJs, SearchRangeParamsJs
│   ├── message.rs       # BaseMessageBuilder, HashValues
│   └── datetime.rs      # DateTimeCodeEnumerator, RangedTimeCodeTable, ユーティリティ関数
└── ...
```

### 依存関係

```
params.rs ← message.rs ← datetime.rs
    ↑           ↑
    └───────────┴─── (外部モジュールから参照)
```

---

## RangedTimeCodeTable 仕様 ✅ 実装済み

### 概要

時刻範囲に基づいて許可された秒のtime_codeを事前計算するテーブル。

### 設計

```rust
/// 範囲制限タイムコードテーブル（1日分、86400要素）
/// 
/// 各インデックスが1日の秒数（0-86399）に対応。
/// - Some(time_code): 許可された秒、事前計算されたtime_code
/// - None: 許可されていない秒
pub type RangedTimeCodeTable = Box<[Option<u32>; 86400]>;

pub fn build_ranged_time_code_table(
    range: &TimeRangeParams, 
    hardware: HardwareType
) -> RangedTimeCodeTable;
```

**メリット**:
- 許可判定とtime_code取得を1回のルックアップで完了
- time_code計算を事前に済ませ、イテレーション中の計算コスト削減

---

## DateTimeCodeEnumerator 仕様 ✅ 実装済み

### 概要

開始秒から指定秒数分の `DateTimeCode` を順次生成するIterator。
許可範囲外の秒はスキップされるが、進捗計算にはスキップ分も含まれる。

### インターフェース

```rust
/// 日時コード（イテレータの出力）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateTimeCode {
    pub date_code: u32,
    pub time_code: u32,
}

/// 日時コード列挙器
pub struct DateTimeCodeEnumerator<'a> {
    time_code_table: &'a RangedTimeCodeTable,
    current_seconds: i64,
    end_seconds: i64,
    processed_seconds: u32,
}

impl<'a> DateTimeCodeEnumerator<'a> {
    pub fn new(
        time_code_table: &'a RangedTimeCodeTable,
        start_seconds: i64,
        range_seconds: u32,
    ) -> Self;
    
    pub fn processed_seconds(&self) -> u32;
}

impl Iterator for DateTimeCodeEnumerator<'_> {
    type Item = DateTimeCode;
    fn next(&mut self) -> Option<Self::Item>;
}
```

### SIMD バッチ処理対応

Iterator の標準機能 `take(4)` で4要素ずつ取得可能：

```rust
let table = build_ranged_time_code_table(&time_range, hardware);
let mut enumerator = DateTimeCodeEnumerator::new(&table, start_seconds, range_seconds);

loop {
    let batch: Vec<DateTimeCode> = enumerator.by_ref().take(4).collect();
    if batch.is_empty() {
        break;
    }
    
    if batch.len() == 4 {
        // 4要素揃っている場合はSIMD
        process_simd_batch(&batch);
    } else {
        // 端数はスカラー処理
        for dt in batch {
            process_scalar(&dt);
        }
    }
}
```

---

## MtSeedBootTimingSearchIterator 仕様

### 概要

複数のtarget_seedsに対してマッチする起動時間を検索するIterator。

### 設計

```rust
#[wasm_bindgen]
pub struct MtSeedBootTimingSearchIterator {
    // 内部状態
    base_message_builder: BaseMessageBuilder,
    time_code_table: RangedTimeCodeTable,
    
    // 検索条件（複数Seed対応）
    target_seeds: Vec<u32>,
    
    // 検索範囲
    start_seconds: i64,
    range_seconds: u32,
    
    // 現在位置
    current_seconds: i64,
    processed_seconds: u32,
}

#[wasm_bindgen]
impl MtSeedBootTimingSearchIterator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        ds_config: &DSConfigJs,
        segment: &SegmentParamsJs,
        time_range: &TimeRangeParamsJs,
        search_range: &SearchRangeParamsJs,
        target_seeds: &[u32],  // 複数Seed対応
    ) -> Result<MtSeedBootTimingSearchIterator, String>;

    /// バッチ処理で検索を進める
    /// 
    /// # Arguments
    /// - `max_results`: 最大結果数（見つかったら早期終了）
    /// - `chunk_seconds`: 処理する秒数（進捗報告用）
    /// 
    /// # Returns
    /// 見つかった結果の配列
    pub fn next_batch(
        &mut self, 
        max_results: u32, 
        chunk_seconds: u32
    ) -> MtSeedBootTimingSearchResults;
    
    #[wasm_bindgen(getter)]
    pub fn is_finished(&self) -> bool;
    
    #[wasm_bindgen(getter)]
    pub fn processed_seconds(&self) -> u32;
    
    #[wasm_bindgen(getter)]
    pub fn total_seconds(&self) -> u32;
    
    #[wasm_bindgen(getter)]
    pub fn progress(&self) -> f64;
}

#[wasm_bindgen]
pub struct MtSeedBootTimingSearchResult {
    pub seed: u32,
    pub year: u32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

#[wasm_bindgen]
pub struct MtSeedBootTimingSearchResults {
    results: Vec<MtSeedBootTimingSearchResult>,
    processed_in_chunk: u32,
}
```

### 使用例（TS側）

```typescript
const iterator = new MtSeedBootTimingSearchIterator(
    dsConfig,
    segment,
    timeRange,
    searchRange,
    new Uint32Array([0x12345678, 0xABCDEF00])  // 複数Seed
);

while (!iterator.is_finished) {
    const results = iterator.next_batch(10, 3600);  // 最大10件、1時間分処理
    
    if (results.length > 0) {
        // 結果が見つかった
        for (const result of results) {
            console.log(`Seed: ${result.seed}, ${result.year}/${result.month}/${result.day}`);
        }
    }
    
    // 進捗報告
    reportProgress(iterator.progress);
    
    // キャンセルチェック
    if (shouldCancel) break;
}
```

---

## リファクタリング方針

### Phase 1: 共通モジュール抽出 ✅ 完了

#### 実装済み内容

```
wasm-pkg/src/search_common/
├── mod.rs           # 定数 + re-exports
├── params.rs        # 内部型 + 公開型（DSConfig, SegmentParams, TimeRangeParams, SearchRangeParams）
├── message.rs       # BaseMessageBuilder, HashValues
└── datetime.rs      # DateTimeCodeEnumerator, RangedTimeCodeTable, ユーティリティ関数
```

- 37テスト通過
- 175テスト（全体）通過

### Phase 2: MtSeedBootTimingSearchIterator 実装 ✅ 完了

#### 目標
`EggBootTimingSearchIterator` と同様のIteratorパターンでMT Seed検索を実装。

#### タスク
1. `mt_seed_boot_timing_search.rs` 新規作成
2. `MtSeedBootTimingSearchIterator` 実装
3. `MtSeedBootTimingSearchResult` / `MtSeedBootTimingSearchResults` 実装
4. SIMD最適化
5. テスト追加

### Phase 3: 既存コード移行

1. `integrated_search.rs` は非推奨化（`#[deprecated]`）
2. 新規開発は `MtSeedBootTimingSearchIterator` を使用
3. 段階的に既存呼び出しを移行

---

## ファイル構成

```
wasm-pkg/src/
├── lib.rs
├── search_common/             # 共通モジュール ✅
│   ├── mod.rs
│   ├── params.rs
│   ├── message.rs
│   └── datetime.rs
├── mt_seed_boot_timing_search.rs   # MT Seed起動時間検索 ✅
├── egg_boot_timing_search.rs  # 孵化乱数検索 (既存)
├── integrated_search.rs       # 旧IV検索 (deprecated予定)
├── datetime_codes.rs          # 日時コード生成 (既存)
└── ...
```

---

## 実装状況

| Step | 内容 | 状態 |
|------|------|------|
| 1 | ブランチ作成 | ✅ 完了 |
| 2 | `search_common/` モジュール分割 | ✅ 完了 |
| 3 | 内部型 + 公開型実装 | ✅ 完了 |
| 4 | `DateTimeCodeEnumerator` 実装 | ✅ 完了 |
| 5 | `RangedTimeCodeTable` 実装 | ✅ 完了 |
| 6 | テスト (37件) | ✅ 通過 |
| 7 | `MtSeedBootTimingSearchIterator` 実装 | ✅ 完了 |
| 8 | 既存コード移行 | ⬜ 未着手 |

---

## 期待効果

| 項目 | Before | After |
|------|--------|-------|
| 重複コード行数 | 約300行 | 約50行 |
| 新機能追加時の変更箇所 | 2ファイル | 1ファイル + 共通モジュール |
| テスト対象 | 分散 | 共通モジュールに集約 |
| TS側パターン | 2種類 | 1種類 (Iteratorパターン) |
| 命名の明確さ | 曖昧 | 目的明確 |

---

## リスク・注意点

1. **後方互換性**: 既存 API を即座に削除しない
2. **パフォーマンス**: リファクタリングによる性能劣化を監視
3. **テスト**: 各 Step 完了時に `cargo test` + ブラウザテスト実施
4. **段階的移行**: 一度に全変更せず、Step ごとに検証
