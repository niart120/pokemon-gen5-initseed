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
| `IntegratedSeedSearcher` | `IVBootTimingSearchIterator` | IV確定のための起動時間検索 |
| `IntegratedSearchResult` | `IVBootTimingSearchResult` | IV検索結果 |
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
pub struct IVBootTimingSearchIterator { ... }

#[wasm_bindgen]
pub struct IVBootTimingSearchResult { ... }
```

#### 2. 内部型（Rust内部のみ）
パフォーマンス最適化やロジック共通化のための型。`#[wasm_bindgen]` なし。

```rust
// ハードウェア種別 (String → enum変換)
pub enum HardwareType { DS, DSLite, ThreeDS }

// 時刻範囲設定 (内部表現)
pub struct DailyTimeRangeConfig { ... }

// SHA-1メッセージ構築
pub struct BaseMessageBuilder { ... }

// 日時コード計算
pub struct DateTimeCodeCalculator { ... }

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

## モジュール分割設計（B案: 役割別分離）

### ファイル構成

```
wasm-pkg/src/
├── search_common/
│   ├── mod.rs           # re-export, 定数
│   ├── params.rs        # DSConfig, SegmentParams, TimeRangeParams, SearchRangeParams
│   │                    # DSConfigJs, SegmentParamsJs, TimeRangeParamsJs, SearchRangeParamsJs
│   ├── message.rs       # BaseMessageBuilder, HashValues
│   └── datetime.rs      # DateTimeCodeEnumerator, ユーティリティ関数
├── search_common.rs     # 削除（mod.rsに移行）
└── ...
```

### 依存関係

```
params.rs ← message.rs ← datetime.rs
    ↑           ↑
    └───────────┴─── (外部モジュールから参照)
```

---

## DateTimeCodeEnumerator 仕様

### 概要

`SearchRangeParams` と `TimeRangeParams` と `HardwareType` を受け取り、
有効な日時の `(date_code, time_code)` を順次生成するイテレータ。

### 設計方針

- **Iterator trait 実装**: Rust標準のIteratorパターンに準拠
- **TimeCodeマスク**: 構築時に `TimeRangeParams` から生成（Option<u32>で直接time_codeを保持）
- **遅延評価**: 呼び出しごとに次の有効な日時を計算
- **進捗計算**: スキップした秒も含めて計算

### TimeCodeマスク設計

従来の `build_allowed_second_mask` (bool配列) を廃止し、`Option<u32>` 配列に変更：

```rust
/// TimeCodeマスク（1日分、86400要素）
/// 
/// 各インデックスが1日の秒数（0-86399）に対応。
/// - Some(time_code): 許可された秒、事前計算されたtime_code
/// - None: 許可されていない秒
type TimeCodeMask = Box<[Option<u32>; 86400]>;

fn build_time_code_mask(time_range: &TimeRangeParams, hardware: HardwareType) -> TimeCodeMask {
    let mut mask: TimeCodeMask = Box::new([None; 86400]);
    for hour in time_range.hour_start..=time_range.hour_end {
        for minute in time_range.minute_start..=time_range.minute_end {
            for second in time_range.second_start..=time_range.second_end {
                let second_of_day = hour * 3600 + minute * 60 + second;
                let time_code = TimeCodeGenerator::get_time_code_for_hardware(
                    second_of_day, 
                    hardware.as_str()
                );
                mask[second_of_day as usize] = Some(time_code);
            }
        }
    }
    mask
}
```

**メリット**:
- 許可判定とtime_code取得を1回のルックアップで完了
- time_code計算を事前に済ませ、イテレーション中の計算コスト削減

### インターフェース

```rust
/// 日時コード（イテレータの出力）
#[derive(Debug, Clone, Copy)]
pub struct DateTimeCode {
    pub date_code: u32,
    pub time_code: u32,
}

/// 日時コード列挙器
pub struct DateTimeCodeEnumerator {
    // 検索範囲
    start_seconds: i64,
    end_seconds: i64,
    
    // 現在位置
    current_seconds: i64,
    
    // TimeCodeマスク（1日分、Option<time_code>）
    time_code_mask: Box<[Option<u32>; 86400]>,
}

impl DateTimeCodeEnumerator {
    /// 新規作成
    pub fn new(
        search_range: &SearchRangeParams,
        time_range: &TimeRangeParams,
        hardware: HardwareType,
    ) -> Self;
    
    /// 処理済み秒数（スキップ含む）
    pub fn processed_seconds(&self) -> u32;
    
    /// 残り秒数
    pub fn remaining_seconds(&self) -> u32;
    
    /// 総秒数
    pub fn total_seconds(&self) -> u32;
}

impl Iterator for DateTimeCodeEnumerator {
    type Item = DateTimeCode;
    fn next(&mut self) -> Option<Self::Item>;
}
```

### SIMD バッチ処理対応

Iterator の標準機能 `take(4)` で4要素ずつ取得可能：

```rust
let enumerator = DateTimeCodeEnumerator::new(&search_range, &time_range, hardware);
let batch: Vec<DateTimeCode> = enumerator.by_ref().take(4).collect();

// SIMD処理
if batch.len() == 4 {
    // 4要素揃っている場合はSIMD
    process_simd_batch(&batch);
} else {
    // 端数はスカラー処理
    for dt in batch {
        process_scalar(&dt);
    }
}
```

### 使用例

```rust
let search_range = SearchRangeParams::new(2024, 1, 1, 86400 * 7)?;
let time_range = TimeRangeParams::new(10, 12, 0, 59, 0, 59)?;
let hardware = HardwareType::DS;

let mut enumerator = DateTimeCodeEnumerator::new(&search_range, &time_range, hardware);

while let Some(dt) = enumerator.next() {
    let message = builder.build_message(dt.date_code, dt.time_code);
    // SHA-1計算...
    
    // 進捗報告
    let progress = enumerator.processed_seconds() as f64 / enumerator.total_seconds() as f64;
}
```

---

## リファクタリング方針

### Phase 1: 共通モジュール抽出 ✅ 完了

#### 1.1 `search_common.rs` 実装済み

```rust
//! 検索処理共通モジュール

// ========== 定数 ==========
pub const EPOCH_2000_UNIX: i64 = 946684800;
pub const SECONDS_PER_DAY: i64 = 86_400;
pub const HARDWARE_FRAME_DS: u32 = 8;
pub const HARDWARE_FRAME_DS_LITE: u32 = 6;
pub const HARDWARE_FRAME_3DS: u32 = 9;

// ========== 内部型 ==========
pub enum HardwareType { DS, DSLite, ThreeDS }
pub struct DailyTimeRangeConfig { ... }
pub struct BaseMessageBuilder { ... }
pub struct DateTimeCodeCalculator { ... }
pub struct HashValues { ... }

// ========== 公開Value Object ==========
#[wasm_bindgen]
pub struct DSConfigJs { ... }

#[wasm_bindgen]
pub struct SegmentParamsJs { ... }

#[wasm_bindgen]
pub struct TimeRangeParamsJs { ... }

#[wasm_bindgen]
pub struct SearchRangeParamsJs { ... }
```

### Phase 2: IVBootTimingSearchIterator 実装

#### 目標
`EggBootTimingSearchIterator` と同様のIteratorパターンでIV検索を実装。

#### 設計
```rust
#[wasm_bindgen]
pub struct IVBootTimingSearchIterator {
    // 設定
    ds_config: DSConfigJs,
    segment: SegmentParamsJs,
    time_range: TimeRangeParamsJs,
    search_range: SearchRangeParamsJs,
    
    // 検索条件
    target_seed: u32,
    
    // 状態
    current_offset: u32,
    is_finished: bool,
    
    // 内部キャッシュ
    base_message_builder: BaseMessageBuilder,
    allowed_second_mask: Box<[bool; 86400]>,
}

#[wasm_bindgen]
impl IVBootTimingSearchIterator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        ds_config: &DSConfigJs,
        segment: &SegmentParamsJs,
        time_range: &TimeRangeParamsJs,
        search_range: &SearchRangeParamsJs,
        target_seed: u32,
    ) -> Result<IVBootTimingSearchIterator, String>;

    pub fn next_batch(&mut self, limit: u32, chunk_seconds: u32) -> IVBootTimingSearchResults;
    
    #[wasm_bindgen(getter)]
    pub fn is_finished(&self) -> bool;
    
    #[wasm_bindgen(getter)]
    pub fn processed_count(&self) -> u32;
}
```

### Phase 3: 既存コード移行

1. `integrated_search.rs` は非推奨化（`#[deprecated]`）
2. 新規開発は `IVBootTimingSearchIterator` を使用
3. 段階的に既存呼び出しを移行

---

## ファイル構成

```
wasm-pkg/src/
├── lib.rs
├── search_common.rs           # 定数・型・ユーティリティ ✅
├── iv_boot_timing_search.rs   # IV起動時間検索 (NEW)
├── egg_boot_timing_search.rs  # 孵化乱数検索 (既存、search_common利用へ移行)
├── integrated_search.rs       # 旧IV検索 (deprecated予定)
├── datetime_codes.rs          # 日時コード生成 (既存)
└── ...
```

---

## 実装状況

| Step | 内容 | 状態 |
|------|------|------|
| 1 | ブランチ作成 | ✅ 完了 |
| 2 | `search_common.rs` 内部型実装 | ✅ 完了 |
| 3 | 公開Value Object追加 | ✅ 完了 |
| 4 | テスト (22件) | ✅ 通過 |
| 5 | `IVBootTimingSearchIterator` 実装 | ⬜ 未着手 |
| 6 | 既存コード移行 | ⬜ 未着手 |

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
