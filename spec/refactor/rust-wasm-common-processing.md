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
