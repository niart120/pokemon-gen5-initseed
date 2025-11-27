# Rust WASM 処理共通化リファクタリング方針

## 対象ファイル
- `wasm-pkg/src/integrated_search.rs` - 初期Seed検索 (32bit MT Seed)
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

### 2. アーキテクチャ上の問題
- `integrated_search.rs`: timer0/vcount/keyCodeのループをRust内で実行
- `egg_boot_timing_search.rs`: timer0/vcount/keyCodeはTS側から固定値で渡される (Iterator パターン)
- ループ構造の違いにより、完全な共通化が困難

### 3. 責務の混在
- SHA-1メッセージ構築 + 日時コード生成 + 検索ロジックが1ファイルに混在

---

## リファクタリング方針

### Phase 1: 共通モジュール抽出

#### 1.1 新規ファイル: `search_common.rs`

```rust
//! 検索処理共通モジュール

// ========== 定数 ==========
pub const EPOCH_2000_UNIX: i64 = 946684800;
pub const SECONDS_PER_DAY: i64 = 86_400;

// ========== Hardware関連 ==========
pub const HARDWARE_FRAME_DS: u32 = 8;
pub const HARDWARE_FRAME_DS_LITE: u32 = 6;
pub const HARDWARE_FRAME_3DS: u32 = 9;

pub fn get_frame_for_hardware(hardware: &str) -> Result<u32, JsValue>;
pub fn validate_hardware(hardware: &str) -> Result<(), JsValue>;

// ========== 日時範囲設定 ==========
pub struct DailyTimeRangeConfig { ... }
impl DailyTimeRangeConfig {
    pub fn new(...) -> Result<Self, JsValue>;
    pub fn combos_per_day(&self) -> u32;
}

pub fn build_allowed_second_mask(range: &DailyTimeRangeConfig) -> Box<[bool; 86400]>;

// ========== 基本メッセージ構築 ==========
pub struct BaseMessageBuilder { ... }
impl BaseMessageBuilder {
    pub fn new(mac: &[u8], nazo: &[u32], frame: u32) -> Result<Self, JsValue>;
    pub fn base_message(&self) -> &[u32; 16];
}
```

#### 1.2 新規ファイル: `search_segment.rs`

```rust
//! セグメント検索基盤

/// 固定セグメントパラメータ (timer0, vcount, key_code)
#[derive(Clone, Copy)]
pub struct SegmentParams {
    pub timer0: u32,
    pub vcount: u32,
    pub key_code: u32,
}

/// 秒単位イテレータ (gen キーワード使用検討)
pub struct SecondsIterator {
    base_seconds_since_2000: i64,
    range_seconds: u32,
    current_offset: u32,
    allowed_second_mask: Box<[bool; 86400]>,
    hardware: String,
}

impl SecondsIterator {
    /// 次の有効な秒とその日時コードを返す
    pub fn next(&mut self) -> Option<(i64, u32, u32)>; // (seconds_since_2000, time_code, date_code)
}

/// SIMD バッチ処理トレイト
pub trait SimdBatchProcessor {
    type Output;
    fn process_batch(&self, messages: &[u32; 64], batch_size: usize) -> Vec<Self::Output>;
}
```

### Phase 2: セグメント固定化 (TS側からの渡し方統一)

#### 現状
- `IntegratedSeedSearcher`: Rust内でtimer0/vcount/keyCodeをループ
- `EggBootTimingSearchIterator`: TS側からtimer0/vcount/keyCodeを固定値で渡す

#### 目標
両方とも `EggBootTimingSearchIterator` パターン（TS側セグメントループ）に統一：

```typescript
// TypeScript側 (共通パターン)
const keyCodes = wasm.generate_key_codes(keyInputMask);
for (const timer0 of range(timer0Min, timer0Max)) {
  for (const vcount of range(vcountMin, vcountMax)) {
    for (const keyCode of keyCodes) {
      const iterator = new SearchIterator(..., timer0, vcount, keyCode);
      while (!iterator.isFinished) {
        const results = iterator.next_batch(limit, chunkSeconds);
        // 結果処理・進捗報告
      }
      iterator.free();
    }
  }
}
```

#### 利点
- 進捗報告の粒度統一
- キャンセル処理の共通化
- Worker並列化パターンの統一

### Phase 3: gen キーワードによる Iterator 実装 (Rust nightly)

#### 検討事項
- Rust nightly の `gen` キーワードでコルーチン的な実装が可能
- ただし stable Rust では使用不可
- 代替: `Iterator` trait の手動実装

#### 採用判断
- 現時点では `Iterator` trait の手動実装を採用
- nightly 依存は避ける

### Phase 4: 処理パイプラインの分離

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│ SecondsIterator │ -> │ SHA1BatchProcessor │ -> │ ResultCollector │
│ (日時コード生成) │    │ (SIMD/Scalar)     │    │ (検索固有処理)  │
└─────────────────┘    └────────────────────┘    └─────────────────┘
                                                         ↓
                                                 ┌───────────────────┐
                                                 │ IntegratedSearch  │
                                                 │ (Seed照合)        │
                                                 └───────────────────┘
                                                         or
                                                 ┌───────────────────┐
                                                 │ EggBootTiming     │
                                                 │ (個体列挙)        │
                                                 └───────────────────┘
```

---

## 実装順序

### Step 1: 共通モジュール抽出
1. `search_common.rs` 作成
2. 定数・型・関数を移動
3. 既存ファイルから `use crate::search_common::*;` で参照

### Step 2: BaseMessageBuilder 導入
1. base_message 構築ロジックを構造体化
2. 既存の構築ロジックを置き換え

### Step 3: SecondsIterator 導入
1. 秒単位イテレータを独立モジュール化
2. 日時コード生成処理を統合

### Step 4: IntegratedSeedSearcher の Iterator 化
1. TS側セグメントループパターンに移行
2. `IntegratedSeedSearchIterator` 作成
3. 既存 API は deprecated 扱い（後方互換性維持）

### Step 5: SIMD バッチ処理の共通化
1. `SimdBatchProcessor` トレイト定義
2. 共通の SIMD/Scalar フォールバック実装

---

## 新規ファイル構成

```
wasm-pkg/src/
├── lib.rs
├── search_common.rs      # 定数・型・ユーティリティ (NEW)
├── search_segment.rs     # セグメント・イテレータ基盤 (NEW)
├── search_simd.rs        # SIMD バッチ処理共通化 (NEW)
├── integrated_search.rs  # 初期Seed検索 (リファクタ)
├── egg_boot_timing_search.rs  # 孵化乱数検索 (リファクタ)
├── datetime_codes.rs     # 既存（変更なし）
└── ...
```

---

## 期待効果

| 項目 | Before | After |
|------|--------|-------|
| 重複コード行数 | 約300行 | 約50行 |
| 新機能追加時の変更箇所 | 2ファイル | 1ファイル + 共通モジュール |
| テスト対象 | 分散 | 共通モジュールに集約 |
| TS側パターン | 2種類 | 1種類 |

---

## リスク・注意点

1. **後方互換性**: 既存 API を即座に削除しない
2. **パフォーマンス**: リファクタリングによる性能劣化を監視
3. **テスト**: 各 Step 完了時に `cargo test` + ブラウザテスト実施
4. **段階的移行**: 一度に全変更せず、Step ごとに検証

---

## 次のアクション

1. ✅ ブランチ作成: `refactor/rust-wasm-common-processing`
2. ⬜ Step 1 実装: `search_common.rs` 作成
3. ⬜ 既存テスト通過確認
4. ⬜ Step 2 以降の実装
