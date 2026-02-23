# 孵化乱数起動時間検索機能 設計仕様書

## 1. 概要

### 1.1 目的

BW/BW2 における孵化乱数（タマゴ生成）の起動時間検索機能を実装する。ユーザーが指定した起動日時・SHA-1パラメータ・消費範囲・個体フィルター条件に基づき、条件に合致する個体とその起動条件（時刻/Timer0/VCount/キー入力等）を列挙する。

### 1.2 背景

既存の `IntegratedSeedSearcher` による初期Seed検索機能と `EggSeedEnumerator` による個体列挙機能はそれぞれ独立して実装されている。本機能ではこれらを統合し、「起動時間条件から直接、条件に合う孵化個体を検索」というユースケースを実現する。

### 1.3 要求概要

1. 起動日時範囲・Timer0/VCount範囲・キー入力マスクから LCG Seed を生成
2. 各 LCG Seed に対して EggSeedEnumerator を用いた個体検索を実施
3. フィルター条件に合致する個体と起動条件のペアをリストとして返却
4. 将来的に TypeScript Worker からの並列呼び出しに対応

## 2. アーキテクチャ

### 2.1 コンポーネント構成

```
┌──────────────────────────────────────────────────────────────────┐
│                         TypeScript UI                            │
│   ┌────────────────────────────────────────────────────────────┐ │
│   │          EggBootTimingSearchPanel (新規)                   │ │
│   │   ┌──────────────┬──────────────┬─────────────────┐       │ │
│   │   │ ProfileCard  │ SearchParams │ EggFilterCard   │       │ │
│   │   │  (既存流用)  │   (新規)     │   (既存流用)    │       │ │
│   │   └──────────────┴──────────────┴─────────────────┘       │ │
│   └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│   ┌────────────────────────────────────────────────────────────┐ │
│   │              EggBootTimingWorkerManager                    │ │
│   │    (Worker並列制御 / チャンク分割 / 進捗管理)              │ │
│   └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                        │
│             ┌──────────┐         ┌──────────┐                   │
│             │ Worker 1 │   ...   │ Worker N │                   │
│             └──────────┘         └──────────┘                   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      WASM (Rust)                                 │
│   ┌────────────────────────────────────────────────────────────┐ │
│   │          EggBootTimingSearcher (新規)                      │ │
│   │   - search_eggs_by_boot_timing()                           │ │
│   │   - IntegratedSeedSearcher (SIMD対応LCG Seed生成)          │ │
│   │   - EggSeedEnumerator (個体列挙)                           │ │
│   └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 データフロー

```
入力パラメータ
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. LCG Seed 生成 (日時×Timer0×VCount×キー入力)     │
│    - IntegratedSeedSearcher の SIMD 実装を活用      │
│    - 既存の search_seeds_integrated_simd を参照    │
└─────────────────────────────────────────────────────┘
    │ 各(datetime, timer0, vcount, keyCode)ごとの lcgSeed
    ▼
┌─────────────────────────────────────────────────────┐
│ 2. 個体検索 (各 LCG Seed に対して)                  │
│    - EggSeedEnumerator で advance 範囲を列挙        │
│    - IndividualFilter でフィルタリング              │
└─────────────────────────────────────────────────────┘
    │ フィルター合格した (EnumeratedEggData, BootCondition) ペア
    ▼
┌─────────────────────────────────────────────────────┐
│ 3. 結果集約                                         │
│    - EggBootTimingSearchResult として返却            │
└─────────────────────────────────────────────────────┘
```

## 3. Rust WASM API 設計

### 3.1 新規構造体

#### EggBootTimingSearcher

```rust
/// 孵化乱数起動時間検索器
#[wasm_bindgen]
pub struct EggBootTimingSearcher {
    /// 内部で使用する IntegratedSeedSearcher 相当のパラメータ
    base_message: [u32; 16],
    key_codes: Vec<u32>,
    allowed_second_mask: Box<[bool; 86400]>,
    hardware: String,
    
    /// 孵化固有パラメータ
    conditions: GenerationConditions,
    parents: ParentsIVs,
    filter: Option<IndividualFilter>,
    consider_npc_consumption: bool,
    game_mode: GameMode,
    
    /// 検索範囲
    user_offset: u64,
    advance_count: u32,
}
```

#### EggBootTimingSearchResult

```rust
/// 検索結果1件
#[wasm_bindgen]
pub struct EggBootTimingSearchResult {
    /// 起動条件
    pub year: u32,
    pub month: u32,
    pub date: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub timer0: u32,
    pub vcount: u32,
    pub key_code: u32,
    
    /// LCG Seed (SHA-1結果)
    pub lcg_seed: u64,
    
    /// 個体情報
    pub advance: u64,
    pub is_stable: bool,
    // ResolvedEgg の各フィールド
    pub ivs: [u8; 6],
    pub nature: u8,
    pub gender: u8,
    pub ability: u8,
    pub shiny: u8,
    pub pid: u32,
    // めざめるパワー
    pub hp_type: u8,
    pub hp_power: u8,
    pub hp_known: bool,
}
```

### 3.2 コンストラクタ

```rust
#[wasm_bindgen]
impl EggBootTimingSearcher {
    /// コンストラクタ
    #[wasm_bindgen(constructor)]
    pub fn new(
        // SHA-1 パラメータ (IntegratedSeedSearcher と共通)
        mac: &[u8],           // MACアドレス 6バイト
        nazo: &[u32],         // nazo値 5要素
        hardware: &str,       // "DS" | "DS_LITE" | "3DS"
        key_input_mask: u32,  // キー入力マスク
        frame: u32,           // フレーム (通常8)
        
        // 時刻範囲 (IntegratedSeedSearcher と共通)
        hour_start: u32, hour_end: u32,
        minute_start: u32, minute_end: u32,
        second_start: u32, second_end: u32,
        
        // 孵化条件パラメータ
        conditions: &GenerationConditionsJs,
        parents: &ParentsIVsJs,
        filter: Option<IndividualFilterJs>,
        consider_npc_consumption: bool,
        game_mode: GameMode,
        
        // 消費範囲
        user_offset: u64,
        advance_count: u32,
    ) -> Result<EggBootTimingSearcher, JsValue>;
}
```

### 3.3 検索メソッド

```rust
#[wasm_bindgen]
impl EggBootTimingSearcher {
    /// メイン検索関数
    pub fn search_eggs_integrated(
        &self,
        // 日時・パラメータ範囲
        year_start: u32, month_start: u32, date_start: u32,
        hour_start: u32, minute_start: u32, second_start: u32,
        range_seconds: u32,
        timer0_min: u32, timer0_max: u32,
        vcount_min: u32, vcount_max: u32,
    ) -> js_sys::Array;  // EggBootTimingSearchResult の配列

    /// SIMD最適化版（優先使用）
    pub fn search_eggs_integrated_simd(
        &self,
        year_start: u32, month_start: u32, date_start: u32,
        hour_start: u32, minute_start: u32, second_start: u32,
        range_seconds: u32,
        timer0_min: u32, timer0_max: u32,
        vcount_min: u32, vcount_max: u32,
    ) -> js_sys::Array;
}
```

### 3.4 内部処理フロー

```rust
// 疑似コード
fn search_eggs_integrated_simd(&self, ...) -> js_sys::Array {
    let results = js_sys::Array::new();
    
    // 1. 日時・Timer0・VCount の全組み合わせをループ
    for timer0 in timer0_min..=timer0_max {
        for vcount in vcount_min..=vcount_max {
            for &key_code in &self.key_codes {
                // SIMD バッチ処理用バッファ
                let mut messages = [0u32; 64];
                let mut batch_metadata = Vec::new();
                
                for second_offset in 0..range_seconds {
                    // 日時が許可範囲内かチェック
                    let (time_code, date_code) = self.calculate_datetime_codes(...)?;
                    
                    // SHA-1 メッセージ構築
                    let message = self.build_message(timer0, vcount, date_code, time_code, key_code);
                    
                    // バッチに追加
                    messages[batch_idx..].copy_from_slice(&message);
                    batch_metadata.push((second_offset, timer0, vcount, key_code, datetime));
                    
                    // 4件溜まったらSIMD処理
                    if batch_len == 4 {
                        self.process_simd_batch(&messages, &batch_metadata, &results);
                        batch_len = 0;
                    }
                }
                
                // 残りを処理
                if batch_len > 0 {
                    self.process_remaining_batch(&messages, &batch_metadata, batch_len, &results);
                }
            }
        }
    }
    
    results
}

fn process_simd_batch(&self, messages: &[u32; 64], metadata: &[...], results: &js_sys::Array) {
    // SIMD SHA-1 計算
    let hash_results = calculate_pokemon_sha1_simd(messages);
    
    for i in 0..4 {
        let lcg_seed = calculate_lcg_seed_from_hash(hash_results[i*5], hash_results[i*5+1]);
        
        // 2. 各 LCG Seed に対して EggSeedEnumerator を実行
        let mut enumerator = EggSeedEnumerator::new(
            lcg_seed,
            self.user_offset,
            self.advance_count,
            self.conditions.clone(),
            self.parents.clone(),
            self.filter.clone(),
            self.consider_npc_consumption,
            self.game_mode,
        );
        
        // 3. フィルター適用済みの結果を収集
        while let Ok(Some(egg_data)) = enumerator.next_egg() {
            let result = EggBootTimingSearchResult {
                // 起動条件
                year: metadata[i].datetime.year,
                month: metadata[i].datetime.month,
                // ... その他の起動条件
                
                // LCG Seed
                lcg_seed,
                
                // 個体情報
                advance: egg_data.advance,
                is_stable: egg_data.is_stable,
                ivs: egg_data.egg.ivs,
                // ... その他の個体情報
            };
            results.push(&result.into());
        }
    }
}
```

## 4. TypeScript 型定義

### 4.1 検索パラメータ

```typescript
// src/types/egg-boot-timing-search.ts

/**
 * 孵化乱数起動時間検索パラメータ
 */
export interface EggBootTimingSearchParams {
  // === SHA-1 / 起動時間パラメータ ===
  /** 開始日時 (ISO8601 UTC) */
  startDatetimeIso: string;
  /** 検索範囲秒数 */
  rangeSeconds: number;
  
  /** Timer0範囲 */
  timer0Range: { min: number; max: number };
  /** VCount範囲 */
  vcountRange: { min: number; max: number };
  
  /** キー入力マスク */
  keyInputMask: number;
  /** MACアドレス */
  macAddress: readonly [number, number, number, number, number, number];
  /** ハードウェア */
  hardware: Hardware;
  /** ROMバージョン */
  romVersion: ROMVersion;
  /** ROM地域 */
  romRegion: ROMRegion;
  
  /** 時刻範囲フィルター */
  timeRange: DailyTimeRange;
  
  // === 孵化条件パラメータ ===
  /** 生成条件 */
  conditions: EggGenerationConditions;
  /** 親個体値 */
  parents: ParentsIVs;
  /** 個体フィルター */
  filter: EggIndividualFilter | null;
  /** NPC消費考慮 */
  considerNpcConsumption: boolean;
  /** ゲームモード */
  gameMode: EggGameMode;
  
  // === 消費範囲 ===
  /** 開始advance */
  userOffset: number;
  /** 検索件数上限 */
  advanceCount: number;
}
```

### 4.2 検索結果

```typescript
/**
 * 起動条件情報
 */
export interface BootCondition {
  datetime: Date;
  timer0: number;
  vcount: number;
  keyCode: number;
  keyInputNames: KeyName[];
  macAddress: readonly [number, number, number, number, number, number];
}

/**
 * 孵化乱数起動時間検索結果
 */
export interface EggBootTimingSearchResult {
  /** 起動条件 */
  boot: BootCondition;
  
  /** LCG Seed (hex string) */
  lcgSeedHex: string;
  
  /** 個体情報 (既存のEnumeratedEggData を拡張) */
  egg: EnumeratedEggDataWithBootTiming;
}

/**
 * Boot-Timing 拡張付き個体データ
 * (既存の EnumeratedEggDataWithBootTiming を流用)
 */
export interface EnumeratedEggDataWithBootTiming extends EnumeratedEggData {
  seedSourceMode: 'boot-timing';
  timer0: number;
  vcount: number;
  bootTimestampIso: string;
  keyInputNames: KeyName[];
  macAddress: readonly [number, number, number, number, number, number];
}
```

### 4.3 Worker 通信

```typescript
/**
 * Worker リクエスト
 */
export type EggBootTimingWorkerRequest =
  | { type: 'START_SEARCH'; params: EggBootTimingSearchParams; requestId?: string }
  | { type: 'STOP'; requestId?: string };

/**
 * Worker レスポンス
 */
export type EggBootTimingWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; processed: number; total: number; found: number }
  | { type: 'RESULTS'; results: EggBootTimingSearchResult[] }
  | { type: 'COMPLETE'; payload: EggBootTimingCompletion }
  | { type: 'ERROR'; message: string; category: string; fatal: boolean };

/**
 * 完了情報
 */
export interface EggBootTimingCompletion {
  reason: 'completed' | 'stopped' | 'max-results' | 'error';
  processedCombinations: number;  // 処理した起動条件の組み合わせ数
  totalCombinations: number;      // 総組み合わせ数
  resultsCount: number;           // 見つかった結果数
  elapsedMs: number;              // 経過時間
}
```

## 5. 既存コードとの統合

### 5.1 流用する型・関数

| カテゴリ | 流用元 | 用途 |
|----------|--------|------|
| 起動条件 | `IntegratedSeedSearcher` | LCG Seed 生成、SIMD最適化 |
| 個体列挙 | `EggSeedEnumerator` | 個体検索、フィルタリング |
| 条件入力 | `GenerationConditionsJs`, `ParentsIVsJs` | 孵化条件の受け渡し |
| フィルター | `IndividualFilterJs` | 個体フィルター |
| ゲームモード | `GameMode` | オフセット計算 |
| 日時計算 | `DateCodeGenerator`, `TimeCodeGenerator` | SHA-1 メッセージ生成 |

### 5.2 新規実装が必要な部分

1. **Rust側**
   - `EggBootTimingSearcher` 構造体
   - `EggBootTimingSearchResult` 構造体
   - `search_eggs_integrated_simd` メソッド

2. **TypeScript側**
   - `EggBootTimingSearchParams` 型
   - `EggBootTimingSearchResult` 型
   - Worker通信型
   - UIコンポーネント（将来）

## 6. パフォーマンス考慮事項

### 6.1 計算量見積もり

```
総計算量 = rangeSeconds × timer0Count × vcountCount × keyCodeCount × advanceCount

例:
- rangeSeconds = 3600 (1時間)
- timer0Count = 3 (0xC79-0xC7B)
- vcountCount = 1 (0x60)
- keyCodeCount = 16 (4キー)
- advanceCount = 1000

総計算量 = 3600 × 3 × 1 × 16 × 1000 = 172,800,000 個体

→ Worker並列化 + SIMDで高速化が必要
```

### 6.2 最適化戦略

1. **SIMD活用**: SHA-1計算の4並列処理
2. **早期フィルタリング**: EggSeedEnumerator内でフィルター適用
3. **Worker並列化**: 日時範囲を分割して複数Workerで処理
4. **進捗報告**: 定期的な進捗更新でUX改善

### 6.3 メモリ管理

- 結果配列は逐次的に返却（バッチ送信）
- 大量結果時は上限設定（maxResults）
- 不要なデータはスコープ外で解放

## 7. 将来の拡張性

### 7.1 BW2対応

- `GameMode` による分岐は既存実装で対応済み
- BW2固有の孵化ロジック変更がある場合は `EggSeedEnumerator` を拡張

### 7.2 UI統合

```
予定されるUIコンポーネント構成:

EggBootTimingSearchPanel
├── ProfileCard (既存)
│   └── DeviceProfile表示・選択
├── EggSearchParamsCard (新規)
│   ├── 起動日時入力
│   ├── Timer0/VCount範囲表示
│   └── キー入力選択
├── EggParamsCard (既存流用)
│   ├── 親個体値入力
│   └── 生成条件入力
├── EggFilterCard (既存流用)
│   └── 個体フィルター
├── EggSearchRunCard (新規)
│   ├── 検索ボタン
│   └── 進捗表示
└── EggSearchResultsCard (新規)
    └── 結果テーブル
```

### 7.3 エクスポート

- 既存の `GenerationExportButton` パターンを流用
- CSV/JSON/TXT形式対応
- 起動条件情報を含めたメタデータ出力

## 8. テスト計画

### 8.1 Rustユニットテスト

```rust
#[cfg(test)]
mod tests {
    // 1. コンストラクタバリデーション
    #[test]
    fn test_constructor_validation() { ... }
    
    // 2. 既知の起動条件からの個体検索
    #[test]
    fn test_search_known_conditions() { ... }
    
    // 3. フィルター適用
    #[test]
    fn test_filter_application() { ... }
    
    // 4. SIMD vs スカラーの結果一致
    #[test]
    fn test_simd_scalar_consistency() { ... }
}
```

### 8.2 TypeScript統合テスト

- WASMローダー初期化テスト
- Worker通信テスト
- 結果型変換テスト

### 8.3 E2Eテスト

- 既知条件での検索結果検証
- パフォーマンスベンチマーク

## 9. 実装フェーズ

### Phase 1: Rust WASM実装

1. `EggBootTimingSearcher` 構造体定義
2. `EggBootTimingSearchResult` 構造体定義
3. `search_eggs_integrated` 実装
4. `search_eggs_integrated_simd` 実装
5. ユニットテスト

### Phase 2: TypeScript統合

1. 型定義追加
2. Worker実装
3. WorkerManager実装
4. 統合テスト

### Phase 3: UI実装（今回のスコープ外）

1. EggBootTimingSearchPanel
2. 結果表示
3. エクスポート機能

## 10. 関連ドキュメント

- `/spec/implementation/egg-seed-enumerator.md` - EggSeedEnumerator仕様
- `/spec/generation-boot-timing-mode.md` - Generation Boot-Timing Mode仕様
- `/spec/implementation/05-webgpu-seed-search.md` - WebGPU Seed検索仕様
