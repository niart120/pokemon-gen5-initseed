# EggBWPanel アーキテクチャ図

## 1. システム全体図

```
┌─────────────────────────────────────────────────────────────┐
│                        EggBWPanel                            │
│                      (React Component)                       │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─────────────────┐
           │                 │
           ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│  EggParamsCard   │  │  EggFilterCard   │
│ (パラメータ入力) │  │  (フィルター設定)│
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  egg-store.ts   │  ← Zustand State Management
         │  (Zustand)      │
         └────────┬────────┘
                  │
                  │ startGeneration()
                  ▼
         ┌─────────────────────┐
         │ EggWorkerManager    │  ← Worker Lifecycle Manager
         │ .start()            │
         │ .onResults()        │
         │ .onComplete()       │
         └────────┬────────────┘
                  │
                  │ postMessage()
                  ▼
         ┌─────────────────────┐
         │   egg-worker.ts     │  ← Web Worker (Background)
         │   (Worker Thread)   │
         └────────┬────────────┘
                  │
                  │ new EggSeedEnumerator()
                  ▼
         ┌─────────────────────┐
         │  EggSeedEnumerator  │  ← WASM Module (Rust)
         │     (WASM/Rust)     │
         │  .next_egg()        │
         └─────────────────────┘
```

## 2. データフロー図

### 2.1 パラメータ入力からWorker起動まで

```
[User Input]
     │
     ▼
┌──────────────────────────────────┐
│ EggParamsCard.tsx                │
│ - baseSeedHex: string            │
│ - parents: ParentsIVs            │
│ - conditions: EggGenConditions   │
└────────┬─────────────────────────┘
         │ updateDraftParams()
         ▼
┌──────────────────────────────────┐
│ egg-store.ts (draftParams)       │
│ - EggGenerationParamsHex         │
└────────┬─────────────────────────┘
         │ validateDraft()
         ▼
┌──────────────────────────────────┐
│ hexParamsToEggParams()           │
│ - BigInt conversion              │
│ - validateEggParams()            │
└────────┬─────────────────────────┘
         │ if valid
         ▼
┌──────────────────────────────────┐
│ egg-store.ts (params)            │
│ - EggGenerationParams            │
└────────┬─────────────────────────┘
         │ startGeneration()
         ▼
┌──────────────────────────────────┐
│ EggWorkerManager.start()         │
│ - Worker 作成/取得               │
│ - postMessage(START_GENERATION)  │
└──────────────────────────────────┘
```

### 2.2 Worker実行と結果受信

```
┌──────────────────────────────────┐
│ egg-worker.ts                    │
│ handleStart(params)              │
└────────┬─────────────────────────┘
         │
         ├─ initWasm()
         │
         ├─ new ParentsIVs()
         │
         ├─ new GenerationConditions()
         │
         ├─ new IndividualFilter() (optional)
         │
         └─ new EggSeedEnumerator(...)
                  │
                  ▼
         ┌────────────────────┐
         │ Loop: next_egg()   │
         └────────┬───────────┘
                  │
                  ├─ parseEnumeratedEggData()
                  │
                  ├─ results.push(eggData)
                  │
                  └─ postMessage(RESULTS)
                           │
                           ▼
         ┌─────────────────────────────┐
         │ EggWorkerManager            │
         │ handleMessage(RESULTS)      │
         └────────┬────────────────────┘
                  │
                  │ callbacks.results.forEach()
                  ▼
         ┌─────────────────────────────┐
         │ egg-store.ts                │
         │ onResults: (payload) => {   │
         │   results.push(...payload)  │
         │ }                            │
         └────────┬────────────────────┘
                  │
                  ▼
         ┌─────────────────────────────┐
         │ EggResultsCard.tsx          │
         │ - Display results table     │
         └─────────────────────────────┘
```

## 3. コンポーネント詳細図

### 3.1 EggBWPanel レイアウト

```
┌─────────────────────────────────────────────────────────────┐
│                      EggBWPanel                              │
│  ┌─────────────────────┐  ┌──────────────────────────────┐ │
│  │   Left Column       │  │      Right Column             │ │
│  │                     │  │                               │ │
│  │  ┌───────────────┐  │  │  ┌────────────────────────┐  │ │
│  │  │ EggRunCard    │  │  │  │  EggResultsCard        │  │ │
│  │  │ - Start/Stop  │  │  │  │  ┌──────────────────┐  │  │ │
│  │  │ - Status      │  │  │  │  │ Results Table    │  │  │ │
│  │  │ - Progress    │  │  │  │  │ - Advance        │  │  │ │
│  │  └───────────────┘  │  │  │  │ - IVs (6 cols)   │  │  │ │
│  │                     │  │  │  │ - Nature         │  │  │ │
│  │  ┌───────────────┐  │  │  │  │ - Gender         │  │  │ │
│  │  │ EggParamsCard │  │  │  │  │ - Ability        │  │  │ │
│  │  │ ┌───────────┐ │  │  │  │  │ - Shiny          │  │  │ │
│  │  │ │Basic      │ │  │  │  │  │ - Hidden Power   │  │  │ │
│  │  │ │- Seed     │ │  │  │  │  │ - PID            │  │  │ │
│  │  │ │- Offset   │ │  │  │  │  │ - Stability      │  │  │ │
│  │  │ │- Count    │ │  │  │  │  └──────────────────┘  │  │ │
│  │  │ └───────────┘ │  │  │  │                        │  │ │
│  │  │ ┌───────────┐ │  │  │  │  (Scroll if needed)    │  │ │
│  │  │ │Parents IV │ │  │  │  │                        │  │ │
│  │  │ │- Male     │ │  │  │  └────────────────────────┘  │ │
│  │  │ │- Female   │ │  │  │                               │ │
│  │  │ └───────────┘ │  │  └───────────────────────────────┘ │
│  │  │ ┌───────────┐ │  │                                    │
│  │  │ │Conditions │ │  │                                    │
│  │  │ │- Everstone│ │  │                                    │
│  │  │ │- Gender   │ │  │                                    │
│  │  │ │- etc.     │ │  │                                    │
│  │  │ └───────────┘ │  │                                    │
│  │  └───────────────┘  │                                    │
│  │                     │                                    │
│  │  ┌───────────────┐  │                                    │
│  │  │ EggFilterCard │  │                                    │
│  │  │ - IV Ranges   │  │                                    │
│  │  │ - Nature      │  │                                    │
│  │  │ - Gender      │  │                                    │
│  │  │ - Ability     │  │                                    │
│  │  │ - Shiny       │  │                                    │
│  │  │ - Hidden Power│  │                                    │
│  │  └───────────────┘  │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 状態管理構造 (egg-store)

```
┌─────────────────────────────────────────┐
│           egg-store (Zustand)           │
├─────────────────────────────────────────┤
│ State:                                  │
│  - draftParams: EggGenerationParamsHex  │
│  - params: EggGenerationParams | null   │
│  - validationErrors: string[]           │
│  - status: EggStatus                    │
│  - workerManager: EggWorkerManager      │
│  - results: EnumeratedEggData[]         │
│  - lastCompletion: EggCompletion        │
│  - errorMessage: string | null          │
├─────────────────────────────────────────┤
│ Actions:                                │
│  - updateDraftParams()                  │
│  - validateDraft()                      │
│  - startGeneration()                    │
│  - stopGeneration()                     │
│  - clearResults()                       │
│  - reset()                              │
└─────────────────────────────────────────┘
```

## 4. Worker通信プロトコル

### 4.1 メッセージフロー

```
UI Thread                          Worker Thread
    |                                   |
    |--- START_GENERATION ------------->|
    |    {                               |
    |      type: 'START_GENERATION',    |
    |      params: EggGenerationParams  |
    |    }                              |
    |                                   |
    |<--------- READY ------------------|
    |    { type: 'READY', version: '1' }|
    |                                   |
    |                                [WASM Init]
    |                                   |
    |                             [Enumeration Loop]
    |                                   |
    |<--------- RESULTS ----------------|
    |    {                              |
    |      type: 'RESULTS',             |
    |      payload: {                   |
    |        results: [                 |
    |          { advance, egg, ... }    |
    |        ]                          |
    |      }                            |
    |    }                              |
    |                                   |
    |<--------- RESULTS ----------------|
    |    (multiple times)               |
    |                                   |
    |<--------- COMPLETE ---------------|
    |    {                              |
    |      type: 'COMPLETE',            |
    |      payload: {                   |
    |        reason: 'max-count',       |
    |        processedCount: 1000,      |
    |        filteredCount: 150,        |
    |        elapsedMs: 5432            |
    |      }                            |
    |    }                              |
    |                                   |
    |--- STOP ----------------------->  |
    |    { type: 'STOP' }               |
    |                                   |
    |<--------- COMPLETE ---------------|
    |    {                              |
    |      type: 'COMPLETE',            |
    |      payload: {                   |
    |        reason: 'stopped', ...     |
    |      }                            |
    |    }                              |
```

### 4.2 エラーハンドリングフロー

```
Worker Thread                      UI Thread
    |                                   |
    |--- ERROR ------------------------>|
    |    {                              |
    |      type: 'ERROR',               |
    |      message: string,             |
    |      category: 'VALIDATION' |     |
    |                 'WASM_INIT' |     |
    |                 'RUNTIME',        |
    |      fatal: boolean               |
    |    }                              |
    |                                   |
    |                             [Handle Error]
    |                                   |
    |                          [If fatal: Terminate]
```

## 5. WASM連携詳細

### 5.1 WASM オブジェクト構築フロー

```
egg-worker.ts
     │
     ├─ getWasm()
     │     │
     │     └─ wasm_pkg module
     │
     ├─ new wasm.ParentsIVs()
     │     └─ { male: IvSet, female: IvSet }
     │
     ├─ new wasm.GenerationConditions()
     │     ├─ has_nidoran_flag: boolean
     │     ├─ everstone: EverstonePlan
     │     ├─ uses_ditto: boolean
     │     ├─ allow_hidden_ability: boolean
     │     ├─ female_parent_has_hidden: boolean
     │     ├─ reroll_count: number
     │     ├─ trainer_ids: TrainerIds
     │     └─ gender_ratio: GenderRatio
     │
     ├─ new wasm.IndividualFilter() (optional)
     │     ├─ iv_ranges: [StatRange; 6]
     │     ├─ nature: Option<number>
     │     ├─ gender: Option<number>
     │     ├─ ability: Option<number>
     │     ├─ shiny: Option<number>
     │     ├─ hidden_power_type: Option<number>
     │     └─ hidden_power_power: Option<number>
     │
     └─ new wasm.EggSeedEnumerator()
           ├─ base_seed: u64 (BigInt)
           ├─ user_offset: u64 (BigInt)
           ├─ count: u32
           ├─ conditions: GenerationConditions
           ├─ parents: ParentsIVs
           ├─ filter: Option<IndividualFilter>
           ├─ consider_npc_consumption: boolean
           └─ game_mode: GameMode (number)
                 │
                 └─ .next_egg() → EnumeratedEggData | null
```

### 5.2 メモリ管理

```
try {
  const parents = new wasm.ParentsIVs();
  const conditions = new wasm.GenerationConditions();
  const filter = new wasm.IndividualFilter();
  const enumerator = new wasm.EggSeedEnumerator(...);
  
  // ... 使用
  
} finally {
  // 必ず解放
  enumerator.free();
  filter.free();
  conditions.free();
  parents.free();
}
```

## 6. パフォーマンス最適化ポイント

### 6.1 バッチ処理

```
const BATCH_SIZE = 100;
const batch: EnumeratedEggData[] = [];

while (true) {
  const data = enumerator.next_egg();
  if (!data) break;
  
  batch.push(parseData(data));
  
  if (batch.length >= BATCH_SIZE) {
    postResults(batch);  // ← バッチ送信
    batch.length = 0;
  }
}

if (batch.length > 0) {
  postResults(batch);
}
```

### 6.2 結果数制限

```
// Store内で結果数を制限
onResults((payload) => {
  set((state) => {
    const newResults = [...state.results, ...payload.results];
    if (newResults.length > MAX_DISPLAY) {
      return {
        results: newResults.slice(-MAX_DISPLAY)
      };
    }
    return { results: newResults };
  });
});
```

## 7. テストアーキテクチャ

```
┌────────────────────────────────────────┐
│         Test Pyramid                   │
├────────────────────────────────────────┤
│  E2E Tests (Playwright)                │
│  - UI操作フロー                        │
│  - 結果表示確認                        │
├────────────────────────────────────────┤
│  Integration Tests (Vitest)            │
│  - Worker + WASM 統合                  │
│  - WorkerManager + Store 統合          │
├────────────────────────────────────────┤
│  Unit Tests (Vitest)                   │
│  - 型バリデーション                    │
│  - WorkerManager ロジック              │
│  - Store アクション                    │
│  - コンポーネント単体                  │
└────────────────────────────────────────┘
```

## 8. セキュリティと制約

### 8.1 入力バリデーション

```
validateEggParams():
  - count: 1 ≤ count ≤ 100000
  - rerollCount: 0 ≤ rerollCount ≤ 5
  - tid: 0 ≤ tid ≤ 65535
  - sid: 0 ≤ sid ≤ 65535
  - IV values: 0 ≤ iv ≤ 32
```

### 8.2 メモリ制約

```
MAX_DISPLAY_RESULTS = 10000
- 超過分は古い結果から削除
- Worker側でもバッチサイズ制限
```

## 9. 拡張ポイント

### 9.1 将来的な機能追加

```
┌────────────────────────────────┐
│ Mode Switcher (Future)         │
├────────────────────────────────┤
│ [ 個体一覧表示 ]  [ 起動時間検索 ] │
│      (Current)      (WIP)      │
└────────────────────────────────┘
```

### 9.2 エクスポート機能 (予定)

```
EggResultsCard
  └─ Export Button
       ├─ CSV
       ├─ JSON
       └─ Clipboard Copy
```

## 10. IV入力のUI設計

### 10.1 親個体IV入力 (0-31 + Unknown)

```
┌───────────────────────────────────────────────────────────┐
│ ♂親 個体値                                                 │
├───────────────────────────────────────────────────────────┤
│  HP   Atk   Def   SpA   SpD   Spe                         │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                       │
│ │31 │ │31 │ │31 │ │31 │ │31 │ │31 │  ← 数値入力 (0-31)    │
│ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘                       │
│  [ ]   [ ]   [ ]   [✓]   [ ]   [ ]   ← 不明チェックボックス │
│                     ↑                                      │
│              チェック時は入力無効化、値=32                   │
└───────────────────────────────────────────────────────────┘
```

### 10.2 フィルターIV範囲入力

```
┌───────────────────────────────────────────────────────────┐
│ フィルター: IV範囲                                          │
├───────────────────────────────────────────────────────────┤
│  HP:   [ 0 ]━━━━━━━━━━━━━━━━━━[ 31 ]   [ ] 不明を含む     │
│  Atk:  [ 0 ]━━━━━━━━━━━━━━━━━━[ 31 ]   [✓] 不明を含む     │
│  Def:  [ 0 ]━━━━━━━━━━━━━━━━━━[ 31 ]   [ ] 不明を含む     │
│  SpA:  [ 0 ]━━━━━━━━━━━━━━━━━━[ 31 ]   [ ] 不明を含む     │
│  SpD:  [ 0 ]━━━━━━━━━━━━━━━━━━[ 31 ]   [ ] 不明を含む     │
│  Spe:  [ 0 ]━━━━━━━━━━━━━━━━━━[ 31 ]   [ ] 不明を含む     │
│                                         ↑                  │
│                チェック時は max=32 に強制                   │
└───────────────────────────────────────────────────────────┘
```

## 11. 起動時間関連機能のアーキテクチャ

### 11.1 起動時間列挙モード（Boot Timing Enumeration）

モード切替と処理フロー:

```
┌─────────────────────────────────────────────────────────────┐
│                     EggBWPanel                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Seed入力モード切替                                   │    │
│  │  ◉ LCG Seed 直接入力                                 │    │
│  │  ○ 起動時間から導出                                  │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼ (LCG Mode)                        ▼ (Boot Timing Mode)
┌────────────────────┐            ┌──────────────────────────┐
│ Seed Hex 入力      │            │ Boot Timing パラメータ    │
│ ┌────────────────┐ │            │ - Timestamp              │
│ │ 1234567890ABCD │ │            │ - Timer0 範囲            │
│ └────────────────┘ │            │ - VCount 範囲            │
└─────────┬──────────┘            │ - MAC Address            │
          │                       │ - Key Input              │
          │                       └────────────┬─────────────┘
          │                                    │
          │       ┌────────────────────────────┤
          │       │ deriveBootTimingEggSeedJobs()
          │       │ - Seed候補リスト生成
          │       │ - DerivedEggSeedJob[] 作成
          │       ▼
          │  ┌─────────────────────────────────┐
          │  │ DerivedEggSeedRunState          │
          │  │ - jobs: DerivedEggSeedJob[]     │
          │  │ - cursor: number                │
          │  │ - aggregate: {...}              │
          │  └─────────────┬───────────────────┘
          │                │
          └────────┬───────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ EggWorkerManager    │
         │ - LCG: 1回実行      │
         │ - Boot: N回順次実行 │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   egg-worker.ts     │
         │ EggSeedEnumerator   │
         │ (同一Worker利用)    │
         └─────────────────────┘
```

### 11.2 Boot Timing 結果表示の拡張

```
┌────────────────────────────────────────────────────────────────────────┐
│ 結果テーブル (Boot Timing Mode)                                         │
├────────┬───────┬────────────────┬────────┬──────────────────────────────┤
│ Advance│  IV   │ Nature/Gender  │ Shiny  │ Seed候補情報                 │
├────────┼───────┼────────────────┼────────┼───────┬────────┬─────────────┤
│   0    │ 31/.. │ いじっぱり/♂  │  通常  │ T0=C79│ VC=2D │ 0x1234...   │
│   1    │ 31/.. │ ようき/♀     │  ★    │ T0=C79│ VC=2D │ 0x1234...   │
│   0    │ 30/.. │ ひかえめ/♂   │  通常  │ T0=C7A│ VC=2D │ 0x5678...   │
│   ...  │  ...  │     ...       │  ...   │  ...  │  ...  │    ...      │
└────────┴───────┴────────────────┴────────┴───────┴────────┴─────────────┘
```

### 11.3 起動時間検索モード（Boot Timing Search）- SearchPanel類似機能

```
┌─────────────────────────────────────────────────────────────┐
│                EggSearchPanel (将来実装)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 検索条件入力                                           │  │
│  │ - 目標個体条件 (IV, 性格, 性別, 特性, 色違い等)        │  │
│  │ - 日時範囲 (開始日時 ～ 終了日時)                      │  │
│  │ - 消費範囲 (最小消費 ～ 最大消費)                      │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────┐
        │         EggSearchWorkerManager                  │
        │  - 日時範囲を分割して並列検索                   │
        │  - 条件を満たすSeedを収集                       │
        └────────────────────────┬───────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │          egg-search-worker.ts                   │
        │  - 各日時候補に対してSeedを計算                 │
        │  - 指定消費範囲で条件マッチング                 │
        └────────────────────────────────────────────────┘
```

## 12. BW2版 EggPanel 拡張経路

### 12.1 BW と BW2 の根本的差異

⚠️ **重要**: BW2 のタマゴ生成ロジックは BW とは**根本的に異なる**ため、
WASM レイヤーから完全に独立した実装が必要となる。

| 項目 | BW | BW2 |
|------|-----|------|
| **LCG Seed 決定** | 既存ロジック | **完全に異なる** (未実装) |
| **個体値決定** | `EggSeedEnumerator` 内で一体的に処理 | **独立したインタフェース** (未実装) |
| **PID 決定** | `EggSeedEnumerator` 内で一体的に処理 | **独立したインタフェース** (未実装) |

### 12.2 BW2 アーキテクチャ（将来構想）

```
┌─────────────────────────────────────────────────────────────┐
│                      App.tsx                                 │
│  ┌────────────────┐        ┌────────────────┐               │
│  │   EggBWPanel   │        │  EggBW2Panel   │               │
│  │  (BW専用UI)    │        │  (BW2専用UI)   │               │
│  └───────┬────────┘        └───────┬────────┘               │
│          │                         │                         │
│          ▼                         ▼                         │
│  ┌───────────────┐         ┌───────────────┐                │
│  │ EggBWStore    │         │ EggBW2Store   │                │
│  │ (BW専用状態)  │         │ (BW2専用状態) │                │
│  └───────┬───────┘         └───────┬───────┘                │
│          │                         │                         │
│          ▼                         ▼                         │
│  ┌────────────────┐        ┌─────────────────┐              │
│  │ egg-bw-worker  │        │ egg-bw2-worker  │              │
│  │ (BW専用)       │        │ (BW2専用)       │              │
│  └───────┬────────┘        └───────┬─────────┘              │
└──────────┼─────────────────────────┼────────────────────────┘
           │                         │
           ▼                         ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│ EggSeedEnumerator   │    │ EggBW2IVGenerator +             │
│ (BW用、既存)        │    │ EggBW2PIDGenerator              │
│                     │    │ (BW2用、未実装)                  │
│ - 一体的な個体生成  │    │ - 個体値/PID生成が独立          │
└─────────────────────┘    └─────────────────────────────────┘
```

### 12.3 共通化可能な部分（UIスタイルのみ）

```
src/components/egg/
├── common/                      ← 一部スタイル共通化可能
│   ├── EggResultsCard.tsx       (結果表示形式が同じ場合)
│   └── EggRunCard.tsx           (開始/停止UIは共通)
├── bw/                          ← BW専用（既存設計を維持）
│   ├── EggBWPanel.tsx
│   ├── EggBWParamsCard.tsx
│   └── EggBWFilterCard.tsx
└── bw2/                         ← BW2専用（完全に独立）
    ├── EggBW2Panel.tsx
    ├── EggBW2ParamsCard.tsx
    └── EggBW2FilterCard.tsx     (フィルター条件も異なる可能性)
```

### 12.4 BW2 WASM インタフェース（将来構想）

```
BW2 では個体値生成と性格値生成が独立したインタフェースを持つ:

┌─────────────────────────────────────────────────────────────┐
│                    egg-bw2-worker.ts                         │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │  EggBW2IVGenerator      │  │  EggBW2PIDGenerator     │  │
│  │  - 独立した個体値生成   │  │  - 独立したPID生成      │  │
│  │  - BW2固有のロジック    │  │  - BW2固有のロジック    │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                    ↑                    ↑                   │
│                    └──────────┬─────────┘                   │
│                               │                             │
│                    個体値とPIDを組み合わせて                  │
│                    最終的な個体データを生成                   │
└─────────────────────────────────────────────────────────────┘
```

### 12.5 注意事項

- BW2 の WASM 実装は**未実装**であり、本仕様書は将来的なアーキテクチャ構想を示すもの
- BW と BW2 で `EggSeedEnumerator` を共有する設計は**採用しない**
- BW2 実装時には、WASM インタフェースの詳細仕様を別途策定する必要がある
