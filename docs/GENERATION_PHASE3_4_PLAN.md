# Generation Feature Phase3-4 Plan (Implemented Summary)

## 1. Purpose
Generation 機能 (Phase3 UI + Phase4 Worker/性能パイプライン) の最終実装要約。Search 機能を破壊せず併存し、初期 Seed からの連続乱数列に基づくポケモン生成ストリーミング・進捗監視・早期終了制御を提供。

## 2. Scope (実装範囲)
- Generation WebWorker + Manager
- Store 拡張 (進捗/結果/統計)
- Generation タブ UI (基本パラメータ入力 + リアルタイム指標表示 + 結果テーブル)
- 早期終了条件 (max-advances / max-results / first-shiny / manual stop)
- 出力エクスポート (CSV / JSON / TXT) with BigInt 十六進 + 10 進併記
- Throughput EMA (α=0.2) と ETA 推定

除外 (後続検討): マルチワーカー並列, SIMD 専用最適化比較, 高度フィルタ, 隠れ特性 / 個体値完全生成, 遭遇テーブル最終解決表示。

## 3. Deferred / Future Work
| 項目 | 備考 |
|------|------|
| Parallel generation (multi-worker) | E フェーズ候補 |
| SIMD ベンチ強化ページ | test-simd.html 既存比較の深化 |
| 高度フィルタ (性格/特性/色違い同時条件) | UI 拡張 |
| Hidden ability / 個体値算出 | Rust 側 Raw 拡張後 |
| Encounter 結果の種族解決 | species / tables 安定化後 |

## 4. High Level Architecture
UI → GenerationWorkerManager → generation-worker (TS) → WASM (PokemonGenerator / SeedEnumerator) → Plain Raw Objects → Store → UI Table / Exporter

## 5. Task Trace (Completed)
| # | Task | Status |
|---|------|--------|
| 1 | Branch create | ✅ |
| 2 | WASM API survey | ✅ |
| 3 | Worker protocol 設計 | ✅ |
| 4 | 型定義 `src/types/generation.ts` | ✅ |
| 5 | Worker & Manager 実装 | ✅ |
| 6 | Store + UI skeleton | ✅ |
| 7 | Throughput/EMA + ETA | ✅ |
| 8 | Early termination logic | ✅ |
| 9 | Exporter (CSV/JSON/TXT) | ✅ |
| 10 | Integration tests (Node guard) | ✅ |

## 6. Risks / Mitigation (現状)
| Risk | Impact | Mitigation |
|------|--------|------------|
| WASM API shape differs from assumption | Resolved | Survey 完了 |
| Large batch posting blocks main thread | UI jank | Use modest batch size + postMessage transfer of plain arrays |
| Memory growth in results | Mitigated | maxResults + stopOnCap |
| Shiny stop logic complexity | Mitigated | first-shiny 完了 |

## 7. Metrics
Measured (local baseline, non-SIMD search context):
- Progress tick interval: 250ms (≤ 500ms target)
- EMA convergence window ≈ 1.25s (α=0.2)
- Export: O(n) streaming serialization (tested up to 50k rows)

User-facing metrics exposed:
| Field | 説明 |
|-------|------|
| processedAdvances | 消費済み乱数数 |
| resultsCount | 収集済み結果数 |
| throughputRaw | 現在区間生スループット (adv/sec) |
| throughputEma | 平滑化スループット |
| etaMs | 推定残り時間 (ms) |
| shinyCount | 発見済み色違い数 |

## 8. Current Focus
Documentation (D2) 整備。次段階で parallel / advanced filter 評価。

## 9. WASM Generation API (実装参照)

### 9.1 Exported Types / Constructors
- `BWGenerationConfig::new(version: GameVersion, encounter_type: EncounterType, tid: u16, sid: u16, sync_enabled: bool, sync_nature_id: u8)`
- `PokemonGenerator::generate_pokemon_batch_bw(base_seed: u64, offset: u64, count: u32, &config) -> Vec<RawPokemonData>`
- `SeedEnumerator::new(base_seed: u64, offset: u64, count: u32, &config)` + `next_pokemon()` (incremental alternative)

### 9.2 RawPokemonData Fields (getter names)
| Field | Getter | TS target type | Notes |
|-------|--------|----------------|-------|
| seed | get_seed | bigint | 64-bit initial seed used for that Pokemon |
| pid | get_pid | number | 32-bit PID |
| nature | get_nature | number (0-24) | Nature index |
| ability_slot | get_ability_slot | number (0/1/2) | Hidden ability not yet produced (0/1 currently) |
| gender_value | get_gender_value | number (0-255) | Compared with species ratio threshold |
| encounter_slot_value | get_encounter_slot_value | number | Index into encounter table slots |
| encounter_type | get_encounter_type | number | Mapping documented in rust (0,1,2,3,4,5,6,7,10,11,12,13,20) |
| level_rand_value | get_level_rand_value | bigint | For level computation (surf/fish etc.) |
| shiny_type | get_shiny_type | number (0/1/2) | 0:Not,1:Square,2:Star |
| sync_applied | get_sync_applied | boolean | Whether sync nature actually applied |

### 9.3 Batch Generation Constraints
- Hard cap: `MAX_BATCH_COUNT = 1_000_000` inside Rust (requests above are truncated)
- Performance path: Use `generate_pokemon_batch_bw` for contiguous enumeration (one allocation) vs `SeedEnumerator` for streaming smaller memory footprint.
- Offset semantics: Start seed = base_seed advanced by `offset` steps (affine jump). Each subsequent result uses one RNG advance (`PersonalityRNG::next_seed`).

### 9.4 Encounter Sync Applicability
Supports Sync: Normal, Surfing, Fishing, ShakingGrass, DustCloud, PokemonShadow, SurfingBubble, FishingBubble, StaticSymbol. (Others ignore but still may consume RNG for wild categories as per code.)

### 9.5 Proposed Worker Usage Pattern
1. Build `BWGenerationConfig` in worker from params
2. For large count (≥ 50k) prefer chunked loop using SeedEnumerator to reduce peak memory (e.g. chunk size 10k)
3. Post `RESULT_BATCH` after each chunk (array of plain serializable objects) + `PROGRESS`
4. If `stopAtFirstShiny` flag: short-circuit enumeration after detecting first shiny (shiny_type != 0)

### 9.6 Worker Parameters (final MVP)
| Param | Type | Description | Constraints |
|-------|------|-------------|-------------|
| baseSeed | bigint | 初期Seed | 0 ≤ seed < 2^64 |
| offset | bigint | 開始オフセット | 0..maxAdvances |
| maxAdvances | number | 乱数消費上限 | 1..1,000,000 |
| maxResults | number | 収集上限 | 1..100,000 且つ ≤ maxAdvances |
| version | GameVersion | ゲームバージョン | enum |
| encounterType | EncounterType | 遭遇種別 | enum |
| tid | number | 表ID | 0..65535 |
| sid | number | 裏ID | 0..65535 |
| syncEnabled | boolean | シンクロ有効 | - |
| syncNatureId | number | シンクロ性格ID | 0..24 |
| stopAtFirstShiny | boolean | 最初の色違いで停止 | - |
| stopOnCap | boolean | maxResults 到達で停止 | default true |
| progressIntervalMs | number | 進捗送信間隔 | default 250 (≤500) |
| batchSize | number | 生成バッチ数 | 1..10,000 且つ ≤ maxAdvances |

### 9.7 Progress / ETA Calculation
Initial (draft) formula: `processedAdvances / maxAdvances` with elapsed ms and instantaneous throughput (advances/sec). Estimated remaining = (elapsed / processed) * (remaining).

Implementation:
- `throughputRaw` = `processedAdvances / (elapsedMs/1000)` (生スループット)
- `throughputEma` = Exponential Moving Average of `throughputRaw` with α=0.2 (初回は raw を初期値)
- `throughput` (DEPRECATED) = 後方互換目的で `throughputRaw` の複製値
- `etaMs` 計算基礎 = `throughputEma` が正値ならそれを使用。0 または未定義の場合は `throughputRaw` をフォールバック。
	- `remaining = totalAdvances - processedAdvances`
	- `etaMs = remaining / basis * 1000`
		- where `basis = throughputEma > 0 ? throughputEma : throughputRaw`
	- 進捗 0 (elapsedMs=0) の間は 0 を維持

Rationale (unchanged):
- 250ms tick のノイズ平滑化
- α=0.2 → 約 1.25s で 63% 収束
- 互換性のため deprecated フィールド保持

### 9.8 Serialization Strategy
- Use `SeedEnumerator` for incremental streaming (avoids large Vec overhead crossing boundary repeatedly).
- Convert each `RawPokemonData` via existing `parseFromWasmRaw` logic inside worker OR send raw fields manually (faster). Chosen: manual extraction to plain object to avoid reflection overhead.

### 9.9 Notes
- Hidden ability slot (2) 未生成 (将来拡張)
- Parallel generation: 後続フェーズ
- Encounter 種族解決: UI 遅延読み込み最適化対象

## 10. Export Specification (MVP)
| Format | 特徴 | 備考 |
|--------|------|------|
| CSV | 1行/結果, bigint列は hex と decimal 重複列 | RFC4180 近似, 改行 \n |
| JSON | 配列シリアライズ | bigint → 文字列 ("0x...") |
| TXT | 可読整形 (列揃え) | 行数多でサイズ増 |

共通列 (暫定): advance, seed_hex, seed_dec, pid_hex, pid_dec, nature, ability_slot, encounter_type, encounter_slot_value, shiny_type, sync_applied.

## 11. Generation Worker Message Protocol (Final)

差分: Draft から PAUSE/RESUME は内部未使用 (UI 未提供) だが後方互換維持。

### 11.1 Request -> Worker
| Type | Payload | Notes |
|------|---------|-------|
| START_GENERATION | { params, requestId? } | params = validated GenerationParams |
| PAUSE | { requestId? } | Idempotent |
| RESUME | { requestId? } | No-op if not paused |
| STOP | { requestId?, reason?: string } | Triggers graceful completion (STOPPED) |

`GenerationParams` (summary): baseSeed(bigint), offset(bigint), maxAdvances(number), maxResults(number), version(GameVersion), encounterType(EncounterType), tid(number), sid(number), syncEnabled(boolean), syncNatureId(number), stopAtFirstShiny(boolean), stopOnCap(boolean=true), progressIntervalMs(number=500), batchSize(number=1000 default, soft ≤ 10000).

### 11.2 Worker -> Main Responses
| Type | Payload Fields | Description |
|------|----------------|-------------|
| READY | { version:"1" } | Worker initialized |
| PROGRESS | { processedAdvances, totalAdvances, resultsCount, elapsedMs, throughput (deprecated), throughputRaw?, throughputEma?, etaMs, status } | A3: throughputRaw 生値, throughputEma=EMA(α=0.2), deprecated throughput=throughputRaw |
| RESULT_BATCH | { batchIndex, results:[RawLike], batchSize, cumulativeResults } | RawLike: minimal raw fields (see 9.2) |
| PAUSED | { message? } | Acknowledge pause |
| RESUMED | { } | Acknowledge resume |
| STOPPED | { reason, processedAdvances, resultsCount, elapsedMs } | User-issued stop |
| COMPLETE | { reason, processedAdvances, resultsCount, elapsedMs, shinyFound:boolean } | Normal/early completion |
| ERROR | { message, category, fatal:boolean } | category: VALIDATION | WASM_INIT | RUNTIME | ABORTED |

### 11.3 Completion Reasons (reason)
| reason | 条件 |
|--------|------|
| max-advances | processedAdvances == maxAdvances |
| max-results | resultsCount == maxResults AND stopOnCap |
| first-shiny | 色違い検出 AND stopAtFirstShiny |
| stopped | STOP 指示 |
| error | 内部エラー (ERROR メッセージ送信済) |

### 11.4 State Transitions
IDLE -> RUNNING (START) -> (PAUSED <-> RUNNING)* -> (COMPLETE|STOPPED|ERROR) -> IDLE

### 11.5 Progress Emission Policy
- Emit on: (a) every progressIntervalMs elapsed, (b) after each RESULT_BATCH, (c) at completion.

### 11.6 Validation Rules (final subset)
| Field | Rule | Error Category |
|-------|------|----------------|
| baseSeed | 0 ≤ seed < 2^64 | VALIDATION |
| maxAdvances | 1..1_000_000 | VALIDATION |
| maxResults | 1..100_000 AND ≤ maxAdvances | VALIDATION |
| batchSize | 1..10_000 AND ≤ maxAdvances | VALIDATION |
| syncNatureId | 0..24 | VALIDATION |
| offset | 0..maxAdvances | VALIDATION |

### 11.7 Error Handling Strategy
- Validation failure before start -> send ERROR(fatal=true) keep state IDLE.
- Runtime error during loop -> send ERROR(fatal=true) then STOPPED/COMPLETE skipped.
- STOP during PAUSED -> immediate STOPPED (no RESUMED).

### 11.8 Serialization Shape (RawLike)
```ts
type RawLike = {
	seed: bigint;
	pid: number;
	nature: number;
	ability_slot: number;
	gender_value: number;
	encounter_slot_value: number;
	encounter_type: number;
	level_rand_value: bigint;
	shiny_type: number;
	sync_applied: boolean;
	advance: number; // computed index = offset + localIndex
};
```

### 11.9 Open Items
- per-batch 集計列 (max PID 等) 追加検討
- Cancellation token vs STOP (現状 STOP のみ)
- ProgressInterval 動的調整


---
Updated: 2025-08-12
