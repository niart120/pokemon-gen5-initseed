# Generation Feature Phase3-4 Plan (Draft)

## 1. Purpose
Add Generation UI (Phase3 subset) and Generation Worker + performance pipeline (Phase4) without disrupting existing Search feature.

## 2. Scope (This PR)
- Generation worker scaffolding (no full WASM loop yet)
- Manager + state wiring
- Generation tab + panel skeleton
- Planning types placeholder

## 3. Out of Scope (Future PRs)
- Full WASM PokemonGenerator batching
- Advanced filters, export formats
- Parallel generation / SIMD benchmarking

## 4. Architecture (High Level)
UI -> GenerationManager -> generation-worker -> WASM(PokemonGenerator) -> raw-parser -> resolver -> UI table

## 5. Initial Tasks (Subset Executed in This Branch Start)
1. Branch create ✅
2. WASM API survey (PokemonGenerator exports) ⏳
3. Worker protocol design (START/PAUSE/RESUME/STOP/PROGRESS/RESULT_BATCH/COMPLETE/ERROR)
4. Type definitions file `src/types/generation.ts`
5. Worker & Manager skeletons
6. Store extension + Generation tab skeleton

## 6. Risks / Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| WASM API shape differs from assumption | Rework | Survey early (Task2) |
| Large batch posting blocks main thread | UI jank | Use modest batch size + postMessage transfer of plain arrays |
| Memory growth in results | Crash | Configurable max results + early stop |
| Shiny stop logic complexity | Delay | Implement after basic loop validated |

## 7. Metrics (Targets)
- Throughput (goal later): >=10k results/sec (wild baseline)
- Progress latency: < 500ms

## 8. Next Immediate Action
Design worker protocol & TypeScript types based on confirmed WASM API.

## 9. WASM Generation API (Survey Result)

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

### 9.6 Preliminary Worker Parameters (draft)
| Param | Type | Description |
|-------|------|-------------|
| baseSeed | bigint | initial seed entered by user |
| offset | bigint | start advances (initial 0 for MVP) |
| maxAdvances | number | enumeration upper bound (cap to 1,000,000) |
| maxResults | number | UI results cap; worker stops collecting beyond (but continues counting for progress unless stopOnCap flag) |
| version | GameVersion | B / W / B2 / W2 (map to WASM enum) |
| encounterType | EncounterType | user-selected |
| tid | number | 0-65535 |
| sid | number | 0-65535 |
| syncEnabled | boolean | sync toggle |
| syncNatureId | number | 0-24 |
| stopAtFirstShiny | boolean | early termination condition |

### 9.7 Progress Reporting Formula (draft)
Initial (draft) formula: `processedAdvances / maxAdvances` with elapsed ms and instantaneous throughput (advances/sec). Estimated remaining = (elapsed / processed) * (remaining).

Updated (A3 Implementation):
- `throughputRaw` = `processedAdvances / (elapsedMs/1000)` (生スループット)
- `throughputEma` = Exponential Moving Average of `throughputRaw` with α=0.2 (初回は raw を初期値)
- `throughput` (DEPRECATED) = 後方互換目的で `throughputRaw` の複製値
- `etaMs` 計算基礎 = `throughputEma` が正値ならそれを使用。0 または未定義の場合は `throughputRaw` をフォールバック。
	- `remaining = totalAdvances - processedAdvances`
	- `etaMs = remaining / basis * 1000`
		- where `basis = throughputEma > 0 ? throughputEma : throughputRaw`
	- 進捗 0 (elapsedMs=0) の間は 0 を維持

Rationale:
- 短い tick 間隔 (250ms 固定) における瞬間値ノイズを平滑化し ETA の過度な揺れを抑制
- α=0.2 は 1/α=5 tick (≈1.25s) 程度で 63% 収束するバランス値
- 後方互換フィールド `throughput` を保持し UI 減衰移行コストを最小化

### 9.8 Serialization Strategy
- Use `SeedEnumerator` for incremental streaming (avoids large Vec overhead crossing boundary repeatedly).
- Convert each `RawPokemonData` via existing `parseFromWasmRaw` logic inside worker OR send raw fields manually (faster). Chosen: manual extraction to plain object to avoid reflection overhead.

### 9.9 Open Points
- Hidden ability slot (2) currently not produced; future extension may require species ability table update.
- Parallel generation (multi-worker) not required in this phase; revisit after baseline metrics.

## 10. Planned Next PR Section (will evolve)
To be filled after protocol & types finalized.

## 11. Generation Worker Message Protocol (Draft)

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

### 11.3 Completion Reasons (reason field)
- "max-advances" : processedAdvances reached maxAdvances
- "max-results" : resultsCount reached maxResults and stopOnCap
- "first-shiny" : shiny encountered and stopAtFirstShiny=true
- "stopped" : STOP command
- "error" : internal error (also ERROR message sent earlier)

### 11.4 State Transitions
IDLE -> RUNNING (START) -> (PAUSED <-> RUNNING)* -> (COMPLETE|STOPPED|ERROR) -> IDLE

### 11.5 Progress Emission Policy
- Emit on: (a) every progressIntervalMs elapsed, (b) after each RESULT_BATCH, (c) at completion.

### 11.6 Validation Rules (subset)
| Field | Rule | Error Category |
|-------|------|----------------|
| baseSeed | 0 <= seed < 2^64 | VALIDATION |
| maxAdvances | 1..1_000_000 (hard cap) | VALIDATION |
| maxResults | 1..100_000 | VALIDATION |
| batchSize | 1..10_000 and ≤ maxAdvances | VALIDATION |
| syncNatureId | 0..24 | VALIDATION |

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
- Whether to include per-batch max PID / stats (defer)
- Cancellation token vs STOP message (current: STOP only)


---
Draft generated on initial scaffold.
