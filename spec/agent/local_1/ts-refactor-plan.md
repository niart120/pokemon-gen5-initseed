# TypeScript側リファクタリングプラン

## 概要
Rust WASM側で実装した `search_common` モジュールの新しいパラメータ型を、TypeScript側で使用するように改修する。

## 対象ファイル

### 改修対象
- `src/workers/egg-boot-timing-worker.ts` - EggBootTimingSearchIterator の呼び出し改修

### 新規作成候補
- `src/workers/iv-boot-timing-worker.ts` - IVBootTimingSearchIterator 用Worker（未実装）

## Rust側の新API

```rust
// EggBootTimingSearchIterator::new
pub fn new(
    ds_config: DSConfigJs,           // MAC, Nazo, Hardware
    segment_params: SegmentParamsJs, // Timer0, VCount, KeyCode
    time_range: TimeRangeParamsJs,   // hour/minute/second range
    search_range: SearchRangeParamsJs, // year/month/day, range_seconds
    conditions: GenerationConditionsJs,
    parents: ParentsIVsJs,
    filter: Option<IndividualFilterJs>,
    consider_npc_consumption: bool,
    game_mode: u8,
    user_offset: i64,
    advance_count: u32,
)
```

## 改修内容

### 1. ヘルパー関数追加
```typescript
function buildDSConfig(wasmAny, params, nazo): DSConfigJs
function buildSegmentParams(wasmAny, timer0, vcount, keyCode): SegmentParamsJs
function buildTimeRangeParams(wasmAny, timeRange): TimeRangeParamsJs
function buildSearchRangeParams(wasmAny, startDate, rangeSeconds): SearchRangeParamsJs
```

### 2. コンストラクタ呼び出し変更

**Before** (27個以上の個別パラメータ):
```typescript
new wasmAny.EggBootTimingSearchIterator(
  new Uint8Array(params.macAddress),
  new Uint32Array(nazo),
  params.hardware,
  timer0, vcount, keyCode,
  params.timeRange.hour.start, params.timeRange.hour.end,
  params.timeRange.minute.start, params.timeRange.minute.end,
  params.timeRange.second.start, params.timeRange.second.end,
  year, month, day, rangeSeconds,
  conditions, parentsIVs, filter,
  considerNpcConsumption, gameMode, userOffset, advanceCount
)
```

**After** (構造化パラメータ):
```typescript
const dsConfig = buildDSConfig(wasmAny, params, nazo);
const segmentParams = buildSegmentParams(wasmAny, timer0, vcount, keyCode);
const timeRange = buildTimeRangeParams(wasmAny, params.timeRange);
const searchRange = buildSearchRangeParams(wasmAny, searchStartDate, rangeSeconds);

new wasmAny.EggBootTimingSearchIterator(
  dsConfig, segmentParams, timeRange, searchRange,
  conditions, parentsIVs, filter,
  params.considerNpcConsumption,
  eggGameModeToWasm(params.gameMode),
  BigInt(params.userOffset),
  params.advanceCount
)
```

## 実装順序

1. 改修プラン文書作成
2. WASM型定義更新確認 (`npm run build:wasm`)
3. Egg Worker ヘルパー関数追加
4. Egg Worker コンストラクタ更新
5. テスト実行・動作確認

## 検証方法

- `npm run test` - 既存テスト通過確認
- `test-integration.html` - 統合テスト実行
- 実際のEggブートタイミング検索動作確認
