# Resolver Integration - Phase 2-5 (assembler 廃止)

## Overview

Resolver は WASM の Raw 出力（snake_case）を、遭遇テーブル/種族データを用いて解決するドメイン層です。旧 pokemon-assembler は廃止され、`pokemon-resolver.ts` と `build-resolution-context.ts` を利用します。

## Key Features

### 1. Data Integration
- Raw WASM (`types/pokemon-raw.ts`) → `resolvePokemon()` で speciesId/level/gender 等を解決
- 遭遇スロット/レベルは `data/encounter-tables` と `buildResolutionContext()` のテーブルに基づく
- UI 名称は `toUiReadyPokemon()` で最低限を付与（i18nは上位層）

### 2. Special Encounter Handling
- 旧 assembler の砂煙ロジックは削除。必要なら resolver/コンテキスト層に仕様化して追加する。

### 3. Sync Rule Enforcement
- シンクロ適用可能性は DomainEncounterType と仕様に基づいてテストで検証。

## Usage

```typescript
import { buildResolutionContext } from '@/lib/initialization/build-resolution-context';
import { resolvePokemon, toUiReadyPokemon } from '@/lib/integration/pokemon-resolver';
import type { UnresolvedPokemonData } from '@/types/pokemon-raw';

const ctx = buildResolutionContext({ version: 'B', location: 'Route1', encounterType: 0 });
const raw: UnresolvedPokemonData = {
  seed: 0x12345678n,
  pid: 0x87654321,
  nature: 12,
  sync_applied: true,
  ability_slot: 1,
  gender_value: 100,
  encounter_slot_value: 0,
  encounter_type: 0,
  level_rand_value: 2,
  shiny_type: 0,
};
const resolved = resolvePokemon(raw, ctx);
const ui = toUiReadyPokemon(resolved);
```

## Type Definitions

### `UnresolvedPokemonData` (snake_case)
WASM 計算結果（snake_case）:
- `seed`: bigint 初期seed
- `pid`: number
- `nature`: 0-24
- `sync_applied`: boolean
- `ability_slot`: 0-2
- `gender_value`: 0-255
- `encounter_slot_value`: number
- `encounter_type`: DomainEncounterType number
- `level_rand_value`: number
- `shiny_type`: 0(normal)/1(square)/2(star)

### `ResolvedPokemonData` / `UiReadyPokemonData`
`resolvePokemon()` の出力（ID中心）と、`toUiReadyPokemon()` の最小UI付与:
- `speciesId?`, `level?`, `gender?` などの解決済みフィールド
- `natureName`, `shinyStatus` は UI 便宜のための付加

## Encounter Types

### Wild Encounters (Sync Eligible)
- `Normal` (0): Grass, cave, dungeon encounters
- `Surfing` (1): Water surface encounters
- `Fishing` (2): Fishing rod encounters
- `ShakingGrass` (3): Special grass encounters
- `DustCloud` (4): Dust cloud encounters (with item logic)
- `PokemonShadow` (5): Shadow encounters
- `SurfingBubble` (6): Special water encounters
- `FishingBubble` (7): Special fishing encounters

### Static Encounters
- `StaticSymbol` (10): Legendary Pokemon (sync eligible)
- `StaticStarter` (11): Starter Pokemon (sync NOT eligible)
- `StaticFossil` (12): Fossil Pokemon (sync NOT eligible)
- `StaticEvent` (13): Event Pokemon (sync NOT eligible)

### Roaming Encounters (Sync NOT Eligible)
- `Roaming` (20): Roaming legendary Pokemon (**critical**: sync never applies)

## Special Features

### Dust Cloud Logic
旧ロジックは削除。必要であれば domain 仕様策定後に追加。

### Sync Rule Validation
The `validateSyncRules()` method ensures:
1. Roaming encounters never have sync applied
2. Static starters/fossils/events don't have sync applied
3. Only eligible encounter types can have sync

## Testing

テストは resolver 統合に更新されました。`src/test/phase2-integration.test.ts` を参照してください。

## Implementation Notes

- **Source of Truth**: WASM implementation is authoritative
- **Integration Strategy**: Uses IntegratedSeedSearcher approach (no direct WASM calls)
- **Minimal Changes**: Focus on TypeScript integration layer only
- **Type Safety**: Full TypeScript typing with proper interfaces
- **Validation**: Built-in validation for sync rule compliance

## Constraints Satisfied

✅ **Raw parsed values + encounter table + resolution logic integration**  
✅ ~~Special encounter (dust cloud) item appearance determination~~（一時撤去）  
✅ **Strict sync application scope (wild only, roaming excluded)**  
✅ **Roaming sync non-application testing**  
✅ **Representative encounter type validation**  
✅ **Source of Truth: wasm-pkg (Rust) implementation**  
✅ **IntegratedSeedSearcher approach (no individual WASM calls)**