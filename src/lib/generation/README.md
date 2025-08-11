# Resolver Generation - Phase 2-5 (assembler 廃止)

## Overview

Resolver は WASM の Raw 出力（snake_case）を、遭遇テーブル/種族データを用いて解決するドメイン層です。旧 pokemon-assembler は廃止され、`pokemon-resolver.ts` と `build-resolution-context.ts` を利用します。

## Key Features

### 1. Data Generation/Resolution
- Raw WASM (`types/pokemon-raw.ts`) → `resolvePokemon()` で speciesId/level/gender 等を解決
- 遭遇スロット/レベルは `data/encounter-tables` と `buildResolutionContext()` のテーブルに基づく
- UI 名称は `toUiReadyPokemon()` で最低限を付与（i18nは上位層）

### 2. Special Encounter Handling
- 旧 assembler の砂煙ロジックは削除。必要なら resolver/コンテキスト層に仕様化して追加する。

### 3. Sync Rule Enforcement
- シンクロ適用可能性は DomainEncounterType と仕様に基づいてテストで検証。

## Usage

```ts
import { buildResolutionContext } from '@/lib/initialization/build-resolution-context';
import { resolvePokemon, toUiReadyPokemon } from '@/lib/generation/pokemon-resolver';
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

- See `src/types/pokemon-raw.ts`
- Generated/Resolved outputs remain UI-agnostic

## Testing

統合テストは `src/test/phase2-integration.test.ts` を参照
