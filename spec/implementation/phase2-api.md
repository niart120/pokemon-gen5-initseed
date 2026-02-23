# Phase 2 — TypeScript 統合 API ドキュメント（実装準拠・日本語版）

本書は、WASM で生成した生ポケモンデータをエンカウントテーブルと種族データで拡張し、表示・検証に適した形へ統合する Phase 2 の TypeScript API を実装準拠で説明します。

## 全体像（正しい入口の明確化）

```
WASM PokemonGenerator → RawPokemonData → Integration Service → EnhancedPokemonData
                               │                   │
                               └─ parseRawPokemonData          └─ エンカウントテーブル + 生成種族データ
```

- 生成エントリは PokemonGenerator（WASM）です。IntegratedSeedSearcher は「初期Seed探索」用であり、生成パイプラインの入口ではありません。
- 型・パーサは `src/types/pokemon-raw.ts`（WASM境界の snake_case Raw）と Resolver 一連（`src/lib/generation/raw-parser.ts`, `src/lib/generation/pokemon-resolver.ts`）を基準とします。
- エンカウントテーブルは JSON データのみに依存し、統合サービスではフォールバックを行いません（テスト用途のサンプルは別途有り）。
- 種族データは生成済み JSON アダプタ（`src/data/species/generated`）を使用します。

## 1) RawPokemonData とパーサ（`src/types/pokemon-raw.ts` / `src/lib/generation/raw-parser.ts`）

役割
- WASM から返る RawPokemonData を TypeScript へ安全に変換
- 名称変換（性格/色違い種別）などのユーティリティ提供

主要型（抜粋）
- RawPokemonData: seed/pid/nature/syncApplied/abilitySlot/genderValue/encounterSlotValue/encounterType/levelRandValue/shinyType
- EnhancedPokemonData: RawPokemonData を拡張し species/ability/gender/level/encounter/natureName/shinyStatus を付与

主要関数
- parseRawPokemonData(wasmData): WASM の getter/properties 双方に対応して厳密変換
- getNatureName(natureId): 0–24 を検証して名称を返却
- getShinyStatusName(shinyType): 列挙に基づく名称化

補足
- enum EncounterType/ShinyType を公開

## 2) WASM 生成サービス（廃止済み）

役割
- 以前は PokemonGenerator の高水準ラッパ（入力検証/例外一元化）を提供していましたが、現在は Worker/Resolver 境界に集約しました。

公開型
- 専用の公開型は提供しません。必要に応じて呼び出し側で最小の型を定義してください。

公開クラス
- なし（削除済み）

ユーティリティ
- なし（削除済み）

エラー
- 呼び出し側で try/catch によりハンドリングしてください。

備考
- 生成は PokemonGenerator を直接呼び出してください。

## 3) 統合サービス（Resolverベース）

役割
- RawPokemonData + エンカウントテーブル + 生成種族データ を結合し EnhancedPokemonData を作成

公開型
- IntegrationConfig: version/defaultLocation/applySynchronize/synchronizeNature
- IntegrationResult: pokemon（Enhanced）+ metadata（found フラグ・warnings）

公開クラス/関数
- シングルトン/サービス層は提供しません。`pokemon-resolver.ts` の関数群を直接利用します。

エラー/方針
- IntegrationError（code: MISSING_ENCOUNTER_TABLE など）
- 統合サービスは JSON データのみを参照し「フォールバック無し」。テーブル未発見は例外。
- 同期（Synchronize）は野生系エンカウント（EncounterType 0..7）のみ対象。静的/イベント/ローミングは対象外。

EnhancedPokemonData へのマッピング注意
- 現状の生成種族データには型や性別比の完全情報が未整備のため、以下は暫定値：
  - species.types: ['Normal'] のプレースホルダ
  - species.genderRatio: genderless のとき -1、それ以外は 0
 追って生成データ拡充時に正値へ置換予定。

## 4) エンカウントテーブル（`src/data/encounter-tables.ts`）

役割
- `getEncounterTable(version, location, method)` / `getEncounterSlot(table, slotValue)` を提供（レベル計算は Resolver 側に集約）

方針
- 統合サービスは JSON テーブル必須（フォールバック無し）。
- テスト/デモ向けのサンプルはアセンブラ側（後述）で提供。

備考
- レベル計算は `pokemon-resolver.ts` にて Raw の `level_rand_value` とスロットの `levelRange` を用いて実装（一元化）。

出典（実装に合わせる）
- エンカウントテーブル: pokebook.jp（BW/BW2 の各ページ）

## 5) 生成種族データ（`src/data/species/generated`）

役割
- 生成済み JSON から種族・特性を引くアダプタ

公開関数（代表）
- getGeneratedSpeciesById(id)
- selectAbilityBySlot(slot, abilities)

注意
- abilities は `{ ability1/ability2/hidden }` それぞれに `names.{ en, ja }` を持つ構造です。

## 6) アセンブラ（関数型 API、テスト/デモ用）

場所
- `src/lib/integration/pokemon-assembler.ts`

役割
- サンプルエンカウントテーブル（`createSampleEncounterTables()`）や同期ルール検証（`validateSyncRules()`）を含むデモ/テスト支援 API。
- 本番統合は `PokemonIntegrationService` を使用し、フォールバックは行いません。

主な関数
- createAssemblerContext(version, region, tables?)
- assembleData(ctx, raw) / assembleBatch(ctx, raws)
- setEncounterTable(ctx, type, table) / getEncounterTables(ctx)
- validateSyncRules(enhanced[]): 同期適用の適合性検証

## 使用例

WASM 生成 → 統合までの最小例：

```ts
// 生成 → 統合（擬似コード）
import { parseFromWasmRaw } from '@/lib/generation/raw-parser';
import { resolvePokemon, toUiReadyPokemon } from '@/lib/generation/pokemon-resolver';

async function generateEnhancedPokemon(seed: bigint) {
  const wasmRaw = /* PokemonGenerator.generate_*_bw(...) */ null as any;
  const raw = parseFromWasmRaw(wasmRaw);
  const ctx = /* buildResolutionContext(...) */ null as any;
  const enhanced = resolvePokemon(ctx, raw);
  return toUiReadyPokemon(enhanced);
}
```

## エラー処理

> 例外は呼び出し側で適切にハンドリングしてください。

## テスト

- 統合テスト: `src/test/phase2-integration.test.ts`
- アセンブラ検証: `src/test/integration/pokemon-assembler.test.ts`
- 生成サービス: （廃止）

## データ出典（現行実装に準拠）

- 技術資料: rusted-coil、xxsakixx（BWエンカウント・乱数）
- エンカウントテーブル: pokebook.jp（BW/BW2 各ページ）
- 補助資料: 必要に応じて Bulbapedia など

## 付記（今後の拡張）

- 生成種族データの型/性別比/タイプ情報の拡充に合わせ、EnhancedPokemonData へのプレースホルダを正値化します。
- 隠れ特性はフラグ導入後に isHidden の正確化を予定。

