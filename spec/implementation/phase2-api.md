# Phase 2 — TypeScript 統合 API ドキュメント（実装準拠・日本語版）

本書は、WASM で生成した生ポケモンデータを遭遇テーブルと種族データで拡張し、表示・検証に適した形へ統合する Phase 2 の TypeScript API を実装準拠で説明します。

## 全体像（正しい入口の明確化）

```
WASM PokemonGenerator → RawPokemonData → Integration Service → EnhancedPokemonData
                               │                   │
                               └─ parseRawPokemonData          └─ 遭遇テーブル + 生成種族データ
```

- 生成エントリは PokemonGenerator（WASM）です。IntegratedSeedSearcher は「初期Seed探索」用であり、生成パイプラインの入口ではありません。
- 型・パーサは `src/types/pokemon-enhanced.ts`（UI向けヘルパ含む）と `src/types/pokemon-raw.ts`（WASM層）を基準とします。
- 遭遇テーブルは JSON データのみに依存し、統合サービスではフォールバックを行いません（テスト用途のサンプルは別途有り）。
- 種族データは生成済み JSON アダプタ（`src/data/species/generated`）を使用します。

## 1) RawPokemonData とパーサ（`src/types/pokemon-enhanced.ts` / `src/types/pokemon-raw.ts`）

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

## 2) WASM 生成サービス（`src/lib/services/wasm-pokemon-service.ts`）

役割
- PokemonGenerator の高水準ラッパ。入力検証と例外を一元化

公開型
- WasmGenerationConfig: version/region/hardware/tid/sid/syncEnabled/syncNatureId/macAddress/keyInput/frame
- PokemonGenerationRequest: seed/config/(count/offset)
- PokemonGenerationResult: pokemon[]/stats（時間・件数・初期seed）

公開クラス
- WasmPokemonService
  - initialize(): WASM 初期化
  - isReady(): 準備完了判定
  - generateSinglePokemon(req): RawPokemonData を 1 件生成
  - generatePokemonBatch(req): RawPokemonData を複数生成
  - static createDefaultConfig(): デモ用既定設定

ユーティリティ
- getWasmPokemonService(): シングルトン取得（内部で initialize）
- generatePokemon(seed, config?): 1件生成の簡易関数
- generatePokemonBatch(seed, count, config?): 複数生成の簡易関数

エラー
- WasmServiceError（code: WASM_INIT_FAILED/GENERATION_FAILED/BATCH_GENERATION_FAILED/NOT_INITIALIZED/…）

備考
- 実装は `BWGenerationConfig` を構築し `PokemonGenerator.generate_*_bw` を呼び出します。

## 3) 統合サービス（`src/lib/services/pokemon-integration-service.ts`）

役割
- RawPokemonData + 遭遇テーブル + 生成種族データ を結合し EnhancedPokemonData を作成

公開型
- IntegrationConfig: version/defaultLocation/applySynchronize/synchronizeNature
- IntegrationResult: pokemon（Enhanced）+ metadata（found フラグ・warnings）

公開クラス/関数
- PokemonIntegrationService
  - integratePokemon(raw, config): 1件統合
  - integratePokemonBatch(raws, config): 複数統合
  - validateIntegrationResult(result): レベル範囲/性格/特性スロット等の整合性検証
  - getIntegrationStats(results): 統計集計
- getIntegrationService(): シングルトン
- integratePokemon()/integratePokemonBatch(): ユーティリティ
- createDefaultIntegrationConfig(): 既定設定作成

エラー/方針
- IntegrationError（code: MISSING_ENCOUNTER_TABLE など）
- 統合サービスは JSON データのみを参照し「フォールバック無し」。テーブル未発見は例外。
- 同期（Synchronize）は野生系遭遇（EncounterType 0..7）のみ対象。静的/イベント/ローミングは対象外。

EnhancedPokemonData へのマッピング注意
- 現状の生成種族データには型や性別比の完全情報が未整備のため、以下は暫定値：
  - species.types: ['Normal'] のプレースホルダ
  - species.genderRatio: genderless のとき -1、それ以外は 0
 追って生成データ拡充時に正値へ置換予定。

## 4) 遭遇テーブル（`src/data/encounter-tables.ts`）

役割
- `getEncounterTable(version, location, method)` / `getEncounterSlot(table, slotValue)` / `calculateLevel(levelRandValue, range)` を提供

方針
- 統合サービスは JSON テーブル必須（フォールバック無し）。
- テスト/デモ向けのサンプルはアセンブラ側（後述）で提供。

出典（実装に合わせる）
- 遭遇テーブル: pokebook.jp（BW/BW2 の各ページ）

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
- サンプル遭遇テーブル（`createSampleEncounterTables()`）や同期ルール検証（`validateSyncRules()`）を含むデモ/テスト支援 API。
- 本番統合は `PokemonIntegrationService` を使用し、フォールバックは行いません。

主な関数
- createAssemblerContext(version, region, tables?)
- assembleData(ctx, raw) / assembleBatch(ctx, raws)
- setEncounterTable(ctx, type, table) / getEncounterTables(ctx)
- validateSyncRules(enhanced[]): 同期適用の適合性検証

## 使用例

WASM 生成 → 統合までの最小例：

```ts
import { getWasmPokemonService } from '@/lib/services/wasm-pokemon-service';
import { getIntegrationService } from '@/lib/services/pokemon-integration-service';

async function generateEnhancedPokemon(seed: bigint) {
  const wasmService = await getWasmPokemonService();
  const rawData = await wasmService.generateSinglePokemon({
    seed,
    config: {
      version: 'B', region: 'JPN', hardware: 'DS',
      tid: 12345, sid: 54321,
      syncEnabled: true, syncNatureId: 10,
      macAddress: [0, 0, 0, 0, 0, 0], keyInput: 0, frame: 1,
    },
  });

  const integration = getIntegrationService();
  const result = integration.integratePokemon(rawData, {
    version: 'B',
    defaultLocation: 'Route 1',
    applySynchronize: true,
    synchronizeNature: 10,
  });

  return result.pokemon;
}
```

## エラー処理

```ts
import { WasmServiceError } from '@/lib/services/wasm-pokemon-service';
import { IntegrationError } from '@/lib/services/pokemon-integration-service';

try {
  const p = await generateEnhancedPokemon(0x12345678n);
} catch (e) {
  if (e instanceof WasmServiceError) {
    // 初期化・生成時のエラー
  } else if (e instanceof IntegrationError) {
    // 遭遇テーブル未発見などの統合エラー
  }
}
```

## テスト

- 統合テスト: `src/test/phase2-integration.test.ts`
- アセンブラ検証: `src/test/integration/pokemon-assembler.test.ts`
- 生成サービス: `src/test/integration/wasm-service.test.ts`

## データ出典（現行実装に準拠）

- 技術資料: rusted-coil、xxsakixx（BW遭遇・乱数）
- 遭遇テーブル: pokebook.jp（BW/BW2 各ページ）
- 補助資料: 必要に応じて Bulbapedia など

## 付記（今後の拡張）

- 生成種族データの型/性別比/タイプ情報の拡充に合わせ、EnhancedPokemonData へのプレースホルダを正値化します。
- 隠れ特性はフラグ導入後に isHidden の正確化を予定。

