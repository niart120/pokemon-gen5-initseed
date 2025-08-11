# Resolver 直結リファクタリング計画（UI互換モジュール撤去まで）

目的: UI 層から互換モジュール（`src/types/pokemon-ui.ts`）を段階的に排除し、WASM境界 → snake_case Raw → Domain Resolver → UI 専用薄い表示変換、の単一路線へ統合する。


## 全体方針（アーキテクチャ）


## マイルストーン分割と DoD

### M1: Services を resolver 出力に乗せ替え
  - `src/lib/services/wasm-pokemon-service.ts`
  - `src/lib/services/pokemon-integration-service.ts`
  - 返却を `snake_case Raw` もしくは `ResolvedPokemonData` に統一。
  - 互換 API（`parseRawPokemonData` 等）呼び出しを廃止し、`raw-parser.ts` + `pokemon-resolver.ts` を直結。
  - 既存テスト継続緑（互換 shim 経由の期待値と一致）。
  - 新規: resolver 経由サービスの最小 E2E（1系統）
  - services から互換モジュールの import が 0 件。
  - `npm run -s lint && npm run -s test` PASS。

### M2: UI コンポーネントの受け取り型を `UiReadyPokemonData` に統一
  - `src/components/**`（検索・結果表示・詳細）
  - UI が `pokemon-ui.ts` の camelCase Raw に依存している箇所を、`toUiReadyPokemon()` の結果に切替。
  - 表示用の name 解決はドメイン由来のテーブル/enum（`domain.ts`）へ寄せる。
  - スナップショットまたはレンダリングテストの更新。
  - UI 層から `pokemon-ui.ts` import が 0 件。

### M3: Assembler 重複排除
  - `src/lib/integration/pokemon-assembler.ts`
  - 参照するロジックを `pokemon-resolver.ts` に移譲し、Assembler を薄いパススルー（必要なら非推奨 deprecate）に変更 → 撤去。
  - 参照先の差し替えでテスト緑を維持。
  - Assembler ファイル撤去、または空実装＋利用 0 件。

### M4: 互換モジュール撤去
  - `src/types/pokemon-ui.ts`
  - repo 全体で import 0 件。
  - ファイル削除、型・関数の置き換え完了。
  - `grep -R "pokemon-ui" src/` で 0 件。
  - Lint/Tests PASS。


## 実施チェックリスト（詳細）

  - [ ] `pokemon-ui.ts` の全使用箇所を CI で検出できるよう ESLint ルール/禁則コメントを追加（暫定）。
  - [ ] `pokemon-ui.ts` の全エクスポートに `/** @deprecated Use resolver */` を付与。

  - [ ] `wasm-pokemon-service.ts` が `parseFromWasmRaw` → `resolvePokemon/resolveBatch` の直列パイプを返す。
  - [ ] `pokemon-integration-service.ts` も同様に統一。
  - [ ] `ResolutionContext` の生成と DI（encounter tables、species、abilities、gender ratios）。初期化時に一度構築し、Zustand またはシングルトンで共有。

  - [ ] 受け取り型を `UiReadyPokemonData`（resolver 提供）へ変更。
  - [ ] 表示名は `domain.ts` のテーブルと `toUiReadyPokemon()` の補助に限定。

  - [ ] Assembler 内の分岐・決定ロジックを resolver に集約。
  - [ ] Assembler import 0 件後、削除。

  - [ ] `pokemon-ui.ts` の import 0 件確認。
  - [ ] 削除後の Lint/Tests PASS。


## 変更影響ファイル（初期見立て）
  - `src/lib/services/wasm-pokemon-service.ts`
  - `src/lib/services/pokemon-integration-service.ts`
  - `src/lib/integration/pokemon-resolver.ts`（機能補強）
  - `src/lib/integration/raw-parser.ts`（必要なら型整備のみ）
  - `src/lib/integration/pokemon-assembler.ts`（撤去）
  - `src/components/**`（結果リスト、詳細、オプション依存）
  - `src/types/domain.ts`（名称テーブル・enum の参照一元化）
  - `src/data/**`（ResolutionContext 供給元）


## 追加実装（resolver 補強）
  - species/gender/abilities/encounters の参照をまとめて持つ。
  - 初期化ビルダー: `buildResolutionContext()` を `src/lib/initialization/` に追加。
  - `resolvePokemon(raw, ctx)` / `resolveBatch(raw[], ctx)`
  - `toUiReadyPokemon(resolved, locale?)`
  - 性別不明/単性種
  - 隠れ特性の存在有無
  - 群れ/特殊遭遇のレベル境界
  - 乱数境界（0/最大値）と shiny 判定（全タイプ）


## 品質ゲート
  - `npm run -s lint` PASS
  - `npm run -s test` PASS（既存 200+ ケース + resolver 追加テスト）
  - resolver バッチの Throughput が既存と同等以上（±5% 以内目標）


## リリース/PR 戦略
  - PR1: M1（Services 直結 + 最小テスト）
  - PR2: M2（UI 統一）
  - PR3: M3（Assembler 撤去）
  - PR4: M4（互換モジュール削除 + クリーンアップ）


## ロールバック方針


## 着手順（次アクション）
1. `pokemon-ui.ts` の全エクスポートに `@deprecated` を付与し ESLint 禁則（import 監視）を追加。
2. `buildResolutionContext()` を `src/lib/initialization/` に追加（Data 層からの収集 + キャッシュ）。
3. `wasm-pokemon-service.ts` を resolver 直結に切替（UI 返却は `toUiReadyPokemon`）。
4. 影響 UI を 1 画面ずつ `UiReadyPokemonData` へ移行（PR2）。
5. Assembler を参照 0 件にして撤去（PR3）。
6. 互換モジュール削除（PR4）。


## 進捗メモ（2025-08-11）

  - `generateSnakeRawPokemon` / `generateSnakeRawBatch`: snake_case の `RawPokemonData` を直接返却。
  - `generateResolvedPokemon` / `generateResolvedBatch`: `ResolutionContext` を受け取り resolver で解決。
  - `generateUiReadyPokemon`: ラベル付与のみ（`toUiReadyPokemon`）。

次の差分（継続 M1）:
