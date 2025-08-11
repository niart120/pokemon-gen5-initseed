# Resolver 直結リファクタリング計画（UI互換モジュール撤去まで）

目的: UI 層から互換モジュール（`src/types/pokemon-ui.ts`）を段階的に排除し、WASM境界 → snake_case Raw → Domain Resolver → UI 専用薄い表示変換、の単一路線へ統合する。

---

## 全体方針（アーキテクチャ）
- WASM境界: `parseWasmLikeToRawPokemonData()`（snake_case Raw を返す）を唯一の境界とする。
- Domain 層: `pokemon-resolver.ts` に集約（純粋・決定的、I/O 副作用なし、バッチ API 提供）。
- UI 層: 表示名・i18n・ラベル変換のみ（構造やビジネスロジックは持たない）。
- 互換モジュール: `src/types/pokemon-ui.ts` は暫定の shim。最終的に撤去。

---

## マイルストーン分割と DoD

### M1: Services を resolver 出力に乗せ替え
- 対象
  - `src/lib/services/wasm-pokemon-service.ts`
  - `src/lib/services/pokemon-integration-service.ts`
- 変更
  - 返却を `snake_case Raw` もしくは `ResolvedPokemonData` に統一。
  - 互換 API（`parseRawPokemonData` 等）呼び出しを廃止し、`raw-parser.ts` + `pokemon-resolver.ts` を直結。
- テスト
  - 既存テスト継続緑（互換 shim 経由の期待値と一致）。
  - 新規: resolver 経由サービスの最小 E2E（1系統）
- DoD
  - services から互換モジュールの import が 0 件。
  - `npm run -s lint && npm run -s test` PASS。

### M2: UI コンポーネントの受け取り型を `UiReadyPokemonData` に統一
- 対象
  - `src/components/**`（検索・結果表示・詳細）
- 変更
  - UI が `pokemon-ui.ts` の camelCase Raw に依存している箇所を、`toUiReadyPokemon()` の結果に切替。
  - 表示用の name 解決はドメイン由来のテーブル/enum（`domain.ts`）へ寄せる。
- テスト
  - スナップショットまたはレンダリングテストの更新。
- DoD
  - UI 層から `pokemon-ui.ts` import が 0 件。

### M3: Assembler 重複排除
- 対象
  - `src/lib/integration/pokemon-assembler.ts`
- 変更
  - 参照するロジックを `pokemon-resolver.ts` に移譲し、Assembler を薄いパススルー（必要なら非推奨 deprecate）に変更 → 撤去。
- テスト
  - 参照先の差し替えでテスト緑を維持。
- DoD
  - Assembler ファイル撤去、または空実装＋利用 0 件。

### M4: 互換モジュール撤去
- 対象
  - `src/types/pokemon-ui.ts`
- 前提
  - repo 全体で import 0 件。
- 変更
  - ファイル削除、型・関数の置き換え完了。
- DoD
  - `grep -R "pokemon-ui" src/` で 0 件。
  - Lint/Tests PASS。

---

## 実施チェックリスト（詳細）

- 依存の見える化・段階的切替
  - [ ] `pokemon-ui.ts` の全使用箇所を CI で検出できるよう ESLint ルール/禁則コメントを追加（暫定）。
  - [ ] `pokemon-ui.ts` の全エクスポートに `/** @deprecated Use resolver */` を付与。

- Services 層
  - [ ] `wasm-pokemon-service.ts` が `parseWasmLikeToRawPokemonData` → `resolvePokemon/resolveBatch` の直列パイプを返す。
  - [ ] `pokemon-integration-service.ts` も同様に統一。
  - [ ] `ResolutionContext` の生成と DI（encounter tables、species、abilities、gender ratios）。初期化時に一度構築し、Zustand またはシングルトンで共有。

- UI 層
  - [ ] 受け取り型を `UiReadyPokemonData`（resolver 提供）へ変更。
  - [ ] 表示名は `domain.ts` のテーブルと `toUiReadyPokemon()` の補助に限定。

- Assembler
  - [ ] Assembler 内の分岐・決定ロジックを resolver に集約。
  - [ ] Assembler import 0 件後、削除。

- 互換モジュール削除
  - [ ] `pokemon-ui.ts` の import 0 件確認。
  - [ ] 削除後の Lint/Tests PASS。

---

## 変更影響ファイル（初期見立て）
- Services
  - `src/lib/services/wasm-pokemon-service.ts`
  - `src/lib/services/pokemon-integration-service.ts`
- Integration/Domain
  - `src/lib/integration/pokemon-resolver.ts`（機能補強）
  - `src/lib/integration/raw-parser.ts`（必要なら型整備のみ）
  - `src/lib/integration/pokemon-assembler.ts`（撤去）
- UI
  - `src/components/**`（結果リスト、詳細、オプション依存）
- Types/Data
  - `src/types/domain.ts`（名称テーブル・enum の参照一元化）
  - `src/data/**`（ResolutionContext 供給元）

---

## 追加実装（resolver 補強）
- `ResolutionContext` 拡充
  - species/gender/abilities/encounters の参照をまとめて持つ。
  - 初期化ビルダー: `buildResolutionContext()` を `src/lib/initialization/` に追加。
- resolver API
  - `resolvePokemon(raw, ctx)` / `resolveBatch(raw[], ctx)`
  - `toUiReadyPokemon(resolved, locale?)`
- 代表エッジケース
  - 性別不明/単性種
  - 隠れ特性の存在有無
  - 群れ/特殊遭遇のレベル境界
  - 乱数境界（0/最大値）と shiny 判定（全タイプ）

---

## 品質ゲート
- Build/Lint/Tests
  - `npm run -s lint` PASS
  - `npm run -s test` PASS（既存 200+ ケース + resolver 追加テスト）
- パフォーマンス
  - resolver バッチの Throughput が既存と同等以上（±5% 以内目標）

---

## リリース/PR 戦略
- ブランチ: `feat/resolver-rearchitecture`（継続）
- PR 分割
  - PR1: M1（Services 直結 + 最小テスト）
  - PR2: M2（UI 統一）
  - PR3: M3（Assembler 撤去）
  - PR4: M4（互換モジュール削除 + クリーンアップ）
- それぞれ CI 緑、変更範囲ごとにレビュー容易性を確保。

---

## ロールバック方針
- 各 PR は独立に revert 可能。
- `pokemon-ui.ts` は M3 まで温存し、万一の rollback 先として利用。
- services での切替は export レベルの差し替えで 1 点反転可能にしておく。

---

## 着手順（次アクション）
1. `pokemon-ui.ts` の全エクスポートに `@deprecated` を付与し ESLint 禁則（import 監視）を追加。
2. `buildResolutionContext()` を `src/lib/initialization/` に追加（Data 層からの収集 + キャッシュ）。
3. `wasm-pokemon-service.ts` を resolver 直結に切替（UI 返却は `toUiReadyPokemon`）。
4. 影響 UI を 1 画面ずつ `UiReadyPokemonData` へ移行（PR2）。
5. Assembler を参照 0 件にして撤去（PR3）。
6. 互換モジュール削除（PR4）。

---

## 進捗メモ（2025-08-11）

- `WasmPokemonService` に非破壊で resolver 連携 API を追加（M1 地ならし）:
  - `generateSnakeRawPokemon` / `generateSnakeRawBatch`: snake_case の `RawPokemonData` を直接返却。
  - `generateResolvedPokemon` / `generateResolvedBatch`: `ResolutionContext` を受け取り resolver で解決。
  - `generateUiReadyPokemon`: ラベル付与のみ（`toUiReadyPokemon`）。
- 既存の UI 互換 API（`generateSinglePokemon` / `generatePokemonBatch`）は据え置きで互換維持。
- Lint/Test 緑（Test Files 23 passed, Tests 241 passed）。

次の差分（継続 M1）:
- サービス利用側で新 API を優先採用する呼び出し口を追加（段階移行）。
- `pokemon-integration-service.ts` の内部依存を resolver 直列に寄せ、`pokemon-ui.ts` 依存を減らす。
- 互換モジュールの import 件数を計測し、M4 の削除条件に反映。
