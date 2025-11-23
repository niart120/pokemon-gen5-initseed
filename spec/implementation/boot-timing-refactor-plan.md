# Boot Timing モード向けリファクタリング方針

## 1. 差分の俯瞰
`feature/boot-timing-draft` は以下の領域を中心に main ブランチから大きく乖離している。

- **UI**: `GenerationParamsCard.tsx` / `GenerationResultsTableCard.tsx` / `GenerationRunCard.tsx` / `ResultDetailsDialog.tsx`
- **状態管理**: `app-store.ts` / `generation-store.ts`
- **ドメインロジック**: `boot-timing-derivation.ts` / `generation-worker-manager.ts`
- **出力/検証**: `generation-exporter.ts` / `generation-exporter.test.ts`
- **仕様ドキュメント**: `spec/generation-boot-timing-mode.md`

これらを対象に、単一責任・コードスタイル・不要コードの観点で洗い出した課題とリファクタリング方針を以下に整理する。

## 2. リファクタリング課題と方針

### 2.1 `src/components/generation/GenerationParamsCard.tsx`
- **課題**
  - 800行超のコンポーネントに UI・状態同期・バリデーション補助・キー入力ダイアログが同居しており単一責任を満たしていない。
  - Boot Timing 専用 UI と LCG 共通 UI が同じ階層に混在し、条件分岐が増大して可読性が低下。
  - DeviceProfile 連動処理 (`profileSummaryLines` 等) がコンポーネント内に散在し、他画面から再利用不可。
- **方針**
  1. `GenerationParamsCard` を「表示レイアウト」とし、ロジックを `useGenerationParamsForm()`（共通）と `useBootTimingDraft()`（モード専用）に分離。
  2. Boot Timing 入力領域を `BootTimingControls`（日時・キー入力・プロフィール概要）として切り出し、キー選択ダイアログも独立コンポーネント化。
  3. DeviceProfile 連動は `app-store.applyProfileToGeneration` ではなく専用ユーティリティに集約し、View からは read-only の `bootTimingSnapshot` を受け取るだけにする。

### 2.2 `src/store/generation-store.ts` / `src/store/app-store.ts`
- **課題**
  - `generation-store.ts` が 1,000 行近くあり、派生Seedキュー制御・結果フィルタリング・UI補助の責務が混在。
  - Derived Seed 実行フロー (`derivedSeedState`, `_onWorkerComplete`) が Zustand slice 内に直接書かれておりテスト困難。
  - `app-store.applyProfileToGeneration` で Boot Timing 情報のマージを都度実装しており、同様の処理が他所に複製される恐れ。
- **方針**
  1. 派生Seed関連処理を `src/store/modules/boot-timing-runner.ts`（仮）へ移動し、Zustand からは純粋なサービスを呼び出す形に変更。
  2. `computeFilteredRowsCache` と UI補助（Encounter context 更新）を別ファイルへ分割し、slice から import する。
  3. DeviceProfile 反映ロジックを `bootTimingDraftFromProfile(profile, currentDraft)` のようなユーティリティに抽出して重複を排除。

### 2.3 `src/components/generation/GenerationResultsTableCard.tsx` / `src/components/search/results/ResultDetailsDialog.tsx`
- **課題**
  - Generation結果テーブルの行描画にロジック（Timer0/VCount整形、日時フォーマット、キー表示）が直書きされており、検索結果ダイアログでも同様の書式を再実装している。
  - 仮想スクロール行キーの生成に `row.advance` 以外の情報を付与したが、ヘルパー化されておらず他テーブルでは再利用不可。
  - ResultDetailsDialog が Generation ストアへ直接 `setDraftParams` を呼び出しており、検索 UI と生成 UI の関心事が密結合。
- **方針**
  1. Timer0/VCount/キー表示フォーマットを `@/lib/generation/result-formatters.ts` に集約し、テーブルとダイアログ双方で利用。
  2. 仮想スクロール行コンポーネント `GenerationResultRow` を作成し、キー生成や `ref` 設定もそこへ閉じ込める。
  3. ResultDetailsDialog からは `useBootTimingClipboard()` フック（generation store に依存した処理）を呼び出すだけにし、UI層の責務を明確化。

### 2.4 `src/lib/export/generation-exporter.ts`
- **課題**
  - `adaptGenerationResults` が UI用データ組み立てと CSV/TXT/JSON 変換ロジックの両方を担っており、出力形式追加時の影響範囲が大きい。
  - boot-timing メタ（Timer0/VCount/KeyInput/MAC）の整形が各フォーマッタに重複している。
  - 単体テストでは CSV header を固定配列でチェックしており、列追加時にテストと実装の同期コストが高い。
- **方針**
  1. `adaptGenerationResults` を「ドメイン→UI-ready」のみに限定し、書式変換は `csvFormatter`, `jsonFormatter`, `txtFormatter` へ分離。
  2. boot-timing 共通メタは `buildBootTimingMeta()` に一本化し、各フォーマッタには整形済み構造を渡す。
  3. テストでは `DISPLAY_COLUMN_COUNT` 等の定数を import して検証し、ヘッダの単純複製を避ける。

### 2.5 `src/lib/generation/generation-worker-manager.ts` / `src/lib/generation/boot-timing-derivation.ts`
- **課題**
  - Boot Timing 派生ロジックが `SeedCalculator` 依存と `GenerationParamsHex` 生成を同時に行っており、テスト時に SeedCalculator を差し替えにくい。
- **方針**
  1. `deriveBootTimingSeedJobs` を `messageBuilder`（SeedCalculator 依存）と `paramsBuilder`（GenerationParams 変換）に分割し、後者は純粋関数としてテスト可能にする。

