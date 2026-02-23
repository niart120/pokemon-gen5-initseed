# ポケモン生成機能 仕様書インデックス

`spec/` 配下のドキュメントを目的ごとに分類し、現状ステータスと参照先を明確化します。アクティブな資料はここから辿れるように維持し、廃止済み資料は `legacy-docs/spec/` に移動しています。

## ドキュメント在庫

| カテゴリ | パス | ステータス | メモ |
| --- | --- | --- | --- |
| **コア仕様** | `pokemon-generation-feature-spec.md` | Active | Generation 機能のMVP要件・入力/出力定義。UI更新に合わせて随時同期する。 |
|  | `pokemon-data-specification.md` | Draft | 種族・エンカウントデータ構造の仕様。実データ生成フロー整備後に更新が必要。 |
| **実装仕様** | `implementation/README.md` | Active | 実装仕様書の入口。アーキテクチャ/アルゴリズム/データ管理の詳細を参照。 |
|  | `implementation/01-architecture.md` | Active | UI/サービス/WASM 層の責務分離とデータフロー。 |
|  | `implementation/02-algorithms.md` | Active | 核心アルゴリズムのサマリ。`implementation/algorithms/` の詳細ドキュメントと連携。 |
|  | `implementation/03-data-management.md` | Needs update | GenerationDataManager の仕様案。実装と照合しながら整備が必要。 |
|  | `implementation/05-webgpu-seed-search.md` | Planned | WebGPU 検索パス導入計画。実装着手時に最新構成へ更新する。 |
|  | `implementation/phase2-api.md` | Active | WASM 出力を UI 仕様へ統合する TypeScript API の指針。 |
|  | `implementation/ui-guidelines.md` | Active | Generation/Search 共通 UI コンポーネントのスタイル指針。 |
|  | `implementation/backlog.md` | Backlog | Generation 機能に関する未処理タスクの簡易メモ。 |
| **アルゴリズム詳細** | `implementation/algorithms/README.md` | Active | 各アルゴリズム仕様書の目次。 |
|  | `implementation/algorithms/*.md` | Active | Personality RNG / Encounter / Offset / PID / Pokemon Generator / Special Encounters の詳細仕様。 |

## アーカイブ済み資料

以下は混乱を避けるため `legacy-docs/spec/` へ移動しました。

- `04-implementation-phases.md`: 初期のフェーズ計画。現行ロードマップは `todo.md` やアクティブ仕様を参照。
- `GENERATION_UI_SCREENS.md`: Phase1 リファクタ時点のスクリーン資料。
- `generation-control-buttons-spec.md`: Generation 制御ボタンの Task 7 仕様案。
- `generation-params-layout-proposal.md`: Task 6 のレイアウト提案。

## 運用メモ
- 新しい仕様書を追加する際は、本インデックスへカテゴリとステータスを登録してください。
- 既存資料の内容が実装と乖離した場合は、更新または `legacy-docs/spec/` への移動を検討します。
- GitHub Pages への公開用ビルド成果物は `docs/` 内に置き、設計ドキュメントはここに集約します。
