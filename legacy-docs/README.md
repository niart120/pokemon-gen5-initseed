# Legacy Documents

アーカイブ済みの資料を保管するディレクトリです。現行仕様から外れた内容や履歴参照のみが目的の文書をここに配置します。

## 移管方針
- プロジェクトの最新仕様と矛盾する文書は `legacy-docs/` に移します。
- GitHub Pages 配信用の `docs/` にはビルド成果物のみを配置します。
- アーカイブ文書は編集対象外とし、参照が必要な場合は新しい資料に再整理してから反映します。

## 含まれている文書
- `PRD.md`: 初期段階の要求定義書。現行アーキテクチャと乖離しているためアーカイブ化。
- `IMPLEMENTATION_STATUS.md`: 過去の進捗報告。最新状況は README と `spec/` 配下の資料を参照してください。
- `ENCOUNTER_IMPLEMENTATION.md`: 旧エンカウント実装メモ。最新仕様は `spec/` と `src/lib/generation/` のコード、および対応するテストを参照してください。

### `spec/`
- `04-implementation-phases.md`: フェーズ計画の旧版。現行ロードマップは `todo.md` やアクティブな仕様書を参照。
- `GENERATION_UI_SCREENS.md`: Phase1 リファクタ時点のスクリーン資料。最新 UI はアプリケーションと `spec/pokemon-generation-ui-spec.md` を基準としてください。
- `generation-control-buttons-spec.md`: Generation 制御ボタンの Task 7 案。実装済み UI ガイドラインは `spec/implementation/ui-guidelines.md` に統合済み。
- `generation-params-layout-proposal.md`: Task 6 のレイアウト提案。現在のフォーム構成は `spec/pokemon-generation-feature-spec.md` と UI ガイドラインを参照。
