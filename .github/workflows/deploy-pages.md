# GitHub Pages 自動デプロイワークフロー

## 概要
このワークフローは `copilot/*` ブランチから `main` ブランチへのマージが完了したときに、自動的にGitHub Pagesへのデプロイを実行します。

## トリガー条件
- `copilot/*` ブランチ（ワイルドカード）から `main` ブランチへのプルリクエストがマージされたとき
- 例: `copilot/feature-a`, `copilot/fix-b` など、`copilot/` で始まる任意のブランチ

## 実行内容
1. `main` ブランチをチェックアウト
2. Node.js (v20) と Rust toolchain をセットアップ
3. wasm-pack をインストール
4. 依存関係をキャッシュから復元またはインストール
5. `npm run deploy` を実行
   - プロジェクトをビルド (`npm run build`)
   - `dist/` フォルダの内容を `docs/` フォルダにコピー
6. `docs/` フォルダの変更を `main` ブランチにコミット＆プッシュ

## PAT（Personal Access Token）について

### ✅ PATは不要です

このワークフローでは **`secrets.GITHUB_TOKEN`** を使用しています。これはGitHub Actionsが自動的に提供するトークンで、以下の理由から追加のPATは不要です：

1. **自動プロビジョニング**: `GITHUB_TOKEN` はワークフロー実行時に自動的に生成されます
2. **適切な権限**: `permissions: contents: write` により、リポジトリへの書き込み権限が付与されます
3. **セキュリティ**: ワークフロー実行後、トークンは自動的に無効化されます

### もしPATが必要な場合

以下のような特殊なケースでは、カスタムPATが必要になる可能性があります：

1. **他のリポジトリへのアクセスが必要な場合**
2. **GitHub Apps APIを使用する場合**
3. **ワークフロートリガーチェーンが必要な場合**（`[skip ci]` により回避済み）

その場合の設定方法：

#### PAT作成時に必要な権限
- `repo`: フルコントロール（コミット・プッシュに必要）
- または最小権限として：
  - `contents: write`: リポジトリコンテンツの書き込み

#### GitHub Secretsへの設定
1. GitHub リポジトリの Settings → Secrets and variables → Actions
2. "New repository secret" をクリック
3. Name: `DEPLOY_PAT` (任意の名前)
4. Value: 生成したPersonal Access Token
5. ワークフローファイルで `token: ${{ secrets.DEPLOY_PAT }}` に変更

## 注意事項

### コミットメッセージの `[skip ci]`
- デプロイコミットには `[skip ci]` が含まれています
- これにより、デプロイコミット自体が新たなワークフローをトリガーすることを防ぎます
- 無限ループを防止するための重要な設定です

### ブランチ保護ルール
もし `main` ブランチに保護ルールが設定されている場合：
- GitHub Actions による直接プッシュを許可する設定が必要です
- または、PAT を使用して保護ルールをバイパスする必要があります

## トラブルシューティング

### ワークフローが実行されない
- PR が `copilot/*` ブランチから作成されているか確認
- PR が実際にマージされているか確認（closeだけでは実行されません）

### プッシュに失敗する
- リポジトリの Actions 権限設定を確認
  - Settings → Actions → General → Workflow permissions
  - "Read and write permissions" が有効になっているか確認

### ビルドが失敗する
- `npm run deploy` がローカルで正常に実行できるか確認
- 依存関係が正しくインストールされているか確認
