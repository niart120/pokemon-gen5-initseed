# Copilot Instructions メンテナンス情報

## ✅ GitHub公式推奨に準拠した構造
```
.github/
├── copilot-instructions.md        # 基本的なリポジトリ情報（30行以内）
├── instructions/                   # ファイル固有の指示
│   ├── development.instructions.md
│   ├── testing.instructions.md
│   ├── rust-wasm.instructions.md
│   └── debugging.instructions.md
```

## 🔄 定期レビューチェックリスト
- [ ] copilot-instructions.mdが30行以内か
- [ ] Instructions Filesに適切な`applyTo`が設定されているか
- [ ] アンチパターンが混入していないか
- [ ] 「short, self-contained statements」が維持されているか
- [ ] 不要な装飾表現（絵文字・感情的修辞）が除去されているか

## 📚 参考情報
- GitHub公式ドキュメント: https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions?tool=vscode

## 🎯 メンテナンス原則
1. **追加より削除を優先** - 簡潔性が最重要
2. **汎用性を保つ** - 特定タスク向けの内容は避ける
3. **公式推奨準拠** - GitHub公式ドキュメントを最優先
4. **検証後適用** - 変更後は必ず動作確認

## 📝 変更履歴
### v2.0
- GitHub公式ドキュメント完全準拠に修正
- Prompt Files → Instructions Files に移行
- copilot-instructions.mdを29行に簡潔化
- applyToフロントマターの適切な使用

### v1.0
- 詳細な作業指示を含む302行版
- GitHubアンチパターンを含んでいた
