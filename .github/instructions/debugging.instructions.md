---
applyTo: "**"
---

# デバッグ・問題解決ガイド

## 基本デバッグ手順
1. ブラウザ開発者ツール (F12) でConsole/Networkタブ確認
2. Workerログ（`search-worker.ts` / `parallel-search-worker.ts`）の進捗・エラー出力確認
3. Rust側ログ (`console_log!`) 確認

## 新テストシステムでのデバッグ

### 開発テスト
> `test-development.html` は廃止済みです。

### 統合テスト (`test-integration.html`)  
```bash
# 統合テスト実行
# http://localhost:5173/test-integration.html
# → Run All Integration Tests で包括テスト
```

### SIMD機能テスト (`test-simd.html`)
```bash
# SIMD最適化テスト実行
# http://localhost:5173/test-simd.html
# → SIMD vs 通常実装のパフォーマンス比較
```

## パフォーマンス分析

### 本番パフォーマンス監視
> 旧 `ProductionPerformanceMonitor` は廃止済みです。

### 開発詳細分析
> DevelopmentPerformanceAnalyzer は廃止済みです。

## Agent/E2Eテスト専用デバッグ

### Context圧迫回避
```bash
# 軽量モードでサーバー起動（ログ出力最小化）
npm run dev:agent

# 詳細ログが必要な場合のみ通常モード
npm run dev
```

### ログ出力制御
- `npm run dev:agent`: 検証ログを最小限に抑制（Context保護）
- `npm run dev`: 完全な検証ログ出力（開発・デバッグ用）

### Playwright-MCP使用時の推奨事項
1. 基本動作確認は軽量モードで実行
2. 問題発生時のみ詳細モードに切り替え
3. Context使用量監視（50,000文字で警告）

## よくある問題

### WebAssembly読み込み失敗
```bash
npm run build
```

### 計算結果の不整合
1. `npm run test` で TypeScript/Vitest を実行
2. `npx vitest run --config vitest.browser.config.ts src/test/webgpu/webgpu-runner-integration.test.ts` で WebGPU 統合動作を確認
3. `npm run test:rust` / `npm run test:rust:browser` で wasm 側の整合性を確認
4. `IntegratedSeedSearcher.search_seeds_integrated_simd` の結果を既知ケースと比較

### パフォーマンス劣化
- `npx vitest run --config vitest.browser.config.ts src/test/webgpu/webgpu-runner-profiling.test.ts` でワークロード別の測定を取得
- `test-simd.html`でSIMD最適化効果確認
- メモリリーク検査（Memory tab）

### UI表示・操作問題
- mcp-playwright: ブラウザ自動化によるUI動作確認
- スクリーンショット・要素検証による問題特定
- 新テストページでの自動操作テスト

### アーキテクチャ問題
- 本番コードとテストコードの依存関係確認
- 循環依存の検出・解決
- 適切な責任分離の維持

## 緊急時リセット
```bash
npm run clean && npm run build
```

## デバッグツール活用
- **WebGPUブラウザテスト**: `src/test/webgpu/*.test.ts`
- **SIMD機能テスト**: `test-simd.html`
- **ブラウザ自動化**: mcp-playwright による UI regression テスト
