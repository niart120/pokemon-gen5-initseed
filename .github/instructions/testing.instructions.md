---
applyTo: "**/*.test.{ts,js}"
---

# テスト実行ガイド

## 新テストアーキテクチャ

### 開発テスト環境
> `public/test-development.html` は廃止済みです。

### 統合テスト環境
- **統合テストページ**: `public/test-integration.html`
  - システム全体の統合テスト
  - WebAssembly読み込みテスト
  - 検索エンジン統合テスト
  - パフォーマンス監視統合テスト
  - データパイプライン統合テスト
  - エンドツーエンドワークフローテスト

## テスト環境
- Vitest + happy-dom（ユニットテスト）
- Vitest Browser + Playwright Chromium（WebGPUテスト）
- WebAssemblyローダー: `initWasmForTesting`
- Node.js環境での実行

## 必須テストカテゴリ
1. **Rust側Unit Test**: `npm run test:rust` (または `cargo test`)
2. **Rust側Browser Test**: `npm run test:rust:browser` (または `wasm-pack test --chrome --headless`)
3. **TypeScript側Unit Test**: `npm run test`
4. **WebAssembly統合テスト**: `wasm-node.test.ts`
5. **開発環境テスト**: 廃止済み（利用不可）
6. **統合テスト**: `http://localhost:5173/test-integration.html`
7. **SIMDテスト**: `http://localhost:5173/test-simd.html`
8. **E2Eテスト**: mcp-playwright使用可能
9. **全テスト実行**: `npm run test:all`

## パフォーマンス計測
- `npx vitest run --config vitest.browser.config.ts src/test/webgpu/webgpu-runner-profiling.test.ts`
- `http://localhost:5173/test-simd.html`
- ブラウザの Performance / Memory ツール

## 品質維持
- TypeScriptテストと WebGPU ブラウザテストを継続的に実行
- Rust テスト（`npm run test:rust` / `npm run test:rust:browser`）で wasm 側の整合性を確認
- パフォーマンス結果を既存ログと比較し回帰を検知
- 本番コードとテストコードの依存関係を再点検

## E2Eテスト・ブラウザ自動化
- **mcp-playwright**: ブラウザ操作・スクリーンショット・UI検証に利用可能
- フロントエンドの動作確認・UI regression テストに活用
- 新テストページの自動実行にも対応
- WebAssembly統合テストの自動化
