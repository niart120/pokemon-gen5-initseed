# ポケモン BW/BW2 初期Seed探索 WebApp

## はじめに
ユーザとの対話は日本語で行うこと。

## プロジェクト概要
ポケットモンスター ブラック・ホワイト/ブラック2・ホワイト2の初期Seed値探索・検証を行うwebアプリケーション

## 技術スタック
- **フロントエンド**: React 18 + TypeScript + Vite
- **計算エンジン**: Rust + WebAssembly (wasm-pack + wasm-bindgen) 
- **UI**: Radix UI components
- **状態管理**: Zustand
- **暗号処理**: Rust `sha1` crate

## フォルダ構造
- `/src`: TypeScript source code and React components
- `/wasm-pkg`: Rust WebAssembly implementation with SIMD optimization
- `/public`: Static assets and test pages
  - `/public/wasm`: WebAssembly modules for distribution
  - `/public/test-integration.html`: 統合テスト
  - `/public/test-simd.html`: SIMD機能テスト
- `/scripts`: Build automation scripts
- `/docs`: GitHub Pages deployment files
- `/spec`: プロジェクト仕様書と設計ドキュメント
  - `/spec/agent`: GitHub Copilot Agent専用ドキュメント配置先

## t_wada Principle
- Code      → How
- Tests     → What
- Commits   → Why
- Comments  → Why not

## アーキテクチャ原則
- **本番・開発コードの分離**: 本番環境に不要なコードを含めない
- **適切な責任分離**: パフォーマンス監視とデバッグ分析の分離
- **統合検索中心の設計**: IntegratedSeedSearcher.search_seeds_integrated_simd による一元化
- **テスト環境の整備**: 開発効率を高める包括的テストシステム
- **依存関係の整理**: 循環依存や不適切な依存を避ける

## コーディング規約
- TypeScript strict mode 使用
- React function-based components 使用
- IntegratedSeedSearcher による統合検索API使用 (個別WASM関数は廃止)
- ESLint/Prettier設定に準拠
- 技術文書は事実ベース・簡潔に記述

## ドキュメンテーション
- `/spec` フォルダに仕様書・設計ドキュメントを配置する
  - GitHub Copilot AgentがPRに紐づく実装を行う場合 `/spec/agent/pr_{PR番号}` に配置すること

## テストシステム

### 統合テスト (`public/test-integration.html`)
- システム全体の統合テスト
- エンドツーエンドワークフローテスト
- ストレステスト

### SIMD機能テスト (`public/test-simd.html`)
- SIMD最適化の動作確認
- パフォーマンス比較測定


## 主要スクリプト
### 開発
- `npm run dev`: 開発サーバー起動
- `npm run dev:agent`: Agent/E2Eテスト用軽量モード（詳細ログ無効化）
- `npm run dev:full`: WASM再ビルド込み開発サーバー起動

### ビルド
- `npm run build`: 本番ビルド
- `npm run build:wasm`: WASM最適化ビルド
- `npm run build:wasm:debug`: WASM デバッグビルド
- `npm run build:wasm:size`: WASM サイズ最適化

### デプロイ
- `npm run deploy`: GitHub Pages向けデプロイ（本番ビルド + docs配置）
- `npm run deploy:build`: 本番環境ビルド
- `npm run deploy:copy`: distからdocsへのファイルコピー

### テスト
- `npm run test`: TypeScript テスト実行
- `npm run test:watch`: TypeScript テスト監視
- `npm run test:rust`: Rust テスト
- `npm run test:rust:browser`: Rust ブラウザテスト
- `npm run test:all`: 全テスト実行
- `npm run test:e2e`: E2Eテスト

### ユーティリティ
- `npm run copy:wasm`: WASMファイルコピー
- `npm run clean`: 全クリーンアップ
- `npm run lint`: ESLint実行
- `npm run preview`: プロダクションプレビュー
