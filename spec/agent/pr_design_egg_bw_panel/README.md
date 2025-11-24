# EggBWPanel 設計仕様書ディレクトリ

本ディレクトリには、EggBWPanel（タマゴ個体生成一覧表示機能）の設計仕様書が格納されています。

## ドキュメント一覧

### 1. [SUMMARY.md](./SUMMARY.md)
設計の要約と概要。プロジェクトの全体像を素早く把握できます。

**含まれる内容:**
- プロジェクト概要
- 主要機能
- アーキテクチャ概要
- 実装フェーズ
- 重要な設計判断

### 2. [SPECIFICATION.md](./SPECIFICATION.md)
詳細な設計仕様書。実装に必要なすべての情報が記載されています。

**含まれる内容:**
- データ型定義（TypeScript）
- Worker実装仕様
- WorkerManager実装仕様
- UIコンポーネント設計
- 状態管理（Zustand）
- テスト戦略
- 実装順序
- 注意事項

### 3. [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)
実装者向けの詳細ガイド。段階的な実装手順とコードサンプルが含まれます。

**含まれる内容:**
- Phase別実装手順
- ステップバイステップガイド
- コードサンプル
- テストコード例
- デバッグ方法
- トラブルシューティング
- パフォーマンス最適化
- 国際化対応
- 実装チェックリスト

### 4. [ARCHITECTURE.md](./ARCHITECTURE.md)
システムアーキテクチャの図解と詳細説明。

**含まれる内容:**
- システム全体図
- データフロー図
- コンポーネント詳細図
- Worker通信プロトコル
- WASM連携詳細
- パフォーマンス最適化ポイント
- テストアーキテクチャ
- セキュリティと制約
- 拡張ポイント

## 読み進める順序

### 初めて読む方
1. **SUMMARY.md** - 全体像を把握
2. **ARCHITECTURE.md** - アーキテクチャを理解
3. **SPECIFICATION.md** - 詳細仕様を確認

### 実装者
1. **IMPLEMENTATION_GUIDE.md** - 実装手順を確認
2. **SPECIFICATION.md** - 必要に応じて詳細参照
3. **ARCHITECTURE.md** - 構造の理解が必要な時に参照

### レビュアー
1. **SUMMARY.md** - 概要確認
2. **SPECIFICATION.md** - 設計の妥当性評価
3. **ARCHITECTURE.md** - アーキテクチャ評価

## 設計の背景

### 目的
EggSeedEnumerator (Rust/WASM) のインタフェースに基づき、タマゴ個体の生成・フィルタリング・表示機能を実装する。

### 要件
- ユーザーがUIから初期Seed（または起動時間）、親個体値、親個体情報を入力
- 指定した消費範囲の個体一覧をテーブル表示
- フィルター機能（個体値範囲、特性、性格、性別、めざパ等）
- Panel内でモード切り替え可能（個体一覧表示 / 起動時間検索[WIP]）

### 設計方針
既存のGenerationPanel実装パターンに準拠:
- Worker ベースの非同期処理
- WorkerManager によるライフサイクル管理
- Zustand による状態管理
- 責任分離: Worker=計算、Manager=制御、UI=表示

## 技術スタック

- **TypeScript** - strict mode
- **React 18** - function components
- **Zustand** - 状態管理
- **Web Workers** - 非同期処理
- **WebAssembly (Rust)** - 高速計算
- **Radix UI** - UIコンポーネント
- **Vitest** - テストフレームワーク

## 関連ドキュメント

### プロジェクト内
- `/spec/implementation/06-egg-iv-handling.md` - タマゴIV仕様
- `wasm-pkg/src/egg_seed_enumerator.rs` - Rust実装
- `wasm-pkg/src/egg_iv.rs` - タマゴIV計算
- `src/workers/generation-worker.ts` - 既存Worker実装パターン
- `src/lib/generation/generation-worker-manager.ts` - 既存Manager実装パターン
- `src/components/layout/GenerationPanel.tsx` - 既存Panel実装パターン

### リポジトリルール
- `.github/instructions/development.instructions.md` - 開発ベストプラクティス
- `.github/instructions/rust-wasm.instructions.md` - Rust WebAssembly開発ガイド
- `.github/instructions/testing.instructions.md` - テスト実行ガイド

## 実装状況

本設計仕様書は**設計フェーズ**のものです。実装は別PRで実施予定です。

### Phase 予定
- **Phase 1**: 型定義とWorker基盤
- **Phase 2**: 状態管理（Zustand）
- **Phase 3**: UIコンポーネント
- **Phase 4**: 統合とテスト

## 質問・フィードバック

仕様に関する質問や改善提案がある場合は、Issue またはPRコメントでお願いします。

## ライセンス

本プロジェクトのライセンスに従います。
