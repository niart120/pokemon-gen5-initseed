# ポケモンBW/BW2 初期Seed探索webアプリ

第5世代ポケモン（ブラック・ホワイト/ブラック2・ホワイト2）の初期Seed値探索・検証を行うwebアプリケーションです。

**🌐 アプリを使用する: https://niart120.github.io/pokemon-gen5-initseed/**

## 概要

このアプリケーションは、ポケモンBW/BW2における初期Seed値の効率的な探索を実現します。ROMバージョン、リージョン、ハードウェア、日時、キー入力といった条件から生成されるメッセージをSHA-1ハッシュ化し、その上位32bitを初期Seedとして算出します。

## 主な機能

- **全28バージョン対応**: BW/BW2の全バージョン・リージョン組み合わせをサポート
- **超高速探索**: WebAssembly SIMD128 + Rust による最適化で2.7億回/秒を実現
- **並列処理**: CPU数に応じたWebWorker並列化による高速化（実験的機能）
- **リアルタイム進捗**: 探索状況の詳細表示と中断・再開機能
- **結果管理**: ソート・フィルタリング・詳細表示機能
- **エクスポート**: CSV/JSON/テキスト形式での結果出力
- **包括的テスト環境**: Playwright-MCP によるE2Eテスト自動化、開発・統合テストページによる品質保証

## Generation 機能概要 (Phase3-4 MVP)

初期Seedから連続する乱数列を列挙し、ポケモン生成コア属性 (PID, 性格, 特性スロット, 色違い種別, スロット値, 同期適用) を WebWorker + WASM でストリーミング取得します。Search (SHA-1 初期Seed探索) と独立したタブ/ストア領域を持ち、相互に干渉しません。

### 主API / 主要構成
- GenerationWorkerManager: WebWorker ライフサイクル管理 (開始/停止/進捗/結果バッチ)
- generation-worker: WASM PokemonGenerator / SeedEnumerator 呼び出し
- Store Slice: processedAdvances, resultsCount, throughputRaw, throughputEma, etaMs, shinyCount を保持
- Exporter: CSV / JSON / TXT (BigInt は hex + decimal 併記 or 文字列化)

### 主パラメータ (GenerationParams 抜粋)
| Param | 説明 | 制約 |
|-------|------|------|
| baseSeed | 初期Seed bigint | 0 ≤ < 2^64 |
| offset | 開始オフセット | 0..maxAdvances |
| maxAdvances | 消費上限 | 1..1,000,000 |
| maxResults | 収集上限 | 1..100,000 且つ ≤ maxAdvances |
| tid / sid | 表/裏ID | 0..65535 |
| syncEnabled + syncNatureId | シンクロ設定 | natureId 0..24 |
| stopAtFirstShiny | 最初の色違いで停止 | boolean |
| stopOnCap | maxResults 到達停止 | default true |
| batchSize | バッチ生成数 | 1..10,000 |
| progressIntervalMs | 進捗間隔 | default 250ms |

### 完了理由 (reason)
- max-advances: 上限消費到達
- max-results: 収集上限 + stopOnCap
- first-shiny: 最初の色違い検出 + stopAtFirstShiny
- stopped: ユーザー停止
- error: 例外 / 検証失敗

### 進捗/性能指標
- throughputRaw = processedAdvances / 実行秒
- throughputEma = EMA(α=0.2) 平滑化
- etaMs = (remainingAdvances / basis) * 1000 (basis=Ema>0?Ema:Raw)

### Search との違い
| 項目 | Search | Generation |
|------|--------|------------|
| 目的 | 初期Seed探索(SHA-1) | Seed から遭遇結果列挙 |
| 計算単位 | メッセージ -> ハッシュ | RNG advance -> Pokemon Raw |
| 早期終了 | 条件Seed発見 | max / shiny / results / stop |
| 出力 | 初期Seed候補 | Pokemon Raw 属性列 |
| 性能指標 | Hash/s (SIMD) | Advances/s (EMA) |

詳細仕様: docs/GENERATION_PHASE3_4_PLAN.md, spec/pokemon-generation-feature-spec.md を参照。

## 技術スタック

- **フロントエンド**: React 18 + TypeScript + Vite
- **UI**: Radix UI (shadcn/ui) + TailwindCSS
- **計算エンジン**: Rust + WebAssembly (wasm-pack) + SIMD128最適化
- **状態管理**: Zustand
- **バックグラウンド処理**: Web Workers + 並列処理対応
- **パフォーマンス監視**: 本番用軽量監視 + 開発用詳細分析

### WebAssembly計算エンジン

本アプリケーションの計算処理は以下のRust WebAssemblyモジュールで実装されています：

- **IntegratedSeedSearcher**: 統合シード探索API（メイン検索エンジン）
- **PersonalityRNG**: BW/BW2仕様64bit線形合同法乱数生成器
- **EncounterCalculator**: 遭遇スロット計算エンジン（BW/BW2別対応）
- **OffsetCalculator**: ゲーム初期化処理とオフセット計算
- **PIDCalculator & ShinyChecker**: PID生成と色違い判定
- **PokemonGenerator**: 統合ポケモン生成エンジン



## 開発・ビルド・テスト

### 基本コマンド

```bash
# 依存関係のインストール
npm install

# 開発サーバー起動
npm run dev

# 開発サーバー起動（軽量モード・E2Eテスト用）
npm run dev:agent

# WebAssemblyビルド
npm run build:wasm

# プロダクションビルド
npm run build

# GitHub Pagesデプロイ
npm run deploy
```

### テスト・検証手順

#### 基本テスト実行

```bash
# TypeScriptテスト実行
npm run test

# Rustテスト実行（WASM単体）
npm run test:rust

# Rustブラウザテスト実行（WASM統合）
npm run test:rust:browser

# 全テスト実行（推奨）
npm run test:all
```

#### 開発・検証用テストページ

テストページでの詳細な動作確認・パフォーマンス測定：

```bash
# 開発サーバー起動後、ブラウザで以下にアクセス

# 開発テスト（個別機能・パフォーマンステスト）
http://localhost:5173/test-development.html

# 統合テスト（システム全体・ワークフローテスト）  
http://localhost:5173/test-integration.html

# SIMD機能テスト（SIMD最適化・パフォーマンス比較）
http://localhost:5173/test-simd.html
```

### 品質保証

本プロジェクトは包括的なテスト環境により品質を保証しています：

- **WASM単体テスト**: Rust Cargoテスト（95テスト以上）
- **TypeScript単体テスト**: Vitestベース
- **統合テスト**: WebAssembly-TypeScript連携テスト
- **ブラウザテスト**: wasm-packによる実環境テスト
- **E2Eテスト**: Playwright-MCPによる自動化テスト

## テスト環境

### 開発テスト
```bash
npm run dev
# → http://localhost:5173/test-development.html
```
- 個別機能のパフォーマンステスト
- WebAssembly統合テスト
- 詳細プロファイリング分析

### 統合テスト
```bash
npm run dev
# → http://localhost:5173/test-integration.html
```
- システム全体の統合テスト
- エンドツーエンドワークフローテスト
- ストレステスト・ベンチマーク

### 並列処理テスト
```bash
npm run dev
# → http://localhost:5173/test-parallel.html
```
- WebAssembly-Worker統合テスト
- 実環境並列処理検証
- メモリ管理・パフォーマンス測定

## 開発者向けメモ
- ユーティリティは `src/lib/utils/<module>` を明示的にインポート（バレル禁止）
    - 例: `import { toMacUint8Array } from '@/lib/utils/mac-address'`
- wasm-bindgen 生成物の直接参照は禁止。必ず `src/lib/core/wasm-interface.ts` を経由
- enum 変換の専用モジュールは廃止。必要な最小限の変換は各境界（Worker/Resolver）でローカル実装

## GitHub Copilot対応

このプロジェクトはGitHub Copilotの最適化された設定を含んでいます：

- `.github/copilot-instructions.md`: 基本的なプロジェクト情報
- `.github/instructions/`: ファイル固有の開発指示
- `.github/prompts/`: 再利用可能なプロンプト（実験的機能）
- `.github/copilot-meta.md`: AI Agent向けメンテナンス情報

### Copilot設定の構造
```
.github/
├── copilot-instructions.md        # リポジトリ全体の基本指示
├── instructions/                   # ファイル固有の指示（自動適用）
│   ├── development.instructions.md
│   ├── testing.instructions.md
│   └── debugging.instructions.md
└── prompts/                       # 手動選択可能なプロンプト
    └── *.prompt.md
```

## パフォーマンス詳細

### SIMD最適化による高速化
WebAssembly SIMD128命令を活用した4並列SHA-1処理により大幅な性能向上を実現：

- **統合探索（SIMD版）**: 約2.7億回/秒
- **従来版比較**: 約2.7倍の性能向上
- **並列処理との組み合わせ**: CPUコア数に応じてさらなる高速化

### ベンチマーク環境
- **CPU**: AMD Ryzen 9 9950X3D 16-Core Processor (16コア/32スレッド, 最大4.3GHz)
- **メモリ**: 64GB RAM
- **OS**: Windows 11 Pro
- **アーキテクチャ**: x64 (AMD64)
- **ブラウザ**: Chrome/Edge (WebAssembly SIMD128対応)

### 技術的特徴
- 4-way並列SHA-1ハッシュ計算
- WebAssembly SIMD128ベクトル命令最適化
- 効率的なバッチ処理アルゴリズム
- メモリ使用量の最適化

## 使用方法

1. ROMバージョン・リージョン・ハードウェアを選択
2. MACアドレスとキー入力を設定
3. 探索日時範囲を指定
4. 目標Seedリストを入力
5. 探索開始で高速検索を実行

## APIドキュメント

詳細なAPI仕様や使用例は以下を参照してください。

 - spec/implementation/phase2-api.md

## 型の境界と単一ソース

- Enumなどのドメイン概念は `src/types/domain.ts` を単一ソースとして利用します。
- 境界の生データ型は `src/types/pokemon-raw.ts`（snake_case）。ラベル付け等は `src/lib/generation/pokemon-resolver.ts` の `toUiReadyPokemon()` を利用します。
- 性格名（Nature）は `DomainNatureNames`（英語名）に集約しています。

## E2Eテスト

包括的なブラウザ自動化テストをPlaywright-MCPで実行できます：

```bash
# 開発サーバー起動
npm run dev

# E2Eテスト実行
# Playwright-MCPのコマンドを使用
```

詳細は以下のドキュメントを参照：
- [E2Eテスト実行手順](docs/E2E_TESTING_WITH_PLAYWRIGHT_MCP.md)
- [Playwright-MCPスクリプト集](docs/PLAYWRIGHT_MCP_SCRIPTS.md)

## データ出典とクレジット

### 技術資料
- ポケモン第5世代乱数調整: https://rusted-coil.sakura.ne.jp/pokemon/ran/ran_5.htm
- BW なみのり・つり・大量発生 野生乱数: https://xxsakixx.com/archives/53402929.html
- BW 出現スロットの閾値: https://xxsakixx.com/archives/53962575.html

### データソース
- ポケモン攻略DE.com: http://blog.game-de.com/pokedata/pokemon-data/ （種族データ）
- ポケモンの友 (Black): https://pokebook.jp/data/sp5/enc_b （遭遇テーブル）
- ポケモンの友 (White): https://pokebook.jp/data/sp5/enc_w （遭遇テーブル）
- ポケモンの友 (Black 2): https://pokebook.jp/data/sp5/enc_b2 （遭遇テーブル）
- ポケモンの友 (White 2): https://pokebook.jp/data/sp5/enc_w2 （遭遇テーブル）

- 本ツールは非公式であり、いかなる保証も行いません。データには誤りが含まれる可能性があります。ゲーム内結果での検証を推奨します。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照
