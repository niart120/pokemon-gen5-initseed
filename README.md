# ポケモンBW/BW2 初期Seed探索 Web アプリ

第5世代（ブラック・ホワイト/ブラック2・ホワイト2）向けの初期 Seed 探索と結果検証を行う Web アプリケーションです。Rust + WebAssembly による SIMD 最適化検索と、React 製 UI による検索条件管理・結果可視化を提供します。

## Live
- https://niart120.github.io/pokemon-gen5-initseed/

## 主な機能
- Rust WebAssembly コアによる高速 SHA-1 初期 Seed 探索（SIMD128 対応）
- Web Worker ベースの検索・世代引き離し処理と進捗監視
- Encounter/Species データを利用した Generation 結果解析（シードから連続乱数の列挙）
- Generation 結果のフィルタリング・ソート・エクスポート（CSV / JSON / TXT）
- WebGPU ベースの検索ランナー（対応ブラウザでの実験的パス）
- Vitest / wasm-pack / Playwright MCP を組み合わせた多層テスト

## アーキテクチャ概要
| レイヤ | 主なモジュール | 概要 |
| --- | --- | --- |
| UI (React + Zustand) | `src/components`, `src/store` | 検索条件・結果 UI、Zustand ストア |
| 検索コア | `src/lib/core`, `src/workers/search-worker*.ts` | wasm-bindgen 経由で `IntegratedSeedSearcher` を実行 |
| Generation | `src/lib/generation`, `src/workers/generation-worker.ts` | シード列挙・Pokemon Resolver |
| データ管理 | `src/data/encounters`, `src/data/species` | Encounter テーブル・種族データのローダ |
| Rust / wasm | `wasm-pkg/src/*.rs` | SHA-1 SIMD、Encounter 計算、Pokemon Generator 等 |

## セットアップ & 開発
```bash
npm install
npm run build:wasm        # 初回のみ wasm-pkg をビルド
npm run dev               # http://localhost:5173 で開発
```

検証ページ（開発サーバー上）
- http://localhost:5173/test-integration.html — 主要フロー統合テスト

ビルド / デプロイ
```bash
npm run build             # wasm + TypeScript をビルド
npm run deploy            # dist → docs へコピー（GitHub Pages 用）
```

## テスト
| コマンド | 内容 |
| --- | --- |
| `npm run test` | Vitest (Node + happy-dom) |
| `npm run test:rust` | `wasm-pkg` の Cargo テスト |
| `npm run test:rust:browser` | wasm-pack によるブラウザ統合テスト |
| `npm run test:webgpu` | WebGPU モードのブラウザテスト (Vitest Browser) |
| `npm run test:e2e` | Playwright MCP を用いた一貫性チェック |
| `npm run test:all` | Rust → Browser → TypeScript の順に実行 |

## ドキュメント
- 設計・仕様: `spec/` 配下
  - `spec/pokemon-generation-feature-spec.md`
  - `spec/pokemon-data-specification.md`
  - `spec/pokemon-generation-ui-spec.md`
- テストガイド: `src/test/README.md`
- アーカイブ済み資料: `legacy-docs/` (`PRD.md`, `IMPLEMENTATION_STATUS.md`, `ENCOUNTER_IMPLEMENTATION.md`)

> `docs/` ディレクトリは GitHub Pages 配信用のビルド成果物です。ドキュメントは格納しないでください。

## データソース・参考
- ポケモン第5世代乱数調整: https://rusted-coil.sakura.ne.jp/pokemon/ran/ran_5.htm
- 遭遇テーブル: https://pokebook.jp/
- 補助資料: https://xxsakixx.com/

## ライセンス
MIT © 2025 niart120
