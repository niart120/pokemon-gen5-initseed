# ポケモン生成機能 核心アルゴリズム実装

> Status: Active. 各詳細ドキュメント (`algorithms/*.md`) を正とし、ここでは概要と参照先のみを整理します。

## 概要

このドキュメントは、ポケモン BW/BW2 の初期Seed探索・検証機能における核心アルゴリズムの実装仕様を定義します。アルゴリズムの詳細は機能別に分割されたファイルに記載されています。

## アルゴリズム構成要素

### 4.1 性格値乱数列エンジン
**ファイル**: [algorithms/personality-rng.md](./algorithms/personality-rng.md)

BW/BW2仕様の64bit線形合同法による乱数生成エンジンの実装。

**主要機能**:
- S[n+1] = S[n] * 0x5D588B656C078965 + 0x269EC3
- シンクロ判定用乱数生成
- 性格決定用乱数生成
## アルゴリズム構成要素

### Personality RNG
**ファイル**: [algorithms/personality-rng.md](./algorithms/personality-rng.md)

BW/BW2 仕様の 64bit 線形合同法による乱数生成エンジン。Seedの更新式、シンクロ判定用の補助乱数、エンカウントスロット計算用のバリエーションを定義しています。

### Encounter Calculator
**ファイル**: [algorithms/encounter-calculator.md](./algorithms/encounter-calculator.md)

エンカウントスロット計算とエンカウントタイプ別確率分布の実装。通常・なみのり・釣り・特殊各種の分布、およびスロット値からテーブルインデックスへの変換をまとめています。

### Offset Calculator
**ファイル**: [algorithms/offset-calculator.md](./algorithms/offset-calculator.md)

ゲーム起動からポケモン生成直前までの乱数消費をモデル化。Probability Table 操作、TID/SID 決定、ブラックシティ/ホワイトフォレストの住人決定、BW2 固有処理などをカバーします。

### Pokemon Generator
**ファイル**: [algorithms/pokemon-generator.md](./algorithms/pokemon-generator.md)

`RawPokemonData` の構造とバッチ生成フローを記述。WASM と TypeScript の責務分担、性能最適化、レベル計算プレースホルダーが含まれます。

### PID & Shiny Checker
**ファイル**: [algorithms/pid-shiny-checker.md](./algorithms/pid-shiny-checker.md)

エンカウントタイプ別 PID 生成、色違い判定、シンクロ適用範囲、乱数消費パターンの違いを整理します。

### Special Encounters
**ファイル**: [algorithms/special-encounters.md](./algorithms/special-encounters.md)

揺れる草むらや砂煙など特殊エンカウント固有の仕様を定義。エンカウントテーブル、レベル計算方針、アイテムドロップ、隠れ特性の扱いを記録しています。
- 不要な計算の回避
- メモリ効率の良いデータ構造

### 正確性の保証
- ゲーム内アルゴリズムの正確な再現
- 乱数消費パターンの厳密な管理
- バージョン間の差異の適切な実装

## 技術的依存関係

- **wasm-bindgen**: WASM/JavaScript間のバインディング
- **WebAssembly**: 高性能な数値計算処理
- **64bit整数演算**: BW仕様の線形合同法に必要
- **TypeScript**: エンカウントテーブル管理と詳細レベル計算
---

**作成日**: 2025年8月3日  
**バージョン**: 1.0  
**作成者**: GitHub Copilot  
**依存**: pokemon-generation-feature-spec.md, pokemon-data-specification.md  
**更新**: 2025年8月3日 - アルゴリズム詳細を機能別ファイルに分割

## 分割ファイル一覧

詳細な実装仕様は以下のファイルを参照：

- [algorithms/personality-rng.md](./algorithms/personality-rng.md) - 性格値乱数列エンジン
- [algorithms/encounter-calculator.md](./algorithms/encounter-calculator.md) - エンカウント計算エンジン  
- [algorithms/offset-calculator.md](./algorithms/offset-calculator.md) - オフセット計算エンジン
- [algorithms/pokemon-generator.md](./algorithms/pokemon-generator.md) - ポケモンデータ構造と統合Generator
- [algorithms/pid-shiny-checker.md](./algorithms/pid-shiny-checker.md) - 性格値・色違い判定
- [algorithms/special-encounters.md](./algorithms/special-encounters.md) - 特殊エンカウント仕様
- [algorithms/README.md](./algorithms/README.md) - 分割ファイル構成の概要
