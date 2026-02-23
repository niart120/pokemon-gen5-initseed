# ポケモン生成機能 実装仕様書 - 目次

## 実装ドキュメント構成


# 実装仕様ドキュメント


### 📁 `/spec/implementation/`

1. **[01-architecture.md](./01-architecture.md)** - アーキテクチャ設計
   - 全体構成設計
   - WASM-TypeScript データインターフェース
   - モジュール設計と責任分離
   - データフロー設計

2. **[02-algorithms.md](./02-algorithms.md)** - 核心アルゴリズム実装
   - 性格値乱数列エンジン（WASM実装）
   - エンカウント計算エンジン（WASM実装）
   - オフセット計算エンジン（WASM実装）
   - 統合Pokemon Generator（WASM実装）
   - 性格値・色違い判定の詳細実装

3. **[03-data-management.md](./03-data-management.md)** - データ管理実装
   - Generation Data Manager（TypeScript側）
   - 種族データ・エンカウントテーブル・特性データ管理
   - データ整合性チェック
   - ゲーム定数管理

4. **[05-webgpu-seed-search.md](./05-webgpu-seed-search.md)** - WebGPU Seed 検索計画
   - GPU ランナー構成
   - Worker 連携戦略
   - バッチ計画とフォールバック
   - テスト計画と導入ステップ
5. **[06-egg-iv-handling.md](./06-egg-iv-handling.md)** - タマゴ個体値 Unknown 仕様
   - Unknown(32) のデータ表現
   - resolve/match/hidden power の演算ルール
   - テスト観点と実装メモ

## 関連ドキュメント

- **[pokemon-generation-feature-spec.md](../pokemon-generation-feature-spec.md)** - 機能仕様書
- **[pokemon-data-specification.md](../pokemon-data-specification.md)** - データ仕様書
- **[algorithms/README.md](./algorithms/README.md)** - アルゴリズム詳細仕様の目次
- **[phase2-api.md](./phase2-api.md)** - WASM統合 API 指針
- **[ui-guidelines.md](./ui-guidelines.md)** - UI統一ガイドライン

> 旧フェーズ計画や UI 案の履歴は `legacy-docs/spec/` に移動済みです。

## 実装時の注意事項

1. **WASM中心アーキテクチャ**: 全ての計算ロジックはWASM側で実装し、TypeScript側はフォールバック実装を行わない
2. **64bit LCG正確性**: BW/BW2の正確な64bit線形合同法の実装が最優先
3. **エンカウントタイプ別実装**: 野生・固定シンボル・徘徊の各パターンを正確に再現
4. **段階的実装**: Phase 1（WASM Core）完了後にPhase 2以降を開始

---

**作成日**: 2025年8月3日  
**バージョン**: 1.0  
**作成者**: GitHub Copilot
