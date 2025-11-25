# EggBWPanel 設計サマリー

## プロジェクト概要
タマゴ個体生成一覧表示機能（EggBWPanel）の設計仕様書

## 目的
EggSeedEnumerator (Rust/WASM) のインタフェースに基づき、タマゴ個体の生成・フィルタリング・表示機能を実装する

## 主要機能
1. **パラメータ設定**
   - 初期Seed または 起動時間指定
   - 親個体値・親個体情報
   - かわらずのいし、性別比、種族特性
   - NPC消費考慮

2. **フィルター機能**
   - 個体値範囲 (各ステータス)
   - 特性、性格、性別
   - めざめるパワー (タイプ・威力)
   - 色違い判定

3. **結果表示**
   - 個体一覧テーブル
   - advance、IV、性格、性別等の表示
   - エクスポート機能 (予定)

## アーキテクチャ

### 技術スタック
- **Worker**: Web Workers による非同期処理
- **WASM**: Rust EggSeedEnumerator 実装
- **状態管理**: Zustand
- **UI**: React + Radix UI

### コンポーネント構造
```
EggBWPanel
├── EggParamsCard       (パラメータ入力)
├── EggFilterCard       (フィルター設定)
├── EggRunCard          (実行制御)
└── EggResultsCard      (結果表示)
```

### データフロー
```
UI (EggParamsCard)
  ↓ パラメータ設定
Zustand Store (egg-store)
  ↓ validateDraft / startGeneration
EggWorkerManager
  ↓ Worker作成・メッセージ送信
egg-worker (Worker)
  ↓ WASM呼び出し
EggSeedEnumerator (WASM)
  ↓ 個体列挙
egg-worker
  ↓ 結果送信
EggWorkerManager
  ↓ コールバック配信
Zustand Store
  ↓ 状態更新
UI (EggResultsCard)
```

## 実装フェーズ

### Phase 1: 型定義とWorker基盤
- `src/types/egg.ts` - 型定義
- `src/workers/egg-worker.ts` - Worker実装
- `src/lib/egg/egg-worker-manager.ts` - WorkerManager実装
- 単体テスト

### Phase 2: 状態管理
- `src/store/egg-store.ts` - Zustandストア
- ストアテスト

### Phase 3: UIコンポーネント
- `EggBWPanel.tsx` - レイアウト
- `EggParamsCard.tsx` - パラメータ入力
- `EggFilterCard.tsx` - フィルター設定
- `EggRunCard.tsx` - 実行制御
- `EggResultsCard.tsx` - 結果表示

### Phase 4: 統合とテスト
- WASM統合テスト
- E2Eテスト
- ドキュメント更新

## 重要な設計判断

### 1. WASM境界の明確化
- すべてのWASM呼び出しは `wasm-interface.ts` 経由
- メモリ管理 (`.free()`) を厳密に実施

### 2. BigInt シリアライゼーション
- Worker通信では BigInt → Number 変換
- UI では16進数文字列で管理

### 3. Unknown IV (32) の入力仕様
- **親個体IV入力**: 0-31 を基本とし、チェックボックスで Unknown (32) を設定
  - チェックボックス OFF: 数値入力フィールドで 0-31 を入力
  - チェックボックス ON: 入力無効化、値は自動的に 32 (Unknown)
- **フィルターIV範囲**: 0-31 を基本とし、チェックボックスで任意範囲指定を有効化
  - チェックボックス OFF: 範囲入力 0-31 のみ
  - チェックボックス ON: 範囲上限が 32 に強制設定され、Unknown を許可

### 4. パフォーマンス最適化
- バッチ処理による結果送信
- 表示上限設定によるメモリ管理

### 5. 起動時間関連機能（拡張設計）
起動時間に関連する機能として2つのモードが必要:

- **起動時間列挙モード**: 指定した起動時間候補（Timer0/VCount範囲）から個体を列挙
  - 既存の GenerationPanel の boot-timing モードと同様のアーキテクチャ
  - 同一 `egg-worker.ts` を使用し、DerivedSeedRunState で複数Seed候補を順次処理
  - 結果テーブルに Timer0/VCount 情報を表示

- **起動時間検索モード**: 条件を満たす個体が得られる起動時間を検索（SearchPanel類似）
  - 日時範囲・消費範囲内で目標条件を満たす起動時刻を逆算
  - 別途 `EggSearchPanel` として独立実装予定
  - 専用の `egg-search-worker.ts` と `EggSearchWorkerManager` を使用

### 6. BW2版 EggPanel（将来拡張設計）
⚠️ **重要**: BW2 のタマゴ生成ロジックは BW とは**根本的に異なる**ため、WASM レイヤーから完全に独立した実装が必要

- **LCG Seed 決定**: BW2 は完全に異なるロジック（未実装）
- **個体値決定/PID決定**: BW2 では独立したインタフェースを持つ（未実装）
- **WASM実装**: `EggBW2IVGenerator` + `EggBW2PIDGenerator` (仮称、未実装)
- **共通化不可**: BW と BW2 で `EggSeedEnumerator` を共有する設計は採用しない
- **UI共通化**: 結果表示形式が同じ場合、一部UIスタイルのみ共通化可能

## 参照ドキュメント

### 仕様書
- `SPECIFICATION.md` - 詳細設計仕様
- `IMPLEMENTATION_GUIDE.md` - 実装ガイド

### 関連仕様
- `/spec/implementation/06-egg-iv-handling.md` - タマゴIV仕様
- `wasm-pkg/src/egg_seed_enumerator.rs` - Rust実装
- `wasm-pkg/src/egg_iv.rs` - タマゴIV計算

### 既存実装パターン
- `src/workers/generation-worker.ts` - Worker実装パターン
- `src/lib/generation/generation-worker-manager.ts` - Manager実装パターン
- `src/components/layout/GenerationPanel.tsx` - Panel実装パターン

## 実装上の注意事項

### コーディング規約
- TypeScript strict mode 使用
- React function components 使用
- ESLint/Prettier 設定準拠
- 既存実装パターンとの一貫性維持

### テスト要件
- 単体テスト: 型、バリデーション、WorkerManager
- 統合テスト: Worker、WASM
- E2Eテスト: UI操作フロー

### 国際化
- すべてのUI文字列は i18n 対応
- `src/lib/i18n/strings/egg-*.ts` にラベル定義

## 次のステップ

1. 仕様書レビュー
2. Phase 1 実装開始
3. 逐次テスト・検証
4. Phase 2-4 の順次実装

## 連絡先・質問
実装中の質問や不明点は Issue またはコメントで確認
