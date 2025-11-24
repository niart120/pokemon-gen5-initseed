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

### 3. Unknown IV (32) の取り扱い
- 親IVに Unknown が含まれる場合、子に伝播
- フィルター範囲 {0, 32} で Unknown を許可

### 4. パフォーマンス最適化
- バッチ処理による結果送信
- 表示上限設定によるメモリ管理

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
