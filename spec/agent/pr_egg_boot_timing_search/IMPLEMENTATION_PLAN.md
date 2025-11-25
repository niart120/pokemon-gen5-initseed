# 実装計画・チェックリスト

## 概要

孵化乱数起動時間検索機能の実装を段階的に進めるための計画書。

## Phase 1: Rust WASM 実装

### 1.1 構造体・型定義

- [ ] `EggBootTimingSearchResult` 構造体定義
  - 起動条件フィールド (year, month, date, hour, minute, second, timer0, vcount, key_code)
  - LCG Seed フィールド (lcg_seed_high, lcg_seed_low)
  - 個体情報フィールド (advance, is_stable, ivs, nature, gender, ability, shiny, pid, hp_*)
  - wasm_bindgen ゲッター実装

- [ ] `EggBootTimingSearcher` 構造体定義
  - SHA-1 パラメータ (base_message, key_codes, allowed_second_mask, hardware)
  - 孵化条件 (conditions, parents, filter, consider_npc_consumption, game_mode)
  - 消費範囲 (user_offset, advance_count)

### 1.2 コンストラクタ

- [ ] パラメータバリデーション
  - MAC アドレス長検証
  - nazo 配列長検証
  - hardware 値検証
  - 時刻範囲検証

- [ ] 基本メッセージテンプレート構築
  - `IntegratedSeedSearcher` と同様のロジック

- [ ] キーコード生成
  - `generate_key_codes` 関数の流用

### 1.3 検索メソッド

- [ ] `search_eggs_integrated_simd` 実装
  - 日時ループ
  - Timer0/VCount/KeyCode ループ
  - SIMD バッチ処理
  - EggSeedEnumerator 呼び出し
  - 結果収集

- [ ] ヘルパーメソッド
  - `calculate_datetime_codes`
  - `build_message`
  - `generate_display_datetime`
  - `enumerate_eggs_for_seed`
  - `create_result`

### 1.4 lib.rs 更新

- [ ] `mod egg_boot_timing_search;` 追加
- [ ] `pub use egg_boot_timing_search::*;` 追加

### 1.5 テスト

- [ ] コンストラクタ検証テスト
- [ ] 検索結果テスト
- [ ] SIMD/スカラー一致テスト

## Phase 2: TypeScript 統合

### 2.1 型定義

- [ ] `EggBootTimingSearchParams` インターフェース
- [ ] `BootCondition` インターフェース
- [ ] `EggBootTimingSearchResult` インターフェース
- [ ] Worker 通信型 (Request/Response)
- [ ] 進捗・完了型
- [ ] バリデーション関数
- [ ] デフォルト値生成関数

### 2.2 Worker 実装

- [ ] WASM 初期化
- [ ] メッセージハンドラ
- [ ] 検索実行ロジック
- [ ] 結果変換関数
- [ ] 停止処理
- [ ] バッチサイズ対応

### 2.3 並列 WorkerManager 実装

既存の `MultiWorkerSearchManager` パターンを流用

- [ ] `EggBootTimingMultiWorkerManager` クラス
  - Worker プール管理 (Map<number, Worker>)
  - Worker 数設定 (setMaxWorkers / getMaxWorkers)
  - 並列検索開始 (startParallelSearch)
  - 一時停止/再開 (pauseAll / resumeAll)
  - 全Worker停止 (terminateAll)

- [ ] チャンク分割計算 (`boot-timing-chunk-calculator.ts`)
  - 日時範囲を Worker 数で分割
  - Timer0×VCount×KeyCode の組み合わせ数考慮
  - 処理量推定 (estimatedOperations)

- [ ] バッチサイズ制御
  - 組み合わせ数に応じた動的バッチサイズ計算
  - メモリと応答性のバランス

- [ ] 進捗集約
  - 各 Worker の進捗を統合 (AggregatedEggBootTimingProgress)
  - 残り時間推定 (calculateAggregatedTimeRemaining)
  - 進捗監視タイマー (500ms間隔)

- [ ] タイマー管理
  - 一時停止時間を除外した経過時間計算
  - TimerState による状態管理

- [ ] エラーハンドリング
  - Worker エラー時の部分継続
  - 全 Worker 失敗時の終了処理

### 2.4 単一 WorkerManager 実装（簡易版）

- [ ] `EggBootTimingWorkerManager` クラス
  - デバッグ用・簡易実行用
  - 並列処理不要時の軽量版

### 2.5 統合テスト

- [ ] WASM ローダーテスト
- [ ] Worker 通信テスト
- [ ] 結果変換テスト
- [ ] 並列実行テスト
- [ ] 一時停止/再開テスト

## Phase 3: UI 実装（将来）

### 3.1 コンポーネント

- [ ] `EggBootTimingSearchPanel`
- [ ] `EggSearchParamsCard`
- [ ] `EggSearchRunCard`
- [ ] `EggSearchResultsCard`

### 3.2 Store 統合

- [ ] Zustand store 拡張
- [ ] 状態管理
- [ ] 永続化

### 3.3 既存コンポーネント連携

- [ ] ProfileCard 連携
- [ ] EggParamsCard 流用
- [ ] EggFilterCard 流用

## 依存関係

```
Phase 1 (Rust)
    ↓
Phase 2 (TypeScript)
    ↓
Phase 3 (UI)
```

## ファイル変更リスト

### 新規作成

| ファイル | Phase | 説明 |
|----------|-------|------|
| `wasm-pkg/src/egg_boot_timing_search.rs` | 1 | Rust 検索器 |
| `src/types/egg-boot-timing-search.ts` | 2 | 型定義 |
| `src/workers/egg-boot-timing-worker.ts` | 2 | Worker |
| `src/lib/egg/boot-timing-egg-worker-manager.ts` | 2 | 単一 Worker Manager |
| `src/lib/egg/boot-timing-egg-multi-worker-manager.ts` | 2 | 並列 Worker Manager |
| `src/lib/egg/boot-timing-chunk-calculator.ts` | 2 | チャンク分割計算 |

### 更新

| ファイル | Phase | 変更内容 |
|----------|-------|----------|
| `wasm-pkg/src/lib.rs` | 1 | モジュールエクスポート追加 |
| `src/types/index.ts` | 2 | 型エクスポート追加 |
| `src/lib/egg/index.ts` | 2 | Manager エクスポート追加 |

## 既存コード流用

### Rust

| 流用元 | 流用内容 |
|--------|----------|
| `integrated_search.rs` | `generate_key_codes`, メッセージ構築ロジック |
| `egg_seed_enumerator.rs` | `EggSeedEnumerator` |
| `datetime_codes.rs` | `DateCodeGenerator`, `TimeCodeGenerator` |
| `sha1.rs`, `sha1_simd.rs` | SHA-1 計算 |
| `egg_iv.rs` | `GenerationConditions`, `IndividualFilter` 等 |

### TypeScript

| 流用元 | 流用内容 |
|--------|----------|
| `types/egg.ts` | 孵化関連型 |
| `types/search.ts` | 検索条件型 |
| `types/parallel.ts` | 並列処理型 (WorkerChunk, WorkerProgress, AggregatedProgress) |
| `lib/search/multi-worker-manager.ts` | 並列 Worker 管理パターン |
| `lib/search/chunk-calculator.ts` | チャンク分割計算パターン |
| `lib/utils/key-input.ts` | キー入力変換、組み合わせ数計算 |
| `lib/core/nazo-resolver.ts` | nazo 値解決 |

## 注意事項

1. **パフォーマンス**: 計算量が大きいため、Worker 並列化と SIMD 活用が必須
2. **メモリ**: 大量結果時は上限設定とバッチ送信が必要
3. **互換性**: 既存の `IntegratedSeedSearcher` との共通処理は将来的に共通モジュール化を検討
4. **テスト**: 既知の起動条件での検索結果を検証用テストケースとして使用
5. **並列処理**: `MultiWorkerSearchManager` のパターンを踏襲し、一貫性のある実装を維持

## 参考ドキュメント

- `/spec/agent/pr_egg_boot_timing_search/SPECIFICATION.md` - 全体仕様
- `/spec/agent/pr_egg_boot_timing_search/RUST_IMPLEMENTATION.md` - Rust 実装詳細
- `/spec/agent/pr_egg_boot_timing_search/TYPESCRIPT_DESIGN.md` - TypeScript 設計
- `/spec/implementation/egg-seed-enumerator.md` - EggSeedEnumerator 仕様
- `/spec/generation-boot-timing-mode.md` - Boot-Timing Mode 仕様
