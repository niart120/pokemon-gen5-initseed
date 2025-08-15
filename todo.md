# Generation Phase3/4 TODO (Revised)

目的: Generation MVP (Phase3) と最小性能向上 (Phase4 初期) を過剰実装や再発明なしで進める。

現在ブランチ: feature/generation-ui-worker-phase3-4

## 完了済み (ベース)
- 型/バリデーション: generation.ts (params, completion reasons)
- 固定進捗間隔: 250ms (FIXED_PROGRESS_INTERVAL_MS)
- Worker skeleton + Enumerator 初期統合 (batching 基本)
- Manager / Store 初期スライス (起動/停止イベント配線)

## Phase3 Extended (P1)
| ID | タスク | DoD |
|----|--------|-----|
| E1 | Encounter 設定切替 (Wild/Static/Fishing) | 切替で params.encounterType 正しく更新 & 再生成反映 |
| E2 | 詳細行展開 (seed/pid/slot) | 行クリックで詳細表示 / キーボード Enter 対応 |
| E3 | ソート/追加フィルタ (nature/slot) | UI 操作で結果並び/件数変化が selector 経由で安定 |
| E4 | アクセシビリティ追補 | Table 行フォーカス移動 + live region 進捗読み上げ |
| E5 | 仮想スクロール | 50k 結果で初回レンダ <100ms (測定ログ) |

## Phase4 Performance (P2)
| ID | タスク | DoD |
|----|--------|-----|
| F1 | Chunk Enumerator 最適化 | main thread blocking frame >16ms 発生率 <1% (計測 60s) |
| F4 | Adaptive batch size | throughput 低下時 バッチ半減で回復 (5 サイクルテスト) |

## Future / Backlog (P3)
- G1 並列 Worker 構想メモ (multi-core scaling)
- G2 Progressive Export (streaming) ※ 大容量時のみ
- G3 高速/低メモリ モードスイッチ UI
- G4 Playwright E2E 拡張 (レスポンシブ + A11y)


## 実装順推奨 (クリティカルパス)
1. A1 → A2 → A3 (Worker 完全化)
2. B1/B2 (Store + Export)
3. C1 (UI 操作線) → A4/A5 (補助/テスト)
4. D1/D2 (検証 & ドキュメント)
5. 拡張 (E 系) は計測結果で優先度再評価

## 現在状態メモ
- Worker: batchIndex/cumulativeResults OK, early termination/shiny 未実装
- Store: results append 基本実装済 (maxResults guard 要再確認)
- UI: タブ/フォーム骨組み未着手


更新日時: 2025-08-12 (Revised)
