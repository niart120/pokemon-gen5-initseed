# Generation Phase3/4 TODO (Revised)

目的: Generation MVP (Phase3) と最小性能向上 (Phase4 初期) を過剰実装や再発明なしで進める。

現在ブランチ: feature/generation-ui-worker-phase3-4

## 完了済み (ベース)
- 型/バリデーション: generation.ts (params, completion reasons)
- 固定進捗間隔: 250ms (FIXED_PROGRESS_INTERVAL_MS)
- Worker skeleton + Enumerator 初期統合 (batching 基本)
- Manager / Store 初期スライス (起動/停止イベント配線)

## Phase3 MVP Tasks (P0)
| ID | タスク | 内容/狙い | DoD (受入基準) |
|----|--------|-----------|----------------|
| A1 | 早期終了+Shiny+Result抽出 | max-results / first-shiny / stopOnCap 実装。RawPokemonData から必要フィールド抽出し GenerationResult 拡張 (advance, seed, pid, nature, ability_slot, encounter_type, encounter_slot_value, shiny_type, sync_applied)。 | 5種完了理由を強制発火テストし COMPLETE.reason 正常。バッチ欠損 (cumulativeResults != resultsCount) なし。shiny_type!=0 で first-shiny 終了。 |
| A2 | 状態機械確定 | PAUSE/RESUME/STOP シーケンスの妥当性 (再開禁止シナリオ含) | 代表 9 シーケンス自動テスト通過。異常遷移で例外/ハングなし。 |
| A3 | Throughput/ETA(EMA) | 瞬間 throughput を EMA(α=0.2) 平滑。ETA=remainingAdv / emaThroughput | 10k advances シミュレーションで ETA 変動幅 ±20% 以内 (最後 10% 区間除く)。 |
| A4 | BigInt 表示整形 util | seed / level_rand_value 16進/10進文字列化 helper | util 単体テスト 5 ケース (0, 中間, max, 前ゼロ埋め, 負で失敗) 通過。 |
| A5 | Shiny リファレンスベクトル & Worker 基本テスト | 既知シード/ID セットで shiny_type 検証 | 参照 10 ケース 100% 一致。 |
| B1 | Store 基本強化 | append + maxResults ガード + selectors(thruputEMA, etaFormatted, shinyCount) | maxResults=100 で 150 advances 処理時 results.length=100 & COMPLETE.reason='max-results' |
| B2 | Export 既存流用アダプタ | GenerationResult → Exporter 入力 shape 変換 | 10 サンプルで CSV/JSON 出力非空 & 例外なし。 |
| C1 | UI 基本実装 | Tab, BasicCard, RangeCard+Presets(Fast/Normal/Exhaustive), ActionBar(Start/Stop/Progress), ResultsTable(Advance/PID/Nature/Shiny), Shiny Only toggle, 状態表示 | test-development.html で Start→完了 / Stop / Shiny 停止 3 シナリオ成功。 |
| D1 | Integration Test (HTML) | 上記 3 シナリオ自動化 | スクリプト実行で pass=3/3 |
| D2 | Docs 更新 | プロトコル & README Generation 章 | docs PR 差分でフィールド/理由リスト網羅。 |

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
| F2 | メモリ計測/閾値策定 | 100k results Heap 増加が線形 & 閾値レポート文書化 |
| F3 | Throughput monitor UI (dev only) | 開発モードで直近 10s グラフ表示 |
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
