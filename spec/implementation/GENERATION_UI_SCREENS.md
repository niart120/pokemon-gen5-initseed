# Generation パネル スクリーン参照資料

最終統一 (Phase1 リファクタ後) の Generation UI 構成とキャプチャ配置ガイド。実際の画像ファイルは後続コミットで `docs/assets/` へ追加予定。ここでは名称・想定レイアウト・主要要素をテキストで定義し、差分レビュー時の参照とする。

## 画像ファイル命名規約
| 用途 | ファイル名 (例) | 備考 |
|------|-----------------|------|
| 全体レイアウト(デスクトップ) | `generation-overview-desktop.png` | 1200px幅程度 |
| 全体レイアウト(モバイル) | `generation-overview-mobile.png` | 375–430px幅想定 |
| コントロールカード | `generation-control-card.png` | 状態: idle |
| 進捗カード(実行中) | `generation-progress-running.png` | 進捗 30–60% 例 |
| パラメータカード(編集可能) | `generation-params-editable.png` | status=idle |
| パラメータカード(実行中ロック) | `generation-params-locked.png` | status=running |
| 結果コントロールカード | `generation-results-control.png` | フィルタ + ソートUI |
| 結果テーブル(フィルタ前) | `generation-results-table-full.png` | 10+件 |
| 結果テーブル(フィルタ後) | `generation-results-table-filtered.png` | shiny-only 等 |

## レイアウト概要 (デスクトップ想定)
```
┌───────────────────────────── Main Column A (left) ─────────────────────────────┐┌──── Side (right) ────┐
│ [Generation Control]  [Progress]                                                ││ 予備 (将来拡張)      │
│ [Parameters - scrollable]                                                       ││                      │
│ [Results Control]                                                               ││                      │
│ [Results Table - fills remaining height, scroll]                                ││                      │
└─────────────────────────────────────────────────────────────────────────────────┘└──────────────────────┘
```

モバイルでは縦一列スタック:
```
[Control]
[Progress]
[Parameters]
[Results Control]
[Results Table]
```

## 各カード要素詳細
### GenerationControlCard
| 要素 | 内容 |
|------|------|
| Start / Pause / Resume / Stop ボタン | 状態遷移に応じた表示 (最大2列折返し) |
| Status ライブテキスト | `Status: running (max-results)` 等 |
| バリデーションエラー | 赤テキスト (alert, polite) |

### GenerationProgressCard
| 要素 | 内容 |
|------|------|
| Status バッジ | Idle / Running / Paused ... |
| Progressバー | 全体進捗 (maxAdvances vs processed) |
| MetricsGrid | Advances / Results / Shiny / Throughput / ETA / Progress% |
| SR-only live region | 状態・件数・ETA 読み上げ |

### GenerationParamsCard
| セクション | 主フィールド |
|-----------|--------------|
| Basics | Version / BaseSeed / Offset |
| Limits | MaxAdvances / MaxResults / BatchSize |
| Trainer IDs | TID / SID / Sync Enabled / Sync Nature |
| Encounter | Encounter Type |
| Stop Conditions | Stop at First Shiny / Stop On Cap |

ロック時 (running/paused/starting): 全 input/select/checkbox disabled。

### GenerationResultsControlCard
| グループ | 内容 |
|---------|------|
| Export / Utility | CSV / JSON / TXT / Clear / Reset |
| Primary | Shiny Only / Sort field+order / Advance Range |
| Secondary | Nature ID list / Shiny type list |
| Live hint | 設定反映確認 (`Results filtering controls configured.`) |

### GenerationResultsTableCard
| 列 | 内容 |
|----|------|
| Advance | advance index |
| PID | 8桁 16進 (0x) |
| Nature | Name (英語) |
| Shiny | ShinyLabel |
| (将来拡張) | AbilitySlot / EncounterType 等 |

スクロール: ヘッダ sticky / zebra row / max-height 親カードに依存。

## カラーバリエーション / 状態例
| 状態 | 期待表示 |
|------|----------|
| idle | Start ボタンのみ / params editable |
| starting | Start ボタン `Starting…` disabled |
| running | Pause + Stop / params disabled / progress更新 |
| paused | Resume + Stop / params disabled |
| completed | Start 再表示 (結果維持) |
| error | Start 再表示 + エラー表示 (別カード拡張予定) |

## スクリーンショット取得推奨手順
1. 開発サーバー起動 `npm run dev`
2. Generation タブへ遷移 (初期 idle 状態 キャプチャ)
3. パラメータ設定 (MaxAdvances 中規模, BatchSize 適正) → Start
4. Running 状態 30-60% で Progress / Table / Results Control / Locked Params を個別キャプチャ
5. Pause → Resume → Completed も必要なら追加
6. `docs/assets/` へ PNG 追加 (最終コミットで差分レビュー)

## 変更履歴
- v1 (2025-08): 初版 (Phase1 リファクタ a11y 完了後) 作成
