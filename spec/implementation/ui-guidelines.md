# UI 統一ガイドライン

対象: search 系・generation 系カードコンポーネント (Card/Control/Progress/Parameters/Results)。目的: 一貫した視認性・操作性・保守性を確保し、冗長なカスタムスタイルを削減。

## 1. タイポグラフィスケール
| 用途 | クラス | 備考 |
|------|--------|------|
| カードタイトル | `text-base font-medium` | 行内アイコン 20px 左配置 |
| セクション小見出し / 補助ラベル | `text-sm font-medium` | 例: グループラベル、統計ラベル |
| 本文 / 入力ラベル | `text-sm` | Shadcn `Label` に準拠 (ただし密度必要なら `text-xs sm:text-sm`) |
| メタ情報 / 補足テキスト | `text-xs text-muted-foreground` | 例: ヒント、説明文 |
| 数値/固定幅 | `font-mono text-sm` | 例: シード値、進捗カウンタ |
| ミニバッジ/ワーカーステータス | `text-[10px] font-medium` | 8–10px 範囲、過小さ回避 |

## 2. カードレイアウト
| 要素 | ルール |
|------|--------|
| Card | `py-2 flex flex-col gap-2 h-full` (高さ制限必要時 `max-h-96`) |
| CardHeader | `pb-0 flex-shrink-0` (追加余白不要) |
| CardTitle | `flex items-center gap-2 text-base` / 右側操作がある場合 `justify-between` |
| CardContent | 上端余白を詰める: 最初のブロックが直接配置されるなら `pt-0`; 内部は `space-y-2` or `space-y-4` |
| スクロール領域 | `flex-1 min-h-0 overflow-y-auto` をレイヤーの最深に1箇所のみ |

## 3. アイコン運用
| 用途 | サイズ | 方針 |
|------|--------|------|
| タイトル左 | 20px | 代表アイコン 1つまで (装飾乱立禁止) |
| ボタン内 | 14–16px | 前置 (左) に揃え、`mr-2` (最後尾不要) |
| ステータス/微小UI | 12px 以下 | 最小限。読解を阻害する場合はテキスト優先 |

## 4. ボタン Variant / 色設計 (shadcn `Button`)
| 状態/目的 | variant | 例 |
|-----------|---------|----|
| 主要開始アクション | default (=primary) | Start Search / Start Generation |
| 一時停止トグル | secondary | Pause |
| 再開 | default | Resume |
| 破壊的停止/削除 | destructive | Stop / Clear Results |
| 補助(インポート/エクスポート/設定) | outline | Import / Export / Reset |
| アイコンのみ/折畳み/詳細表示 | ghost size="sm" | 展開トグル |

サイズは原則 `size="sm"`。複数ボタン横並び時は主要ボタンのみ `flex-1` 拡張可。

## 5. 進捗表示
| 要素 | ルール |
|------|--------|
| コンポーネント | `<Progress />` を統一使用 (高さ: 通常 `h-2`, サブ `h-1.5`) |
| 数値表示 | パーセント + 現在値/総数 (mono) |
| 並列詳細 | 4列上限グリッド / モバイル2列固定 / ワーカー >32 は簡略モード |
| Live region | `sr-only` + `aria-live="polite"` で状態/進捗要約 (1秒以内大量更新は過剰発話回避: 既存ロジックを尊重) |

## 6. バッジ / ステータス
| 状態 | variant / 色 | テキスト |
|------|--------------|----------|
| 完了 | success系 (現在: `bg-green-100 text-green-800`) | Done |
| 実行中 | info系 (青) | Run |
| 一時停止 | gray/secondary | Paused |
| エラー | red/destructive | Error |
| ゼロ件/空 | `secondary` | 0 / Empty |

ワーカー詳細など多数表示箇所はミニサイズ (`text-[10px] px-1.5 py-0.5`)。

## 7. フォーム要素
| 要素 | コンポーネント | ルール |
|------|---------------|--------|
| テキスト/数値入力 | `Input` | 幅自動。特定幅必要時: ユーティリティクラスで最小化 (`w-24`, `w-32`) |
| テキストエリア | `Textarea` | 高さ制御: `min-h-20 max-h-48` + `resize-none` |
| セレクト | `Select` | ラベルは上部に `Label` コンポーネント。複数横並びは親 `flex gap-2` |
| チェックボックス | `Checkbox` | ラベルは `Label htmlFor` 使用。水平: `flex items-center space-x-1` |
| 非アクティブ | `disabled` 属性 + コンポーネント既定スタイルに依存 (独自彩色禁止) |

## 8. レイアウト (GenerationParamsCard 再構成指針)
| グループ | 内容 | レイアウト |
|---------|------|-----------|
| Basics | Version / Base Seed / Offset | 3列 (モバイル2列) |
| Limits | Max Advances / Max Results / Batch Size | 3列 |
| IDs | TID / SID | 2列 |
| Encounter | Encounter Type / Sync Enabled / Sync Nature | 行単位 (Sync有効時のみ Nature 有効) |
| Flags | StopAtFirstShiny / StopOnCap | 横並び |

コンテナ: `space-y-4` でグループ間余白。各グループ内部は `grid grid-cols-2 md:grid-cols-3 gap-2` などレスポンシブ。全体を `flex-1 min-h-0 overflow-y-auto` で縦スクロール。



## 12. コード共通化指針
| 項目 | 実装案 |
|------|--------|
| Card タイトル | `components/ui/app-card.tsx` にラッパ (任意) |
| ボタン集合 | 状態遷移ロジックは専用フック `useGenerationControlButtons()` |
| 進捗計算 | 既存 store selector 再利用 / 百分率 util `calcPercent(done,total)` |
| フォームグループ | 汎用 `FormGroup` (title, children) コンポーネント (装飾最小) |
