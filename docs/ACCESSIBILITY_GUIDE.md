# アクセシビリティ & 統一UI ガイド

本プロジェクトの Generation / Search UI で採用しているアクセシビリティ (a11y) 方針と具体的実装規約をまとめる。実装時はここを基準にレビューする。

## 目的
- 主要操作 (開始/停止/フィルタ/エクスポート) へキーボード & スクリーンリーダーから一貫したアクセスを提供
- コンポーネント差異によるラベル欠落やロールの不統一を排除
- 自動テスト (jest-axe) による回帰検出

## 基本原則
1. 可視テキスト or aria-labelledby により全ての操作可能要素に判別可能な名前
2. 情報構造は landmark/region + heading 階層で明示
3. 動的更新 (進捗/状態) は aria-live="polite" で通知 (過剰多発を避け簡潔テキスト)
4. 視覚レイアウトと DOM 順序の乖離を最小化 (並び替え視覚効果のみを CSS で行わない)
5. テーブルは caption + scope="col" を使用し、データセルは単純構造 (rowgroup化不要)

## カード構造規約
| 要素 | 規約 |
|------|------|
| Card ルート | `role="region"` (フォーム編集主体の場合 `role="form"`) + `aria-labelledby` でヘッダ参照 |
| Header タイトル | `<span id="{card-id}-title">Title</span>`; 汎用ヘッダコンポーネントは `title: ReactNode` 受け取り可 |
| ボタン群 | 意味的まとまり単位で `role="group"` (例: エクスポート/ユーティリティ) |
| ライブ領域 | 画面リフロー不要な短い要約 (例: 進捗率 / 状態) を `sr-only` + `aria-live` |

### ID 命名
`{機能}-{目的}` 形式 (例: `gen-progress-title`, `sort-field-label`). Screen Reader 専用テキストは `*-label` / `*-hint` をサフィックス。

## フォーム要素
- `Label htmlFor` に必ず対応 ID を持つ入力/トリガ
- Radix SelectTrigger は内部テキストが無い場合があるため、`<SelectValue />` を子として挿入 + `aria-labelledby="<label-id> <trigger-id>"`
- Checkbox は可視ラベルが隣接する場合でも確実に `htmlFor` で関連付け。アイコンのみボタンは禁止。

### Select 実装パターン
```tsx
<Label id="lbl-version" htmlFor="version-select">Version</Label>
<Select value={value} onValueChange={setValue}>
  <SelectTrigger id="version-select" aria-labelledby="lbl-version version-select" size="sm">
    <SelectValue />
  </SelectTrigger>
  <SelectContent>{/* <SelectItem/> */}</SelectContent>
</Select>
```

### Checkbox パターン
```tsx
<Checkbox id="stop-first-shiny" aria-labelledby="lbl-stop-first-shiny" checked={...} />
<Label id="lbl-stop-first-shiny" htmlFor="stop-first-shiny">Stop at First Shiny</Label>
```

## テーブル規約
```tsx
<table>
  <caption className="sr-only">Generation results</caption>
  <thead>
    <tr><th scope="col">Advance</th>...</tr>
  </thead>
  <tbody>{/* rows */}</tbody>
</table>
```
- ソートやフィルタ状態はテーブル直上のコントロール群に集約 (ヘッダセルへ aria-sort を付与するのは “現在列” のみが明確な場合に限定; 現行実装は未使用)

## ライブリージョン
| 用途 | 例 | 更新頻度 | 備考 |
|------|----|----------|------|
| 進捗 | `Processed 12,345 advances (34%)` | 250–1000ms 間隔 | 高頻度抑制 (batch/EMA 計算後) |
| 状態 | `Generation paused` | 状態遷移時のみ | 遷移単語のみ |
| フィルタ適用結果 | `Results filtering controls configured.` | 適用時 | verbose 避ける |

## jest-axe テスト
`src/test/generation/a11y-generation-cards.test.tsx` にて主要カードスナップショットを統合検査。

追加コンポーネントを導入したら:
1. `render()` に含める
2. axe ルール違反が出ればラベル/ロール修正

### ローカル実行
```bash
npm run test -- --runTestsByPath src/test/generation/a11y-generation-cards.test.tsx
```

## カラーデザイン / コントラスト
- テキストと背景は WCAG AA (4.5:1) を満たす Tailwind テーマトークンを優先使用
- 状態バッジなど低コントラスト要素は視覚的補助 (アイコン/太字) で意味を補強

## 変更フローチェックリスト
| # | チェック | OK |
|---|----------|----|
| 1 | インタラクティブ要素に名前 (label / aria-labelledby / aria-label) | [] |
| 2 | カード root に `aria-labelledby` でヘッダ参照 | [] |
| 3 | Radix SelectTrigger 内に `<SelectValue />` | [] |
| 4 | ライブ領域は冗長でない & polite | [] |
| 5 | テーブルに caption + scope | [] |
| 6 | グルーピング (fieldset / role=group) 適切 | [] |
| 7 | jest-axe で 0 違反 | [] |

## 今後の拡張候補
- カラムソート状態に応じた `aria-sort` 実装
- 進捗メトリクスへ `meter` 要素適用検討
- 主要操作ボタンへのキーボードショートカット (ARIA visible shortcut hint)

---
v1 (2025-08) 初版作成: Generation パネル Phase1 リファクタ a11y 対応結果を反映。
