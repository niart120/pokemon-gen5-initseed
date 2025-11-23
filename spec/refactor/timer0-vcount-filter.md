# Timer0/VCount Filter リファクタリング指針

## 1. 差分対象概要
- `src/components/generation/GenerationResultsControlCard.tsx`
  - Timer0/VCount フィルタ UI の追加と既存フィールド幅の再調整。
- `src/store/generation-store.ts`
  - `GenerationFilters` への新フィルタ追加と `applyFilters` 内での正規化処理拡張。
- `src/store/selectors/generation-results.ts`
  - Timer0/VCount の単一値フィルタ判定ロジック。
- `src/lib/i18n/strings/*`, `src/hooks/generation/useBootTimingDraft.ts`, `src/hooks/search/useResultDetailsClipboard.ts`
  - 表示テキストの整理とコピー機能の微調整。

## 2. リファクタリング候補
### 2.1 Hex フィルタ入力ハンドラの重複 (GenerationResultsControlCard)
- **問題**: Timer0/VCount の `onChange` ハンドラがほぼ同一実装で 2 箇所に散在。
- **影響**: 入力検証ルール変更時に二重修正が必要となり単一責任を損なう。
- **方針**: `createHexFilterHandler(fieldKey)` のような小さな factory を導入し、`applyFilters({ [fieldKey]: next })` を共通化する。合わせて placeholder/`maxLength` 等の定義も `const HEX_FILTER_FIELDS = [...]` で集約し、UI レイアウト定義とセットで保守しやすくする。

### 2.2 `applyFilters` 内のインライン正規化関数 (generation-store)
- **問題**: `normalizeRange`/`normalizeFilterText` が `applyFilters` 呼び出しごとに生成され、別用途で使い回せない。
- **影響**: 可読性が下がり、他のフィルタ追加時に同等ロジックを再記述するリスク。
- **方針**: ファイル先頭に正規化ユーティリティを切り出し (`normalizeNumericRange`, `normalizeHexFilterText` 等) してテスト可能にする。`applyFilters` は入力→ユーティリティ呼び出しのみに縮退させ、責任を「状態マージ」に限定する。

### 2.3 Hex 正規化ロジックの重複 (generation-store vs selectors)
- **問題**: `normalizeFilterText` と `parseSingleFilter` が似た処理を別々に保持。片方はトリム＋Upper 化、もう片方はさらに 0x 除去と数値化を実施。
- **影響**: 形式変更時に 2 箇所のロジックを同期する必要があり、バグ混入余地が大きい。
- **方針**: 既存のユーティリティに正規化ロジックがないかを確認。なければ `src/lib/utils/hex-filter.ts` (仮称) を用意し、`normalizeHexInput(value, { clampMax })` を定義。Store では「UI 表示用に整形済み文字列を保持」、Selector では同ユーティリティで `parseHexInput()` を実施して数値化する。こうすることで検証ルールを単一点に集約できる。

## 3. 推奨タスク
1. Hex 入力フィールド用の小さな UI ヘルパー/コンポーネントを追加し、Timer0/VCount 双方から利用する。
2. `generation-store.ts` に Hex 文字列正規化ユーティリティを移設し、`applyFilters` を責務ごとに分割する。
3. Selector 側の `parseSingleFilter` を前述ユーティリティに置き換え、UI/Store/Selector で同じ規約を共有する。
