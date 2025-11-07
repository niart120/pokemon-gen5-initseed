# Generation Params Card レイアウト改善提案 (Task 6)

目的: search側カードガイドライン (ui-guidelines.md) に整合した情報構造/視覚構造へ再設計し可読性・操作性・拡張性を確保。

## 1. 情報アーキテクチャ再編
| 新セクション | 含める項目 | 目的 | 備考 |
|--------------|-----------|------|------|
| Basics | Version / Base Seed / Offset | 検索基点 | Base Seed/Offset を version 直後に集約 |
| Limits | Max Advances / Max Results / Batch Size | 計算境界 & 制御 | Batch Sizeは内部処理チューニング。説明テキスト追加余地 |
| Trainer IDs | TID / SID | 光り判定関連 | 二つを横並び (TID,SID) で視認性向上 |
| Encounter | Encounter Type / Sync Enabled / Sync Nature | 出現テーブル/性格補正 | Sync NatureはSync Enabled true時のみ活性化 |
| Stop Conditions | Stop at First Shiny / Stop On Cap | 終了条件 | Check列集約。今後追加(例えば時間制限)拡張余地 |

## 2. 視覚レイアウト方針
- Card: `py-2` / Header `pb-0` / Content `pt-0 space-y-3`
- Section: 見出し行 + グリッド
  - 見出し: `<div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground flex items-center gap-2">` + 必要ならアイコン (統一ポリシー上必須ではない)
  - グリッド: `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3` (自動折返しで高さ均衡) ただし短い check 系は flex 行内配置
- Input Wrapper: 共通 `flex flex-col gap-1` / Label `text-xs font-medium text-muted-foreground` / Control `w-full`
- Hex/数値入力: shadcn `<Input />` (`type="text"` or `number`), `className="font-mono"` (Hexのみ)
- Select: shadcn `<Select>` コンポーネント (後続リファクタで導入)
- Checkbox: shadcn `<Checkbox>` + `<label htmlFor>` ラップ (現行は `input type=checkbox`)
- 可変幅の排除: 既存 `w-28 / w-36 / w-40` を撤廃しグリッドセル幅依存 (一行 = 100%)
- スクロール: 高さ肥大対策として Card 親コンテナ側で `min-h-0` を確保; ParamsCard 自体は不要な max-height 付与しない (将来項目追加で縦伸長 OK)。必要なら周囲レイアウトで列高バランス調整。

## 3. ワイヤーフレーム (テキスト)
```
Card
  Header: [ Generation Parameters ]
  Content (space-y-3)
    Section Heading: BASICS
      [Version  ][Base Seed        ][Offset]
    Section Heading: LIMITS
      [Max Advances][Max Results][Batch Size]
    Section Heading: TRAINER IDs
      [TID][SID]
    Section Heading: ENCOUNTER
      [Encounter Type][Sync Enabled ◻][Sync Nature]
    Section Heading: STOP CONDITIONS
      [☑ Stop at First Shiny][☑ Stop On Cap]
```
小画面 (<640px): 1カラム縦並び
中画面 (≥640px): 2カラム
広画面 (≥1024px): 3カラム (Trainer IDs は2要素で1行埋まる)

## 4. コンポーネント構造案 (擬似コード)
```tsx
<Card className="py-2 flex flex-col gap-2">
  <CardHeader className="pb-0"><CardTitle className="flex items-center gap-2 text-base">Generation Parameters</CardTitle></CardHeader>
  <CardContent className="pt-0 space-y-3">
    <ParamSection title="Basics">
      <VersionSelect />
      <HexInput name="baseSeedHex" label="Base Seed" />
      <HexInput name="offsetHex" label="Offset" />
    </ParamSection>
    <ParamSection title="Limits">
      <NumberInput name="maxAdvances" label="Max Advances" />
      <NumberInput name="maxResults" label="Max Results" />
      <NumberInput name="batchSize" label="Batch Size" />
    </ParamSection>
    <ParamSection title="Trainer IDs">
      <NumberInput name="tid" label="TID" />
      <NumberInput name="sid" label="SID" />
    </ParamSection>
    <ParamSection title="Encounter">
      <EncounterTypeSelect />
      <CheckboxField name="syncEnabled" label="Sync Enabled" />
      <SelectField name="syncNatureId" label="Sync Nature" disabled={!syncEnabled} />
    </ParamSection>
    <ParamSection title="Stop Conditions">
      <CheckboxField name="stopAtFirstShiny" label="Stop at First Shiny" />
      <CheckboxField name="stopOnCap" label="Stop On Cap" />
    </ParamSection>
  </CardContent>
</Card>
```
`ParamSection` は見出し + `div.grid` をラップ。Heading は `aria-level` 制御不要 (Card内部階層) なので `<div role="group" aria-labelledby>` でセクション識別可。

## 5. 相互依存/バリデーション仕様
| フィールド | 依存 | ルール |
|------------|------|--------|
| Sync Nature | Sync Enabled | Sync Enabled=false の場合 disabled & value保持 (内部値は保持) |
| Batch Size | なし | 0以下は1に正規化 (実装時) |
| Max Advances / Max Results | なし | 負値は0に正規化 |
| Base Seed / Offset | 16進 | 入力フィードバック: 無効文字拒否 (現行踏襲) |

## 6. アクセシビリティ
- セクション: `<div role="group" aria-labelledby={id}>` + 見出し要素に `id`
- Checkbox: `<Checkbox id=... /> <label htmlFor=...>` でクリック領域拡大
- 入力説明 (必要なら): `<p id="field-hint" className="text-xs text-muted-foreground">` + `aria-describedby`

## 7. Tailwind/クラス標準化
| 要素 | クラス例 |
|------|---------|
| Section Grid | `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3` |
| Label | `text-xs font-medium text-muted-foreground` |
| Input Root Wrapper | `flex flex-col gap-1` |
| Checkbox Row (短い) | `flex items-center gap-2` |

## 8. 移行ステップ (最小リスク)
1. 補助コンポーネント (ParamSection, Field wrappers) 追加 (GenerationParamsCard内ローカル実装) 
2. 既存フィールドを段階的に ParamSection へ移動 (フィールド名/ロジック未変更) 
3. shadcn UI (Input/Select/Checkbox) への置換 (スタイル崩れをスナップショット付きで検証) 
4. 可変幅 w-XX クラス削除・グリッド導入 
5. ラベル/タイトルタイポ修正 (text-sm→text-base 等) 
6. アクセシビリティ属性付与 & 簡易ユニットテスト 
7. Dead code (旧ラッパー) 削除 & eslint fix 

## 9. 期待効果 (測定可能指標)
- 見出し導入によりセクション境界視認時間 (主観) 減少 / DOM depth + クラス文字数軽微増減
- ラベル幅固定廃止による余白幅ばらつき解消: devtools計測で w-* 固定クラス 0 件
- 640–1024px 幅での縦スクロール領域短縮 (項目再配置による平均行数減少)

## 10. 次アクション
Task 7: Control ボタン仕様精緻化 → Task 8 実装リファクタ.

---
(本ファイルは Task 6 成果物)
