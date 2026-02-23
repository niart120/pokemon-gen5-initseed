# KeyInput 仕様書

## 概要

ポケモンBW/BW2の初期Seed計算において、起動時のキー入力は重要なパラメータの一つである。本ドキュメントではKeyInput関連の型・変換ロジック・表示処理を整理する。

## 用語定義

| 用語 | 説明 | 値域 | 例 |
|------|------|------|-----|
| **keyMask** | 押下キーのビットマスク表現 | `0x0000` - `0x0FFF` | A+B = `0x0003` |
| **keyCode** | ゲーム内部で使用される値（`0x2FFF ^ keyMask`） | `0x2000` - `0x2FFF` | A+B = `0x2FFC` |
| **keyInputMask** | 検索対象キーのマスク（可能性のあるキー集合） | `0x0000` - `0x0FFF` | A,B,Start可 = `0x000B` |
| **keyNames** | キー名の配列 | `KeyName[]` | `['A', 'B']` |
| **keyInputDisplay** | 表示用文字列（ロケール依存） | `string` | `"A-B"` / `"None"` |

## キー定義

```typescript
// lib/utils/key-input.ts
const KEY_DEFINITIONS = [
  ['A', 0],      // bit 0
  ['B', 1],      // bit 1
  ['Select', 2], // bit 2
  ['Start', 3],  // bit 3
  ['[→]', 4],   // bit 4
  ['[←]', 5],   // bit 5
  ['[↑]', 6],   // bit 6
  ['[↓]', 7],   // bit 7
  ['R', 8],      // bit 8
  ['L', 9],      // bit 9
  ['X', 10],     // bit 10
  ['Y', 11],     // bit 11
] as const;

type KeyName = 'A' | 'B' | 'Select' | 'Start' | '[→]' | '[←]' | '[↑]' | '[↓]' | 'R' | 'L' | 'X' | 'Y';

const KEY_CODE_BASE = 0x2FFF;
const KEY_INPUT_DEFAULT = 0x0000;  // キー押下なし
```

### 不可能なキー組み合わせ

以下の組み合わせは物理的に同時押し不可として除外される：
- `[↑] + [↓]` 同時押し
- `[←] + [→]` 同時押し
- `Select + Start + L + R` 同時押し（ソフトリセット）

## 型定義

### types/input.ts（未使用・非推奨）

```typescript
// 注: 別定義が存在するが、実際には使用されていない
export const KEY_MAPPINGS = { A: 0, B: 1, ... };
export type KeyName = keyof typeof KEY_MAPPINGS;
```

### types/search.ts

| 型 | フィールド | 説明 |
|----|-----------|------|
| `SearchConditions` | `keyInput: number` | 検索対象キーマスク（keyMask形式） |
| `InitialSeedResult` | `keyCode: number \| null` | 結果のkeyCode |
| `InitialSeedResult` | `keyInputNames?: KeyName[]` | 解決済みキー名配列 |
| `SearchResult` | `keyInput?: number` | 軽量結果でのマスク値 |
| `SearchResult` | `keyCode?: number \| null` | 軽量結果でのkeyCode |
| `BootCondition` | `keyCode: number` | 起動条件のkeyCode |
| `BootCondition` | `keyInputNames: KeyName[]` | 起動条件のキー名配列 |

### types/generation.ts

| 型 | フィールド | 説明 |
|----|-----------|------|
| `GenerationResult` | `keyInputNames?: KeyName[]` | Boot Timingモード時のキー名 |

### types/pokemon-resolved.ts

| 型 | フィールド | 説明 |
|----|-----------|------|
| `UiReadyPokemon` | `keyInputDisplay?: string` | 表示用文字列（事前解決済み） |
| `UiReadyPokemon` | `keyInputNames?: KeyName[]` | キー名配列 |

### types/egg-boot-timing-search.ts / types/mt-seed-boot-timing-search.ts

| 型 | フィールド | 説明 |
|----|-----------|------|
| `*SearchParams` | `keyInputMask: number` | 検索対象キーマスク |
| `Wasm*Result` | `keyCode: number` | WASM結果のkeyCode |
| `*SearchResult.boot` | `keyCode: number` | ドメイン結果のkeyCode |
| `*SearchResult.boot` | `keyInputNames: KeyName[]` | 解決済みキー名配列 |

## 変換関数

### lib/utils/key-input.ts

| 関数 | 入力 | 出力 | 用途 |
|------|------|------|------|
| `keyMaskToNames(mask)` | keyMask | KeyName[] | マスク→名前配列 |
| `keyNamesToMask(names)` | KeyName[] | keyMask | 名前配列→マスク |
| `keyCodeToNames(keyCode)` | keyCode | KeyName[] | コード→名前配列 |
| `keyCodeToMask(keyCode)` | keyCode | keyMask | コード→マスク |
| `keyMaskToKeyCode(mask)` | keyMask | keyCode | マスク→コード |
| `generateValidKeyCodes(mask)` | keyInputMask | keyCode[] | 有効な全keyCode生成 |
| `normalizeKeyMask(mask)` | any mask | keyMask | 正規化 |
| `hasImpossibleKeyCombination(mask)` | keyMask | boolean | 不可能組み合わせ判定 |

### 内部変換ロジック

```
keyCode → keyMask: mask = keyCode ^ 0x2FFF
keyMask → keyCode: keyCode = mask ^ 0x2FFF
```

## 表示用変換

### lib/i18n/strings/search-results.ts

```typescript
function formatKeyInputDisplay(keyNames: string[], locale: SupportedLocale): string {
  if (keyNames.length === 0) {
    return 'None';  // keyInputNoneLabel
  }
  const joiner = '-';  // keyInputJoiner (ja: '-', en: '-')
  return keyNames.join(joiner);
}
```

### lib/generation/result-formatters.ts

```typescript
function resolveKeyInputDisplay(
  keyNames: KeyName[] | undefined,
  locale: 'ja' | 'en',
  preformatted?: string,  // 事前解決済み文字列があれば優先
): string {
  if (preformatted && preformatted.length > 0) {
    return preformatted;
  }
  if (!keyNames || keyNames.length === 0) {
    return '';
  }
  return formatKeyInputDisplay(keyNames, locale);
}
```

## データフロー

### 1. 検索実行時

```
[UI入力]
  ↓
SearchConditions.keyInput (keyMask形式)
  ↓
[Worker]
generateValidKeyCodes(keyInputMask) → keyCode[]
  ↓
[WASM検索]
WasmResult.keyCode
  ↓
[Worker変換]
keyCodeToNames(keyCode) → keyInputNames
  ↓
InitialSeedResult { keyCode, keyInputNames }
  ↓
SearchResult { keyCode, keyInput? }
```

### 2. UI表示時

```
SearchResult / GenerationResult
  ↓
keyInputNames または keyCode
  ↓
keyCodeToNames(keyCode) → KeyName[]  (keyInputNamesがなければ)
  ↓
resolveKeyInputDisplay(keyNames, locale) → string
  ↓
表示
```

### 3. エクスポート時

| Exporter | 処理 | fallback |
|----------|------|----------|
| `search-exporter` | `keyCode` → `keyCodeToNames` → `resolveKeyInputDisplay` | `mask: 0xXXXX` |
| `egg-search-exporter` | `boot.keyCode` → `keyCodeToNames` → `resolveKeyInputDisplay` | `'-'` |
| `generation-exporter` | `keyInputNames` → `resolveKeyInputDisplay` | `undefined` (空文字) |

## 問題点と改善案

### 1. 重複する型定義

`types/input.ts` の `KeyName` と `lib/utils/key-input.ts` の `KeyName` が別定義で存在。

**対策**: `types/input.ts` を削除し、`lib/utils/key-input.ts` の定義に統一する。

### 2. UI表示とExport表示の不統一

現状、KeyInput表示ロジックが複数箇所に分散：

| 場所 | 処理 |
|------|------|
| `ResultDetailsDialog.tsx` | `keyInputNames` → `keyCodeToNames` fallback → `resolveKeyInputDisplay` |
| `GenerationResultRow.tsx` | `resolveKeyInputDisplay(keyInputNames, locale, keyInputDisplay)` |
| `search-exporter.ts` | 独自 `formatKeyInput` → `mask: 0xXXXX` fallback |
| `egg-search-exporter.ts` | `keyCodeToNames` → `resolveKeyInputDisplay` → `'-'` fallback |
| `generation-exporter.ts` | `resolveKeyInputDisplay` → `undefined` fallback |

**問題**: 
- fallback値が統一されていない（`mask: 0xXXXX`, `'-'`, `'N/A'`, `keyInputUnavailableLabel`）
- 同じ変換ロジックが複数箇所に重複

**対策**: 共通関数 `formatKeyInputForDisplay` を作成し、UI/Export両方で使用する。

```typescript
// lib/utils/key-input.ts に追加
export function formatKeyInputForDisplay(
  keyCode: number | null | undefined,
  keyInputNames: KeyName[] | undefined,
  locale: SupportedLocale,
  fallback: string = ''
): string {
  // keyInputNamesがあればそれを使用
  if (keyInputNames && keyInputNames.length > 0) {
    return formatKeyInputDisplay(keyInputNames, locale);
  }
  // keyCodeがあれば変換
  if (keyCode != null) {
    const names = keyCodeToNames(keyCode);
    if (names.length > 0) {
      return formatKeyInputDisplay(names, locale);
    }
  }
  return fallback;
}
```

### 3. SearchResult型の冗長性

`SearchResult` が `keyInput` と `keyCode` の両方を持つが、実際には：
- `keyInput`: 検索条件のマスク値（検索時に使用、結果には不要）
- `keyCode`: 実際の結果のkeyCode（これだけ保持すればよい）

**現状のSearchResult**:
```typescript
export interface SearchResult {
  // ...
  keyInput?: number;      // ← 不要（検索条件であり結果ではない）
  keyCode?: number | null;
}
```

**対策**: `keyInput` を削除し、`keyCode` のみを保持。search-exporterの `mask: 0xXXXX` fallbackも廃止。

### 4. search-exporterの `mask: 0xXXXX` fallback

`keyCode` がnullで `keyInput`（マスク値）のみ存在する場合のレガシー対応。
`SearchResult.keyInput` 削除に伴い、このfallbackも不要になる。

**対策**: fallbackを統一（空文字または `'-'`）。

## 実装計画

### 実装完了

1. **Phase 1**: 共通関数 `formatKeyInputForDisplay` を `key-input.ts` に追加
   - 表示順序を A,B,X,Y,Start,Select,L,R,[↑],[↓],[←],[→] に統一
   - fallback を `-` に統一
2. **Phase 2**: UI表示を共通関数に統一
   - `ResultDetailsDialog.tsx`
   - `GenerationResultRow.tsx`
   - Boot Timing関連hook
3. **Phase 3**: Exporterを共通関数に統一
   - `search-exporter.ts`
   - `egg-search-exporter.ts`
   - `generation-exporter.ts`
4. **Phase 4**: `SearchResult.keyInput` 削除
5. **Phase 5**: `types/input.ts` 削除（未使用）
6. **不要関数削除**: `resolveKeyInputDisplay` 削除

## 関連ファイル一覧

| パス | 役割 |
|------|------|
| `src/lib/utils/key-input.ts` | 変換ユーティリティ（主要） |
| `src/types/input.ts` | 型定義（未使用・削除候補） |
| `src/types/search.ts` | Search関連型 |
| `src/types/generation.ts` | Generation関連型 |
| `src/types/egg.ts` | Egg Generation関連型 |
| `src/types/egg-boot-timing-search.ts` | Egg Boot Timing Search型 |
| `src/types/mt-seed-boot-timing-search.ts` | MT Seed Boot Timing Search型 |
| `src/types/pokemon-resolved.ts` | UI表示用解決済み型 |
| `src/lib/i18n/strings/search-results.ts` | 表示用フォーマット |
| `src/lib/generation/result-formatters.ts` | 結果フォーマッタ |
| `src/lib/export/search-exporter.ts` | Search Export |
| `src/lib/export/generation-exporter.ts` | Generation Export |
| `src/lib/export/egg-search-exporter.ts` | Egg Search Export |
| `src/workers/egg-boot-timing-worker.ts` | Egg Boot Timing Worker |
| `src/workers/mt-seed-boot-timing-worker.ts` | MT Seed Boot Timing Worker |
| `src/lib/search/search-worker-manager.ts` | Search Worker Manager |
