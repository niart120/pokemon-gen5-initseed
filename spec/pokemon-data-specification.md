# ポケモン生成機能 データ仕様書

Generation タブで使用するデータセットの構造と、実装での扱い方針をまとめる。データはビルド時に JSON としてバンドルされ、ブラウザ実行時は動的フェッチを行わない。

## 1. 種族カタログ（`src/data/species/generated`）

### 1.1 JSON 構造
`gen5-species.json` は全国図鑑 ID を文字列キーに持つ連想配列で、各値は以下の主要フィールドを含む。

- `nationalDex` (number): 全国図鑑番号。
- `names` ({ en: string; ja: string }): ローカライズ済み名称。
- `gender` (`GeneratedGenderSpec`): 性別仕様。`type` が `genderless`・`fixed`・`ratio` のいずれか。`ratio` の場合は `femaleThreshold` (0-255) を保持し、`gender_value < femaleThreshold` なら雌。
- `baseStats` ({ hp, attack, defense, specialAttack, specialDefense, speed }): 種族値。
- `abilities` (`GeneratedAbilities`): 通常特性 1/2 と隠れ特性。存在しない枠は `null`。
- `heldItems` (`Record<'black' | 'white' | 'black-2' | 'white-2', Item[]>`): バージョン別の持ち物候補。

固定性別の場合は `gender.fixed` が `'male'` もしくは `'female'` を保持し、性別不明は `type: 'genderless'` を利用する。

### 1.2 TypeScript アダプタ
`src/data/species/generated/index.ts` で以下を公開する。

- `getGeneratedSpeciesById(id: number): GeneratedSpecies | null`
- `selectAbilityBySlot(slot: number, abilities: GeneratedAbilities)`

Resolver (`enrichForSpecies`) はこれらの API を通じて性別閾値と特性名を `ResolutionContext` に取り込み、UI からはローカライズ済みの文字列が参照できるようになる。

## 2. 遭遇データセット（`src/data/encounters`）

### 2.1 ロケーション別レジストリ（`generated/v1`）
各 JSON は `EncounterLocationsJson` としてエクスポートされる。

```ts
interface EncounterLocationsJson {
  version: 'B' | 'W' | 'B2' | 'W2';
  method: DomainEncounterTypeName; // 例: 'Normal', 'Surfing'
  source: { name: string; url: string; retrievedAt: string };
  locations: Record<string, {
    displayNameKey: string;
    slots: EncounterSlotJson[];
  }>;
}

interface EncounterSlotJson {
  speciesId: number;
  rate: number; // % 表記
  levelRange: { min: number; max: number };
}
```

ビルド時に `import.meta.glob('./generated/v1/**/**/*.json', { eager: true })` で読み込み、`registry[`${version}_${method}`][normalizedLocation]` へ格納する。ロケーションキーは `applyLocationAlias` と `normalizeLocationKey` で正規化し、英語表記の場所でも同一キーにマージされる。`displayNameKey` は UI が翻訳キーとして利用し、未指定の場合は正規化済みキーで補完する。

### 2.2 固定遭遇・イベント（`static/v1`）
固定シンボルやイベント遭遇は `EncounterSpeciesJson` として管理する。

```ts
interface EncounterSpeciesJson {
  version: 'B' | 'W' | 'B2' | 'W2';
  method: DomainEncounterTypeName; // 例: 'StaticSymbol'
  source: { name: string; url: string; retrievedAt: string };
  entries: Array<{
    id: string;
    displayNameKey: string;
    speciesId: number;
    level: number;
    gender?: 'male' | 'female';
    isHiddenAbility?: boolean;
    isShinyLocked?: boolean;
  }>;
}
```

`listStaticEncounterEntries` がバージョンと遭遇種別ごとに絞り込んだ結果を返し、UI で選択された要素を `buildResolutionContext` の `staticEncounter` 引数へ渡す。

### 2.3 スキーマ補足
`src/data/encounters/schema.ts` に TypeScript interface を定義しており、生成スクリプトはこの型に沿って JSON を出力する。`source` 情報は出典と取得日を記録するメタデータとして保持し、データの検証や再取得の判断材料に利用する。

## 3. ランタイムで構築される派生構造

### 3.1 EncounterTable（`src/data/encounter-tables.ts`）
`getEncounterTable(version, location, method)` は前述のレジストリから `EncounterTable` を作成し、Resolver に提供する。

```ts
interface EncounterSlot {
  speciesId: number;
  rate: number;
  levelRange: { min: number; max: number };
}

interface EncounterTable {
  location: string;
  method: DomainEncounterType;
  version: ROMVersion;
  slots: EncounterSlot[];
}
```

WASM が返す `encounter_slot_value` を添字としてスロットを選択する。範囲外の値が渡された場合は例外を投げ、テストで検知できるようにしている。

### 3.2 ResolutionContext（`src/lib/initialization/build-resolution-context.ts`）
`buildResolutionContext` は以下の情報をひとまとめにして返す。

- `encounterTable`: 通常遭遇の場合は `EncounterTable`、固定遭遇の場合は単一スロットの疑似テーブル。
- `genderRatios`: `Map<number, GenderRatio>`。初回アクセス時に遅延構築。
- `abilityCatalog`: `Map<number, string[]>`。こちらも遅延構築。

`enrichForSpecies(ctx, speciesId)` を呼ぶと種族カタログから性別閾値と特性名を取得して各 Map に格納し、UI 側では `toUiReadyPokemon` でローカライズ済みの表示データへ変換する。

### 3.3 GenderRatio 型
`src/types/pokemon-raw.ts` で以下を定義する。

```ts
interface GenderRatio {
  threshold: number;   // 0-256, gender_value < threshold なら雌
  genderless: boolean;
}
```

固定雌は `threshold = 256`、固定雄は `threshold = 0` を割り当て、性別不明は `genderless = true` で表現する。

## 4. データ更新フロー

1. `scripts/fetch-gen5-species.js` で種族データを取得し、`gen5-species.json` を再生成する。
2. `scripts/scrape-encounters.js` で遭遇テーブルを取得する。
3. `scripts/migrate-encounter-display-names.js` などでロケーション名を整形し、`generated`/`static` ディレクトリへ配置する。
4. `npm run test` と `npm run lint` を実行し、Resolver/UI の整合性を確認する。
5. 必要に応じて `legacy-docs/` へ旧仕様を移し、`spec/README.md` のステータスを更新する。

以上により、Generation 機能はビルド済み JSON を参照しつつ WASM の結果を UI 表示に適した形へ解決する。
