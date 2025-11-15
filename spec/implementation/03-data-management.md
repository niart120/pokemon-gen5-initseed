# ポケモン生成機能 データ管理実装

Generation 機能で利用するデータセットを TypeScript 側でどのように読み込み、加工し、Resolver へ渡しているかをまとめる。すべてのデータはビルド時にバンドル可能な JSON として管理し、ランタイムのネットワーク I/O は発生しない。

## 構成概要
- **種族カタログ**: `src/data/species/generated/gen5-species.json`
  - 全国図鑑 ID をキーに能力値・特性・性別比などを収録。
  - `src/data/species/generated/index.ts` が型安全なアクセサを提供。
- **エンカウントテーブル（ロケーション別）**: `src/data/encounters/generated/v1/**/*.json`
  - バージョン × エンカウント種別ごとに場所とスロット構成を定義。
  - `src/data/encounters/loader.ts` がレジストリを構築し、`src/data/encounter-tables.ts` から利用する。
- **固定シンボル / イベントエンカウント**: `src/data/encounters/static/v1/**/*.json`
  - 固定シンボル・配布イベントなどをエントリ単位で管理。
- **解決コンテキスト**: `src/lib/initialization/build-resolution-context.ts`
  - 上記データを束ね、Resolver へ渡す `ResolutionContext` を生成する。

## 1. 種族データ（`src/data/species/generated`）

### JSON レイアウト
`gen5-species.json` は文字列化した全国図鑑番号をキーにした連想配列で、主なフィールドは下記の通り。

- `nationalDex`: 全国図鑑番号。
- `names`: ローカライズ済み名称。`{ en: string; ja: string }` を持つ。
- `gender`: `GeneratedGenderSpec`。`type` に応じて `femaleThreshold`（0-255）または `fixed` を保持する。
- `baseStats`: 種族値。`hp` など 6 ステータスを含む。
- `abilities`: 通常特性 1/2 と隠れ特性。存在しない枠は `null`。
- `heldItems`: バージョン別の持ち物候補。現状 Resolver では未使用だが構造を維持する。

### TypeScript アダプタ
`src/data/species/generated/index.ts` で公開する主要 API。

- `getGeneratedSpeciesById(id: number): GeneratedSpecies | null`
- `selectAbilityBySlot(slot: number, abilities: GeneratedAbilities)`

`GeneratedSpecies` は JSON 構造に対応する型を表し、Resolver の `enrichForSpecies` が性別閾値と特性名を `ResolutionContext` に取り込む際に利用する。

## 2. エンカウントレジストリ（`src/data/encounters/loader.ts`）

### ロケーション別エンカウントテーブル
エンカウントテーブルは `EncounterLocationsJson` として構造化される。

```ts
interface EncounterLocationsJson {
  version: 'B' | 'W' | 'B2' | 'W2';
  method: DomainEncounterTypeName;
  source: { name: string; url: string; retrievedAt: string };
  locations: Record<string, {
    displayNameKey: string;
    slots: EncounterSlotJson[];
  }>;
}

interface EncounterSlotJson {
  speciesId: number;
  rate: number;
  levelRange: { min: number; max: number };
}
```

Vite の `import.meta.glob('./generated/v1/**/**/*.json', { eager: true })` で一括読み込みし、`registry[`${version}_${method}`]` にマージする。ロケーションキーは `applyLocationAlias` と `normalizeLocationKey` で正規化し、表記揺れを吸収する。`displayNameKey` は UI の翻訳キーとして利用し、未指定の場合は正規化後のキーで補完する。

### 固定エンカウントカタログ
固定シンボルやイベントエンカウントは `EncounterSpeciesJson` で表現する。

```ts
interface EncounterSpeciesJson {
  version: 'B' | 'W' | 'B2' | 'W2';
  method: DomainEncounterTypeName;
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

`listStaticEncounterEntries` がバージョンとエンカウント種別で絞り、Resolver には `staticEncounter` として渡す。`displayNameKey` が欠けている場合は ID で補完する。

## 3. エンカウントテーブル API（`src/data/encounter-tables.ts`）

`loader.ts` が構築したレジストリを薄いユーティリティで公開する。

```ts
export interface EncounterSlot {
  speciesId: number;
  rate: number;
  levelRange: { min: number; max: number };
}

export interface EncounterTable {
  location: string;
  method: DomainEncounterType;
  version: ROMVersion;
  slots: EncounterSlot[];
}
```

- `getEncounterTable(version, location, method)`: 該当するロケーションが無ければ `null` を返す。
- `getEncounterSlot(table, slotValue)`: WASM から渡される `encounter_slot_value` を添字として扱う。範囲外の場合は例外を投げる。

## 4. 解決コンテキスト（`src/lib/initialization/build-resolution-context.ts`）

`buildResolutionContext` は画面状態（ゲームバージョン、エンカウント種別、ロケーション/固定エンカウント情報）を入力に `ResolutionContext` を構築する。

- 通常エンカウント: `getEncounterTable` でテーブルを取得し、結果をコンテキストへ格納する。
- 固定エンカウント: 単一スロットの疑似テーブルを生成し、固定レベルで Resolver に渡す。
- 生成結果は `(version, encounterType, location/static)` をキーにしたメモリキャッシュへ保存する。

追加で `enrichForSpecies(ctx, speciesId)` を呼び出すと、種族データから性別閾値と特性名を遅延的に取り込み、`pokemon-resolver.ts` が性別および特性を決定できる状態を作る。

## 5. データ更新フロー

1. **ソース取得**: `scripts/fetch-gen5-species.js` で種族データ、`scripts/scrape-encounters.js` でエンカウントテーブルを収集する。
2. **整形/マイグレーション**: `scripts/migrate-encounter-display-names.js` 等でキー整合性を保つ。
3. **配置**: 生成された JSON を `src/data/species/generated/` および `src/data/encounters/(generated|static)/` に配置する。
4. **検証**: `npm run test`（Resolver 周辺の単体テスト）と `npm run test:rust`（WASM 側）で整合性を確認する。

このサイクルにより、フロントエンドはビルド時に必要なデータをすべて取り込み、ブラウザ実行時は静的 import のみで完結する。
