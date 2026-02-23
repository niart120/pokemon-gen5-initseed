# Generation Boot Timing Mode 仕様

## 1. 背景と目的
- これまで個体生成はユーザーが 64bit LCG Seed を直接入力する前提だった。
- 実機検証フローでは、起動タイミング・Timer0/VCount・キー入力の組み合わせから Seed を復元することが一般的であり、WebApp でも同じ体験を提供する必要がある。
- DeviceProfile に既に ROM/地域/ハード/MAC アドレス等が保存されているため、boot-timing モードではそれらをソースオブトゥルースとして参照し、Seed 派生を自動化する。

## 2. 要求サマリ
1. Generation パネルに **Seed入力モード** を追加し、`lcg` (従来) / `boot-timing` を切替可能にする。
2. boot-timing モードでは DeviceProfile から ROM Version/Region/Hardware/TID/SID/MAC を参照し、ユーザーは起動日時 + Timer0/VCount 範囲 + キー入力のみ指定する。
3. Timer0 と VCount は DeviceProfile の範囲設定をそのまま使用し、直積のペアごとに BaseSeed を導出してから個体生成を実行する。
4. ResultDetailsDialog の日時ラベルをクリックすると、Generation パネルを boot-timing モードへ切り替え、該当結果の日時/Timer0/VCount/KeyInput をプリセットする。
5. エクスポートメタデータへ boot-timing 情報を出力する。
6. GenerationResultsTable に Timer0/VCount 列を追加して、結果ごとの派生条件を参照できるようにする。

## 3. 入力形式
### 3.1 モード切替 UI
- `Seed Input` セクションにラジオボタン (または Segmented Control) を追加: `LCG Seed` / `起動タイミング`。
- モード値は Zustand (`draftParams.seedSourceMode`) として保持し、persist 対象に含める。

### 3.2 boot-timing モードの入力項目
| フィールド | 取得方法 / UI | 備考 |
| --- | --- | --- |
| 起動日時 | Date + Time picker (ローカルタイム) | 必須。秒単位まで入力。保存時は ISO8601 (UTC) へ正規化。 |
| キー入力 | SearchParamsCard と同じトグル UI を共通化 | 既定は `KEY_INPUT_DEFAULT`。ResultDetails 連携時は keyCode から初期化。 |
| 時刻系フラグ | (暗黙) DeviceProfile.hardware に応じて DS/DS Lite PM ビット等は `SeedCalculator.generateMessage` 内で従来通り計算される。 |

- Timer0/VCount を含む DeviceProfile 情報は、SearchParamsCard の「現在の範囲」表示と同じ小さめテキストスタイルでまとめて表示する。
  - 表示内容: `Version / Region / Hardware / MAC / Timer0 Range / VCount Range`
  - 例: `B (JPN) · DS · MAC 00:1B:2C:3D:4E:5F · Timer0 0x0C79-0x0C7B · VCount 0x60-0x60`

### 3.3 DeviceProfile 連動
- モードが boot-timing のとき、以下は profile からのみ取得し UI では編集不可 (表示のみ)。
  - ROM Version / Region
  - Hardware
  - Timer0 Range / VCount Range
  - MAC Address
  - TID / SID
  - shinyCharm / newGame / withSave / memoryLink (既存フィールド)
- Profile 切替時は boot-timing 入力を維持しつつ、自動反映されるよう `useAppStore.applyProfileToGeneration` を更新。

### 3.4 Timer0×VCount ペア列挙
- Timer0 候補: DeviceProfile.timer0Range から `timer0 = min .. max` を自動展開。
- VCount 候補: DeviceProfile.vcountRange から `vcount = min .. max` を自動展開。
- 派生対象ペアは直積 (timer0 昇順 × vcount 昇順)。

### 3.5 バリデーション
- boot-timing モードでは `baseSeedHex` の手入力を無効化 (read-only, grayed out)。
- フィールド未入力時は `validationErrors` に以下を追加。
  - 起動日時未指定
  - Timer0/VCount 範囲不正 (min>max, 0-65535/0-255 超え)
  - Timer0×VCount ペア数が 0 (範囲が空)
- LCG モードに戻した場合は従来のバリデーションのみ実行。

- 日時表示 (`formatResultDateTime`) を `<button>` 若しくは `role="link"` に変更し、クリック時に以下を実行。
  1. `setDraftParams({ seedSourceMode: 'boot-timing' })`。
  2. 起動日時 = `result.datetime`。
  3. キー入力 = `keyCode` が存在する場合は `keyCodeToMask` で復元、無い場合はデフォルト。
- トーストで「起動タイミングモードにコピーしました」を案内する (i18n 追加)。

## 4. 内部ロジック
### 4.1 状態構造
```ts
interface BootTimingDraft {
  enabled: boolean;          // seedSourceMode === 'boot-timing'
  timestampIso: string;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  keyMask: number;
}
```
- `GenerationParamsHex` には `seedSourceMode` と `bootTiming?: BootTimingDraft` を追加。
- Persist 時は `bootTiming` も保存する。

### 4.2 Seed 派生パイプライン
1. `commitParams()` 実行時、`seedSourceMode` を確認。
2. `lcg` の場合: 現状通り `baseSeedHex` から `GenerationParams` を構築。
3. `boot-timing` の場合: 以下の派生を実施。
   - `resolveProfile` から DeviceProfile を取得し、`SearchConditions` 相当の構造体を組み立てる。
   - Timer0/VCount 直積をループし、各ペアで `SeedCalculator.generateMessage` を呼び出し。
   - `SeedCalculator.calculateSeed` で `lcgSeed` を取得し、16進文字列へ変換。
   - 派生結果を `DerivedSeed[]` (timer0, vcount, lcgSeed, keyMask, datetime) として保持する。
4. Generation 実行時は `DerivedSeed[]` をキューに入れ、既存の Generation Worker へ 1 Seed ずつ順番に投入する。
   - 1 Seed 完了後に次の Seed を自動投入し、すべて完了したら `lastCompletion` をまとめる。
   - 進捗表示は「(現在インデックス/総Seed)」を `GenerationRunCard` のステータス行に追加する (UI 実装時に詳細化)。
5. `results` 配列には Seed ごとの出力を全て連結し、`seedHex` などで区別できるよう `UiReadyPokemonData.seedSourceSeedHex` を追加。

### 4.3 バリデーション / エラー処理
- Seed 派生中に `SeedCalculator` が例外を投げた場合は `validationErrors` へメッセージを格納し、Generation 開始を中止。

### 4.4 ResultDetailsDialog 連携
- 日時クリックでのプリセットは `useAppStore` を直接呼び出して実現。
- `lcgSeed` のコピー挙動は従来通り維持するが、boot-timing モードへの切替時は `baseSeedHex` の表示のみ更新 (編集不可)。

## 5. 出力形式
- **GenerationRunCard**
  - 進捗テキストに `Seed n/m` を追加 (n=現在処理中インデックス, m=派生Seed総数)。
- **ResultDetailsDialog**
  - 日時行にホバー時強調表示 + tooltip: `クリックで起動タイミングにコピー`。
- **GenerationResultsTable**
  - Timer0 列: `0x0C79` 形式で各結果の Timer0 を表示。
  - VCount 列: `0x95` 形式で各結果の VCount を表示。
  - boot-timing モード以外では列ヘッダーは残しつつ値は `--` 表示にする。

### 5.2 エクスポートメタデータ
`GenerationExportButton` へ追加する補助情報：
| フィールド | 形式 | 説明 |
| --- | --- | --- |
| seedSourceMode | `lcg` / `boot-timing` | 生成に使用したモード |
| bootTimestamp | ISO8601 | boot-timing モード時のみ |
| timer0 | 16進文字列 or 範囲 | Seed 毎に異なる場合は per-row / まとめて | 
| vcount | 16進文字列 or 範囲 | 同上 |
| keyInput | 文字列 (`L+Start` 等) | `formatKeyInputDisplay` を再利用 |
| macAddress | `AA:BB:CC:DD:EE:FF` | DeviceProfile 由来 |
| derivedSeedIndex | number | 複数 Seed の識別子 (0-based) |

CSV/TXT では列を追加、JSON では `meta.bootTiming` オブジェクトを付与する。

## 6. 影響範囲と後続タスク
1. Zustand (`generation-store.ts`, `app-store.ts`) のフィールド追加と persist 復元処理。
2. `GenerationParamsCard` UI 拡張 + 国際化文字列の追加 (`lib/i18n/strings/generation-params.ts`)。
3. `SeedCalculator` を UI から同期利用するためのヘルパー (`deriveBootTimingSeeds.ts` 仮称) 作成。
4. Generation Worker への連続投入制御 (マネージャ or slice 側で実装)。
5. ResultDetailsDialog / GenerationResultsControlCard / Export の UI 調整とテスト。
6. 回帰テスト: `npm run test`, `npx vitest run --config vitest.browser.config.ts ...`, 必要に応じて WebGPU/E2E。
