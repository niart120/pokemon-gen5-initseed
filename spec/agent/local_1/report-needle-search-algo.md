# レポ針検索 内部計算ロジック設計

## 目的とスコープ
- UIで入力された針列と範囲条件から、消費位置・Timer0・VCountを列挙する計算ロジックを定義する。
- モード別: `LCG Seed`（既知シードから位置特定）、`Boot-Timing`（起動条件からシード算出 → 位置特定）。
- 既存のRust/wasmコアは未提供のため、初版は TypeScript 実装で完結させる。負荷が顕著になった場合のみ wasm 移植を検討する。

## 入出力（論理モデル）
- 入力
  - 共通: 針列（0-7、長さ1-128）、`advanceRange {start,end}`、`timer0Range`、`vcountRange`。
  - LCG Seed: `seedHex`（64bit LCG 状態）。
  - Boot-Timing: `timestampIso`（ローカル→ISO変換済み）、`keyMask`（UI保存値）、`DeviceProfile` 由来の `romVersion`/`romRegion`/`hardware`/`macAddress`。
    - `keyMask` は「押下されているキーの確定セット」を表し、実行時に `keyMaskToKeyCode(keyMask)` で単一の `keyCode` を生成して使用する。
- 出力
  - 行ごとに `consumptionIndex`（advance）、`timer0`、`vcount`。Boot-Timingでは必要なら `keyCode`/`keyNames` を付加可能（UI列は据え置き）。

## 依存・再利用箇所
- LCG演算: Rust `PersonalityRNG::calc_report_needle_direction` と同じ式を TS で使用（`next = seed * 0x5D588B656C078965n + 0x269EC3n; dir = ((next >> 32n) * 8n) >> 32n`）。
- Boot-Timing → LCG Seed: `SeedCalculator.generateMessage` + `SeedCalculator.calculateSeed` を流用し、`keyMaskToKeyCode` / `generateValidKeyCodes` でキー組合せを展開。
- プロファイル: `DeviceProfile` から ROM/HW/MAC, Timer0/VCount 範囲を取得。UIで保持していない情報（ROM/HW/MAC）はプロファイル必須。

## アルゴリズム方針
### 1) 針列前処理
- サニタイズ済み針列を `Uint8Array needles` にする。
- 長さ `m`、探索範囲長 `N = end - start + 1`。

### 2) 針一致判定（単一初期シード）
- 入力: `baseSeed` (u64, BigInt), `rangeStart`, `rangeEnd`, `needles`。
- 手順
  1. `seed = baseSeed` から `rangeStart` だけLCGジャンプ（反復で十分: 最大2000程度）。
  2. 位置 `i` を `rangeStart..=rangeEnd` で走査。
  3. 位置ごとに針列長 `m` 回だけ `next_seed` と `calc_dir` を呼び、全一致なら `consumptionIndex = i + m - 1` を結果に追加。
- 計算量: O((N) * m)、現行UIデフォルトで最大 ~256k ステップ程度。
- 最適化オプション（必要時）
  - ジャンプ計算 `lcg_affine_for_steps` を TS で移植し、`rangeStart` まで一括ジャンプ。
  - 針列が長い場合、先頭不一致で早期break。

### 3) LCG Seed モード
- 単一シードで上記手順を実行。
- 入力シードは 16進文字列を `BigInt("0x...")` で解釈し、64bitに正規化（`& ((1n<<64n)-1n)`）。

### 4) Boot-Timing モード
- 事前条件: プロファイルから `romVersion/romRegion/hardware/macAddress` を取得。これらが無い場合は計算不能。
- 展開手順
   1. `keyMask` をそのまま `keyMaskToKeyCode` で `keyCode` に変換（UIで選択した押下セットをそのまま使う）。
   2. `timer0` を範囲でループし、`vcount` はプロファイル範囲（1値が多い）。
   3. 各 (timer0, vcount) と単一 `keyCode` で `SeedCalculator.generateMessage` → `calculateSeed` を実行し、得た `lcgSeed` を 2) に渡す。
   4. ヒット時に `timer0/vcount` を結果へ。
- 複雑度: `O(T * V * K * N * m)`。典型: timer0~2値, vcount~1値, key codes <= 64 → 問題なし。

## 実装配置案
- 新Worker: `src/workers/report-needle-search-worker.ts`（TSのみ）。理由: UIブロッキング回避と他検索Workerとの一貫性。
- エンジン: `src/lib/report-needle-search/` に純関数モジュールを作成。
  - `lcg.ts`: 乗算・加算定数、`nextSeed`, `jumpSeed`, `calcNeedleDir`。
  - `needle-matcher.ts`: 単一シードに対する一致探索ロジック。
  - `boot-timing-expander.ts`: キー展開と SeedCalculator 連携でシード生成。
- 型: 既存 `ReportNeedleSearchDraft` と互換の入力 DTO を `types/report-needle-search.ts` に追加（Worker向け）。

## メッセージフロー案（Worker）
- `START`: ペイロードにドラフトとプロファイル要約を渡す → Workerで展開・検索 → `RESULTS` (逐次) と `COMPLETE`。
- `PAUSE/RESUME/STOP`: CPU Workerと同じ制御。検索規模が小さい場合は実装を簡略化しても良い。
- `ERROR`: バリデーション/実行時エラーを通知。

## 性能・メモリ評価
- 針列長128、範囲2000、コンボ (T=2, V=1, K=64) の最悪計算量でも < 256k * 128 ≈ 32M ステップ相当で、JS単体でも数百ms〜数秒程度。進捗不要だが長大範囲（>50k）を許容する場合は分割実行＋進捗報告を追加。
- メモリ: 針列と小さな結果配列のみ。結果件数が多い場合でも数百件規模を想定。

## ガード・上限ポリシー
- 消費探索範囲: UIは 0–100000 を上限想定。`end - start + 1` が上限を超える場合は警告して即時拒否（分割実行は行わない）。
- 結果件数: 上限は設けない（メモリ逼迫が判明した場合のみ将来検討）。

## wasm併用の検討
- 現状の wasm にはレポ針検索専用エントリが無い。移植するとしても Rust 側で `calc_report_needle_direction` をバルク計算する程度の利点で、現行範囲では JS で十分。
- 将来の高速化案: Rustで `match_needles(seed, range_start, range_end, needles)` を実装し、JSから呼び出す。WebGPU活用は過剰。

## 未決事項 / ToDo
- 検索結果に `initialSeed` と `currentSeed` の列を追加する（`keyNames` は含めない方針）。
- 実行は Worker 経由とする（UIスレッド実行の簡易版は提供しない）。
- バリデーションとエラーメッセージ: 針列長上限・範囲上限・日時必須などを UI/Worker で同一文言・同一ロジックで扱う。
