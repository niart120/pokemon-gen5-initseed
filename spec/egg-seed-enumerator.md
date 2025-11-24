# 卵Seed列挙器 仕様書

## 1. 概要

### 1.1 目的
育て屋内部で発生する NPC 消費と卵生成ロジックを Rust/WASM 側の列挙器として再現し、UI から一定数の卵候補をストリーミング取得できるようにする。本仕様書では `SeedEnumerator` と同等の API 体験を維持しつつ、卵特有の IV 解決・フィルタリングに必要な処理を定義する。

### 1.2 適用範囲
- 対象ゲーム: ポケットモンスター ブラック / ホワイト / ブラック2 / ホワイト2
- 対象機能: 卵生成計算、NPC 消費安定性判定、個体値フィルタリング
- 非対象: 実機シンクロ、孵化歩数、孵化後の技遺伝

### 1.3 用語
| 用語 | 説明 |
|------|------|
| `PersonalityRNG` | BW/BW2 で利用される 64bit LCG | 
| `MT` | Mersenne Twister 19937。個体値計算用に導入する 32bit 疑似乱数 | 
| `ParentsIVs` | ⽗⺟ポケモンの既知個体値セット | 
| `IVResolutionConditions` | `resolve_egg_iv` が参照する、親 IV / RNG IV のセット | 
| `EnumeratedEggData` | 列挙結果 1 件を表す構造体 | 

## 2. 入力インターフェース

### 2.1 ParentsIVs
初期化時に必須。Rust 側では以下の構造体として受け取る。
```rust
pub struct ParentsIVs {
    pub male: IvSet,
    pub female: IvSet,
}
```
両親の IV は `IvValue (0-31 または 32)` を HP→Speed の順で保持する。`IvValue::Unknown (32)` も許容し、`resolve_egg_iv` でそのまま伝播させる。

### 2.2 列挙パラメータ
| パラメータ | 型 | 説明 |
|------------|----|------|
| `base_seed` | `u64` | LCG 初期 Seed |
| `user_offset` | `u64` | UI から指定される開始 advance。`SeedEnumerator` と同様に加算オフセットを取る |
| `count` | `u32` | 列挙上限件数 |
| `conditions` | `GenerationConditions` | `egg_iv.rs` で定義済みの生成コンテキスト |
| `parents_ivs` | `ParentsIVs` | 2.1 参照 |
| `filter` (任意) | `IndividualFilter` | 結果を絞り込む条件。未指定時は全件返却 |
| `consider_npc_consumption` | `bool` | NPC 消費を計算に含めるかどうか。`true` で `resolve_npc_advance` を適用、`false` で NPC 消費をスキップ（`is_stable=false`) |

## 3. RNG 準備と IV 解決に必要なセットアップ

### 3.1 LCG Seed から MT Seed への変換
1. ゲーム内オフセット計算で消費される前の **BaseSeed** を一時保持する。以降の工程で LCG を進める前に、MT 用の Seed をここから導出する必要がある。
2. `BaseSeed` はクローン（コピー）して MT 専用の `PersonalityRNG` に渡す。コピーした RNG で 1 回だけ `next()` を呼び、その戻り値を `mt_seed` として使用する。本来の LCG シーケンスはコピー前の状態を保ち、以降の `calculate_game_offset` 等で通常通り進行できる。
3. シンプルなヘルパー `derive_mt_seed(base_seed: u64) -> u32` を定義し、内部で上記の「クローンした BaseSeed から 1 回だけ `next()` を呼び、得られた 32bit 値を返す」処理を行う。必要に応じて XOR などの整形を追加できるが、基本は `BaseSeed.next()` の出力をそのまま MT19937 の `seed(u32)` に渡す。

### 3.2 MT による RNG IV 生成
1. `derive_mt_seed` の戻り値で MT (仮実装でも可) を初期化する。
2. 最初の 7 回は `engine.next()` を呼び出し、得られた値は全て破棄する。これはゲーム内部の未確認消費を再現するためのダミー処理。
3. 続く 6 回の `engine.next()` 出力から上位 5bit を取り出し (`value >> 27`)、HP→Speed の順で `rng_iv_set` を構築する。
4. 生成した `rng_iv_set` と `ParentsIVs` を束ねて `IVResolutionConditions` を用意する。
   ```rust
   let iv_sources = IVResolutionConditions {
       male: parents_ivs.male,
       female: parents_ivs.female,
       rng: rng_iv_set,
   };
   ```
5. `iv_sources`（およびその元となる `mt_seed`）は列挙器初期化時に 1 度だけ構築し、その後は全ての卵で使い回す。LCG 乱数列と MT 乱数列はゲーム内部でも独立しており、片方の消費がもう一方に影響しないため、この最適化は実ゲーム挙動と一致する。

## 4. 列挙シーケンス
1. **Advance 位置の算出**: `SeedEnumerator` と同じく `calculate_game_offset` の結果を `user_offset` に足し合わせ、`PersonalityRNG::lcg_affine_for_steps` で初期 Seed を進める。`next_advance` は `user_offset` で初期化し、**卵生成を試みるたび**（フィルタで棄却された場合も含む）に 1 だけ増やす。内部 RNG 消費量や NPC 消費は advance に反映しない。`count` は `target_count` として保持する。
2. **NPC 進行の解決（任意）**: `consider_npc_consumption` が `true` の場合のみ、各卵生成の直前に `resolve_npc_advance(current_seed, 96, 30)` を呼び出す。`frame_threshold` と `slack` は列挙器内部で固定されており、`GenerationConditions` には含めない。`false` の場合は NPC 消費をスキップし、`is_stable` は常に `false` とする。
   - 戻り値 `(next_seed, consumed_frames, is_stable)` の意味:
     - `next_seed`: NPC 消費後の LCG Seed。以降の計算の起点。
     - `consumed_frames`: NPC が消費した LCG ステップ数。
     - `is_stable`: 指定したフレーム閾値内で安定した場合 `true`。
        - `consumed_frames` は内部でのみ使用し、現状 UI へ露出しない。`is_stable` は結果構造体へそのまま転記する（NPC スキップ時は `false`）。
3. **PendingEgg の生成**: `derive_pending_egg(next_seed, &conditions)` を呼び、性格・遺伝スロット・PID 等を含む `PendingEgg` を得る。戻りの Seed は `PersonalityRNG::current_seed()` で取得する。
4. **IV 解決**:
    - 初期化時に計算した `iv_sources` をそのまま利用する。
    - `resolve_egg_iv(&pending, &iv_sources)` を実行し `ResolvedEgg` を得る。
5. **フィルタ適用 (任意)**:
    - `filter` が設定されている場合のみ `matches_filter(&resolved, filter)` を実行。
    - `false` の結果は UI に返さずスキップするが、試行分の advance はすでに加算済みとして扱う。
6. **結果生成**:
    - `EnumeratedEggData { advance, egg: resolved, is_stable }` を構築（`advance` は試行開始時点の `next_advance` を使用）。
    - `current_seed` を `PersonalityRNG::current_seed()` に更新し、`produced` を 1 増やして次ループへ。
    - フィルタが厳しすぎる場合、`target_count` に達しても 1 件も返さず終了する可能性がある。呼び出し側は 0 件完了をケースとして想定すること。

## 5. データ構造定義

### 5.1 IV 補助構造体
```rust
pub struct ParentsIVs {
    pub male: IvSet,
    pub female: IvSet,
}
```
既知の親 IV を受け取り、列挙器内部でコピーして保持する。

### 5.2 列挙結果
```rust
pub struct EnumeratedEggData {
    pub advance: u64,
    pub egg: ResolvedEgg,
    pub is_stable: bool,
}
```
- `advance`: `user_offset` からの試行インデックス。フィルタで棄却された試行も Advance を占有する。
- `egg`: `resolve_egg_iv` の戻り値。`wasm_bindgen` でゲッターを公開する。
- `is_stable`: `resolve_npc_advance` の判定値。

### 5.3 列挙器クラス
```rust
#[wasm_bindgen]
pub struct EggSeedEnumerator {
    current_seed: u64,
    next_advance: u64,
    target_count: u32,
    produced: u32,
    conditions: GenerationConditions,
    parents: ParentsIVs,
    iv_sources: IVResolutionConditions,
    consider_npc_consumption: bool,
    filter: Option<IndividualFilter>,
}
```
公開メソッド:
- `new(base_seed, user_offset, count, conditions, parents, filter)`
- `next_egg() -> Option<EnumeratedEggData>`

## 6. 疑似コード
```rust
pub fn next_egg(&mut self) -> Option<EnumeratedEggData> {
    if self.produced >= self.target_count {
        return None;
    }

    loop {
        let current_advance = self.next_advance;
        self.next_advance = self.next_advance.saturating_add(1);

        let (seed_after_npc, is_stable) = if self.consider_npc_consumption {
            let (next_seed, _frames, stable) = resolve_npc_advance(self.current_seed, 96, 30);
            (next_seed, stable)
        } else {
            (self.current_seed, false)
        };

        let mut rng = PersonalityRNG::new(seed_after_npc);
        let pending = derive_pending_egg(seed_after_npc, &self.conditions);
        let resolved = resolve_egg_iv(&pending, &self.iv_sources).expect("invalid iv input");
        self.current_seed = PersonalityRNG::current_seed_from(rng);
        self.produced += 1;

        if self.filter.as_ref().map_or(true, |f| matches_filter(&resolved, f)) {
            return Some(EnumeratedEggData::new(current_advance, resolved, is_stable));
        }

        if self.produced >= self.target_count {
            return None;
        }
    }
}
```
※ `pending_consumed()` は `derive_pending_egg` 内で使った LCG 消費数を返す内部ヘルパーとして想定。

## 7. テストおよびモック要件
- `MT` は現状未実装のため、Rust 側では固定の疑似 MT 実装または `random-js` と同様のロジックをモックする(`mt19937.rs`を作成する)。最終的には `wasm-pkg` 内でネイティブ実装を行う。
- `derive_mt_seed` と `generate_rng_iv_set` の導出結果を単体テスト化し、特定 Seed に対する 6 つの IV が再現できることを保証する（列挙器ではこのセットを初期化時に固定保持する）。
- `EggSeedEnumerator` のテストでは以下を最低限確認する。
    1. `ParentsIVs` の Unknown 値が `resolve_egg_iv` で許可される。
    2. `consider_npc_consumption=true` の場合のみ `resolve_npc_advance` が呼ばれ、`is_stable` が `EnumeratedEggData` に反映される。`false` の場合は `is_stable=false` となる。
    3. `target_count` 上限に達するとフィルタ結果に関わらず `next_egg` が `None` を返し、`advance` は試行回数に合わせて 1 ずつ増えていく（フィルタで 0 件完了となるケースも含む）。

## 8. 実装メモ
- `resolve_npc_advance` には固定値 `frame_threshold=96`、`slack=30` を渡す。UI/条件からの入力は不要。
- `EnumeratedEggData` の `advance` は UI 側の並び替え指標となるため、64bit 整数のままエクスポートする。
- 列挙中に `resolve_egg_iv` がエラーを返した場合は即座に panic するのではなく `Result` を返して UI でハンドリングできる余地を残すことを推奨。
