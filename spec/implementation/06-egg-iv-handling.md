# 06 - タマゴ個体値の取扱い仕様

## 1. 背景
孵化乱数の探索では、親ポケモンの個体値を完全には把握できないケースがある。従来の 0-31 固定幅では「遺伝箇所が不明な値を取り得る」状況を表現できず、フィルタリングや可視化の際に矛盾が生じる。本仕様では、親側・タマゴ側ともに `32` を「Unknown (未確定)」として扱うルールを定義し、探索エンジンおよび UI で一貫した挙動を保証する。

## 2. 用語とデータ型

### 2.1 IvValue
| 名称 | 型 | 説明 |
| --- | --- | --- |
| `IvValue` | `u8 (0..=32)` | 0-31: 実数値。32: Unknown (未確定) |

- `IvValue::Unknown` は実装上 `32u8`。WASM/TS 間で `repr(u8)` を共有。
- Unknown は「どの値でも取り得る」ことを意味し、演算対象外として扱う。

### 2.2 IV 配列
| 名称 | 型 | 説明 |
| --- | --- | --- |
| `IvSet` | `[IvValue; 6]` | HP, Atk, Def, SpA, SpD, Spe の順で格納 |
| `IvRange` | `{ min: IvValue, max: IvValue }` | inclusive。デフォルトは `{0, 32}` で\"制約なし\"を表現 |

### 2.3 その他
- `HiddenPowerInfo` を `enum { Known { r#type: HiddenPowerType, power: u8 }, Unknown }` に拡張。
- `IndividualFilter.hidden_power_type` / `hidden_power_power` は `HiddenPowerInfo::Known` のみと照合。

### 2.4 PID/色違い関連データ型（Rust 実装準拠）
| 名称 | 型/出典 | 説明 |
| --- | --- | --- |
| `PIDCalculator` | `wasm-pkg/src/pid_shiny_checker.rs` | `generate_egg_pid(r1, r2)` など PID 組立てを担当する既存ユーティリティ |
| `ShinyChecker` | 同上 | `get_shiny_value(tid, sid, pid)` と `get_shiny_type` を提供。`Square`/`Star`/`Normal` 判定を集約 |
| `ShinyType` | `enum { Normal=0, Square=1, Star=2 }` | 既存 Rust 実装に合わせた色違い区分。`Square` は shiny value=0、`Star` は 1..=7、`Normal` はそれ以外 |

### 2.5 生成関連データ型
| 名称 | 型 | 説明 |
| --- | --- | --- |
| `EverstonePlan` | `enum { None, Fixed(Nature) }` | かわらずのいし有無と対象性格を表現。FFI 用に `repr(u8)` を明示 |
| `GenerationConditions` | `struct { has_nidoran_flag: bool, everstone: EverstonePlan, uses_ditto: bool, allow_hidden_ability: bool, reroll_count: u8 (0..=5), trainer_ids: TrainerIds, gender_ratio: GenderRatio }` | `derive_pending_egg` の入力。`TrainerIds` は `tid: u16`, `sid: u16`, `tsv: u16 (tid ^ sid)` を保持 |
| `InheritanceSlot` | `struct { stat: StatIndex, parent: ParentRole }` | 3 スロット分を `PendingEgg.inherits` に格納 |
| `PendingEgg` | `struct { inherits: [InheritanceSlot;3], nature: Nature, gender: Gender, ability: AbilitySlot, shiny: ShinyType, pid: u32 }` | 個体値未確定のタマゴデータ |
| `IVResolutionConditions` | `struct { male: IvSet, female: IvSet, rng: IvSet }` | `resolve_egg_iv` の入力。`rng` は 0-31 のみ |
| `ResolvedEgg` | `struct { ivs: IvSet, nature: Nature, gender: Gender, ability: AbilitySlot, shiny: ShinyType, pid: u32, hidden_power: HiddenPowerInfo }` | 個体値決定後のタマゴデータ。めざめるパワーのタイプ/威力をキャッシュ |
| `StatRange` | `struct { min: IvValue, max: IvValue }` | inclusive 範囲。`{0,32}` で無条件 |
| `IndividualFilter` | `struct { iv_ranges: [StatRange;6], nature: Option<Nature>, gender: Option<Gender>, ability: Option<AbilitySlot>, shiny: Option<ShinyType>, hidden_power_type: Option<HiddenPowerType>, hidden_power_power: Option<u8> }` | フロントエンド検索条件 |

`ParentRole` は `enum { Male, Female }` を基本とする。`GenderRatio`/`AbilitySlot` など既存 enum は `pokemon-data-specification.md` の定義を参照。
## 3. 関数仕様と処理フロー

### 3.1 `derive_pending_egg`
- **署名**: `fn derive_pending_egg(seed: u64, conditions: &GenerationConditions) -> PendingEgg`
- **RNG**: 既存の `PersonalityRNG` を使用し、BW/BW2 仕様の 64bit LCG (`seed = seed * 0x5D588B656C078965 + 0x269EC3`) を 1 ステップごとに進める。上位 32bit を `next_u32()` として取得する。

#### 3.1.1 一体的な生成フロー
非 PID セクションと PID リロールを分離せず、実機同様の処理順でまとめる。

```
let mut rng = PersonalityRNG::new(seed);
let TrainerIds { tid, sid, tsv: _ } = conditions.trainer_ids;

// 1) 性格・かわらずのいし
let nature_roll = rng.next_u32();
let nature = match conditions.everstone {
  EverstonePlan::None => Nature::from_roll(nature_roll),
  EverstonePlan::Fixed(parent_nature) => {
    let inherit = (rng.next_u32() >> 31) == 0;
    if inherit { parent_nature } else { Nature::from_roll(nature_roll) }
  }
};

// 2) 夢特性ロール (値は後で使用)
let ha_roll = rng.next_u32();
if conditions.uses_ditto {
  rng.next_u32(); // メタモン参加時の消費
}

// 3) 遺伝箇所 (パワー系はスコープ外)
let mut inherits: Vec<InheritanceSlot> = Vec::new();
while inherits.len() < 3 {
  let stat = (rng.next_u32() * 6) >> 32;
  let parent_bit = rng.next_u32() >> 31;
  if inherits.iter().any(|slot| slot.stat == stat) { continue; }
  inherits.push(InheritanceSlot::new(stat, parent_bit));
}

// 4) ニドラン/バルビート専用性別ロール
let nidoran_roll = if conditions.has_nidoran_flag {
  Some(((rng.next_u32() as u64).wrapping_mul(2) >> 32) as u8)
} else {
  None
};

// 5) PID + TSV リロール
let mut chosen_pid = None;
for attempt in 0..=conditions.reroll_count {
  let pid = rng.next_u32();
  let shiny_value = ShinyChecker::get_shiny_value(tid, sid, pid);
  let shiny_type = ShinyChecker::get_shiny_type(shiny_value);
  if shiny_type != ShinyType::Normal || attempt == conditions.reroll_count {
    chosen_pid = Some((pid, shiny_type));
    break;
  }
}
let (pid, shiny_type) = chosen_pid.expect("PID must exist");

// 6) 性別判定
let gender = match nidoran_roll {
  Some(0) => Gender::Female,
  Some(_) => Gender::Male,
  None => conditions.gender_ratio.resolve(pid & 0xFF),
};

// 7) 特性スロット
let mut ability = AbilitySlot::from_bit(((pid >> 16) & 1) as u8);
let ha_candidate = conditions.allow_hidden_ability
  && !conditions.uses_ditto
  && female_parent_has_hidden
  && ((ha_roll * 5) >> 32) >= 2;
if ha_candidate {
  ability = AbilitySlot::Hidden;
}

PendingEgg {
  inherits: [inherits[0], inherits[1], inherits[2]],
  nature,
  gender,
  ability,
  shiny: shiny_type,
  pid,
}
```

- 乱数消費は上記の通り一本化し、PID リロールも同じ `PersonalityRNG` で継続する。
- `reroll_count = 0` でもループは 1 度実行され、最初の PID が採用される。
- パワー系アイテムや複数親のかわらずのいしは今回の仕様スコープ外。
- `TrainerIds.tsv` は互換性確保のため保持するが、実装では `tid`/`sid` を直接 `ShinyChecker` に渡す。
- `female_parent_has_hidden` はペア情報から導かれる真偽値であり、メタモンが絡む場合は常に `false` とする。

### 3.2 `resolve_egg_iv`
- **署名**: `fn resolve_egg_iv(pending: &PendingEgg, iv_sources: &IVResolutionConditions) -> ResolvedEgg`
- **前提**: `iv_sources.male/female` は 0-32 を含む `IvSet`、`rng` は 0-31 限定。値域外は `InvalidIvValue` を返す。
- **処理**:
```
let mut resolved = iv_sources.rng;
for slot in pending.inherits {
  let idx = slot.stat as usize;
  resolved[idx] = match slot.parent {
    ParentRole::Male => iv_sources.male[idx],
    ParentRole::Female => iv_sources.female[idx],
  };
}
let hidden_power = hidden_power_from_iv(&resolved);
```
- **伝播ルール**: 親が Unknown=32 の場合はそのまま保持。`rng` ソースは Unknown を生成しない。
- **結果**: `ResolvedEgg { ivs: resolved, hidden_power, ..pending.clone() }` を返却し、以降の工程で再計算を不要化。

### 3.3 `matches_filter`
- **署名**: `fn matches_filter(egg: &ResolvedEgg, filter: &IndividualFilter) -> bool`
- **判定順**:
  1. **個体値**: 各ステータスについて `filter.iv_ranges[i].min <= egg.ivs[i] <= filter.iv_ranges[i].max` を適用。Unknown=32 も同じ比較式で扱う。
  2. **Hidden Power**: `egg.hidden_power` が `Known { .. }` の場合のみ、タイプ/威力条件を比較。`Unknown` のまま条件が `Some` なら即 `false`。
  3. **Optional 条件**: `nature`/`gender`/`ability`/`shiny` で `Some` が指定されている場合に完全一致を要求。`None` はスキップ。
- **戻り値**: すべての条件を満たす場合に `true`。評価は短絡的に行い、最初の不一致で `false` を返す。
```
pub fn hidden_power_from_iv(iv: &IvSet) -> HiddenPowerInfo {
  if iv.iter().any(|&v| v == IvValue::UNKNOWN) {
  }
  let type_bits = ((iv[0] & 1)     )
    | ((iv[1] & 1) << 1)
    | ((iv[2] & 1) << 2)
    | ((iv[3] & 1) << 3)
    | ((iv[4] & 1) << 4)
    | ((iv[5] & 1) << 5);
  let power_bits = (((iv[0] >> 1) & 1)     )
    | (((iv[1] >> 1) & 1) << 1)
    | (((iv[2] >> 1) & 1) << 2)
    | (((iv[3] >> 1) & 1) << 3)
    | (((iv[4] >> 1) & 1) << 4)
    | (((iv[5] >> 1) & 1) << 5);
  let r#type = HiddenPowerType::from_index((type_bits * 15 / 63) as u8);
  let power = ((power_bits * 40 / 63) + 30) as u8;
  HiddenPowerInfo::Known { r#type, power }
}
```
- Unknown が含まれる限り Hidden Power は決定不可とする。
- `matches_filter` は `Known` と照合。UI では Unknown を表示 (例: `?/?`).

### 3.5 `resolve_npc_advance`
- **署名**: `fn resolve_npc_advance(seed: u64, frame_threshold: u8, slack: u8) -> (u64, u32, bool)`
- **引数**:
  - `seed`: 現在の LCG Seed。
  - `frame_threshold`: 判定対象となるフレーム閾値 (u8)。
  - `slack`: 許容猶予フレーム (u8)。
- **戻り値**:
  - `(advanced_seed, consumed, is_stable)` のタプル。
    - `advanced_seed`: 全処理後の LCG Seed。
    - `consumed`: 本処理で消費した乱数回数。
    - `is_stable`: `true` なら閾値超過後の余剰フレームが `slack` を超えており、安定とみなす。

#### 処理手順
1. 初期状態として RNG を 3 回進め、`consumed = 3` とする。
2. `elapsed_frames = 0` をセットし、以下の 5 ステップを順番に実行する。各ステップの加算後に `elapsed_frames > frame_threshold` を判定し、達した瞬間に残りのステップはスキップして手順 3 へ進む。
  - **n 分率の定義**: `n_fraction(rand) = (n * rand) >> 32`。
  - 乱数 1 回目で 4 分率を取得し、結果に応じて `elapsed_frames` に加算する（0: +32, 1: +48, 2: +96, 3: +128）。
  - 乱数 2 回目で 2 分率を取得し、方向を決定する（0: 左、1: 右）。方向に応じて `elapsed_frames` に +20（左）または +16（右）。
  - 乱数 3 回目で再度 4 分率を取得し、同じ配分（+32/+48/+96/+128）を加算する。
  - 乱数 4 回目で再度 2 分率を取得し、直前に決定した方向（ステップ 2）との差で追加フレームを決定する（同方向: +0、異方向: +20）。
  - 乱数 5 回目で 4 分率を取得し、再び +32/+48/+96/+128 のいずれかを加算する。
  - 各乱数消費のたびに `consumed += 1`。
3. 閾値へ到達したタイミングで `overflow = elapsed_frames - frame_threshold` を計算し、`is_stable = overflow >= slack` をセットする（猶予フレーム以上であれば安定）。
4. RNG を追加で 2 回進め (`consumed += 2`)、最終的な Seed を取得する。
5. `(seed, consumed, is_stable)` を返却する。

> 備考: 各加算後に即座に閾値判定を行うため、実際に消費される乱数の個数は最小 3（初期消費分）+1 から最大 3+5 まで変動する。4 分率/2 分率の割り当て値はゲームロジック由来であり、将来的な検証では定数化して共有する。

## 4. テストと検証
- **ユニットテスト**
  - Unknown 親 → Unknown タマゴの伝播確認 (各ステータス別)。
  - `IvRange {0, 32}` で Unknown を通し、`{0, 31}` で弾くケース。
  - Hidden Power 計算が Unknown を入力した場合に `HiddenPowerInfo::Unknown` を返すこと。
- **統合テスト**
  - TSV/特性/性別条件と併用した際に Unknown が副作用を起こさないこと。
  - UI フィルター (0-32 スライダー) とバックエンド `matches_filter` の一致を Playwright で検証。

## 5. 実装メモ
- WASM 側: `IvValue` を `u8` で保持し、SIMD 化は 0-31 範囲に限定。32 はビット演算に含めない。
- TypeScript 側: Zustand ストアの既存 `IVRange` 型に 32 を許容する validation を追加予定。
- シリアライズ: JSON でも数値のまま扱い、特別なタグは追加しない。
