/* tslint:disable */
/* eslint-disable */

export class ArrayUtils {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 32bit配列の合計値を計算
   *
   * # Arguments
   * * `array` - 対象配列
   *
   * # Returns
   * 合計値
   */
  static sum_u32_array(array: Uint32Array): bigint;
  /**
   * 32bit配列の平均値を計算
   *
   * # Arguments
   * * `array` - 対象配列
   *
   * # Returns
   * 平均値
   */
  static average_u32_array(array: Uint32Array): number;
  /**
   * 32bit配列の最大値を取得
   *
   * # Arguments
   * * `array` - 対象配列
   *
   * # Returns
   * 最大値（配列が空の場合は0）
   */
  static max_u32_array(array: Uint32Array): number;
  /**
   * 32bit配列の最小値を取得
   *
   * # Arguments
   * * `array` - 対象配列
   *
   * # Returns
   * 最小値（配列が空の場合は0）
   */
  static min_u32_array(array: Uint32Array): number;
  /**
   * 配列の重複要素を除去
   *
   * # Arguments
   * * `array` - 対象配列
   *
   * # Returns
   * 重複が除去された配列
   */
  static deduplicate_u32_array(array: Uint32Array): Uint32Array;
}

export class BWGenerationConfig {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいBW準拠設定を作成
   */
  constructor(version: GameVersion, encounter_type: EncounterType, tid: number, sid: number, sync_enabled: boolean, sync_nature_id: number, is_shiny_locked: boolean, has_shiny_charm: boolean);
  /**
   * getter methods
   */
  readonly get_version: GameVersion;
  readonly get_encounter_type: EncounterType;
  readonly get_tid: number;
  readonly get_sid: number;
  readonly get_sync_enabled: boolean;
  readonly get_sync_nature_id: number;
  readonly get_is_shiny_locked: boolean;
  readonly get_has_shiny_charm: boolean;
}

export class BitUtils {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 32bit値の左ローテート
   *
   * # Arguments
   * * `value` - ローテートする値
   * * `count` - ローテート回数
   *
   * # Returns
   * ローテートされた値
   */
  static rotate_left_32(value: number, count: number): number;
  /**
   * 32bit値の右ローテート
   *
   * # Arguments
   * * `value` - ローテートする値
   * * `count` - ローテート回数
   *
   * # Returns
   * ローテートされた値
   */
  static rotate_right_32(value: number, count: number): number;
  /**
   * 指定したビット位置の値を取得
   *
   * # Arguments
   * * `value` - 対象の値
   * * `bit_position` - ビット位置（0-31）
   *
   * # Returns
   * 指定ビットの値（0または1）
   */
  static get_bit(value: number, bit_position: number): number;
  /**
   * 指定したビット位置を設定
   *
   * # Arguments
   * * `value` - 対象の値
   * * `bit_position` - ビット位置（0-31）
   * * `bit_value` - 設定する値（0または1）
   *
   * # Returns
   * ビットが設定された値
   */
  static set_bit(value: number, bit_position: number, bit_value: number): number;
  /**
   * ビット数をカウント
   *
   * # Arguments
   * * `value` - 対象の値
   *
   * # Returns
   * 設定されているビット数
   */
  static count_bits(value: number): number;
  /**
   * ビットフィールドを抽出
   *
   * # Arguments
   * * `value` - 対象の値
   * * `start_bit` - 開始ビット位置
   * * `bit_count` - 抽出するビット数
   *
   * # Returns
   * 抽出されたビットフィールド
   */
  static extract_bits(value: number, start_bit: number, bit_count: number): number;
}

export class DSConfigJs {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新規作成
   */
  constructor(mac: Uint8Array, nazo: Uint32Array, hardware: string);
  readonly mac: Uint8Array;
  readonly nazo: Uint32Array;
  readonly hardware: string;
}

/**
 * 砂煙出現内容の種類
 */
export enum DustCloudContent {
  /**
   * ポケモン出現
   */
  Pokemon = 0,
  /**
   * ジュエル類出現
   */
  Jewel = 1,
  /**
   * 進化石類出現
   */
  EvolutionStone = 2,
}

export class EggBootTimingSearchIterator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * コンストラクタ
   *
   * # Arguments
   * - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
   * - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
   * - `time_range`: 時刻範囲パラメータ
   * - `search_range`: 検索範囲パラメータ
   * - `conditions`: 孵化条件
   * - `parents`: 親個体値
   * - `filter_js`: 個体フィルター（オプション）
   * - `consider_npc_consumption`: NPC消費を考慮するか
   * - `game_mode`: ゲームモード
   * - `user_offset`: ユーザー指定オフセット
   * - `advance_count`: 消費数
   */
  constructor(ds_config: DSConfigJs, segment: SegmentParamsJs, time_range: TimeRangeParamsJs, search_range: SearchRangeParamsJs, conditions: GenerationConditionsJs, parents: ParentsIVsJs, filter_js: IndividualFilterJs | null | undefined, consider_npc_consumption: boolean, game_mode: GameMode, user_offset: bigint, advance_count: number);
  /**
   * 次のバッチを取得
   *
   * - result_limit件見つかったら即return
   * - chunk_seconds秒分処理したら結果がなくても一旦return
   * - 検索範囲を全て処理したらfinished=trueになる
   */
  next_batch(result_limit: number, chunk_seconds: number): Array<any>;
  /**
   * 検索が完了したかどうか
   */
  readonly isFinished: boolean;
  /**
   * 処理済み秒数
   */
  readonly processedSeconds: number;
  /**
   * 総秒数
   */
  readonly totalSeconds: number;
  /**
   * 進捗率 (0.0 - 1.0)
   */
  readonly progress: number;
}

export class EggBootTimingSearchResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  readonly year: number;
  readonly month: number;
  readonly date: number;
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
  readonly timer0: number;
  readonly vcount: number;
  readonly keyCode: number;
  readonly lcgSeedHex: string;
  readonly advance: bigint;
  readonly isStable: boolean;
  readonly ivs: Uint8Array;
  readonly nature: number;
  /**
   * Gender: 0=Male, 1=Female, 2=Genderless
   */
  readonly gender: number;
  /**
   * Ability slot: 0=Ability1, 1=Ability2, 2=Hidden
   */
  readonly ability: number;
  /**
   * Shiny type: 0=Normal (not shiny), 1=Square shiny, 2=Star shiny
   */
  readonly shiny: number;
  readonly pid: number;
  readonly hpType: number;
  readonly hpPower: number;
  readonly hpKnown: boolean;
  readonly mtSeed: number;
  readonly mtSeedHex: string;
}

export class EggSeedEnumeratorJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor(base_seed: bigint, user_offset: bigint, count: number, conditions: GenerationConditionsJs, parents: ParentsIVsJs, filter: IndividualFilterJs, consider_npc_consumption: boolean, game_mode: GameMode);
  /**
   * Returns the next egg as a JsValue or undefined if exhausted
   */
  next_egg(): any;
  readonly remaining: number;
}

export class EncounterCalculator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいEncounterCalculatorインスタンスを作成
   */
  constructor();
  /**
   * エンカウントスロットを計算
   *
   * # Arguments
   * * `version` - ゲームバージョン
   * * `encounter_type` - エンカウントタイプ
   * * `random_value` - 乱数値（32bit）
   *
   * # Returns
   * エンカウントスロット番号（0-11）
   */
  static calculate_encounter_slot(version: GameVersion, encounter_type: EncounterType, random_value: number): number;
  /**
   * スロット番号をテーブルインデックスに変換
   *
   * # Arguments
   * * `encounter_type` - エンカウントタイプ
   * * `slot` - スロット番号
   *
   * # Returns
   * テーブルインデックス
   */
  static slot_to_table_index(encounter_type: EncounterType, slot: number): number;
  /**
   * 砂煙の出現内容を判定
   *
   * # Arguments
   * * `slot` - 砂煙スロット値（0-2）
   *
   * # Returns
   * 出現内容の種類
   */
  static get_dust_cloud_content(slot: number): DustCloudContent;
}

/**
 * エンカウントタイプ列挙型
 */
export enum EncounterType {
  /**
   * 通常エンカウント（草むら・洞窟・ダンジョン共通）
   */
  Normal = 0,
  /**
   * なみのり
   */
  Surfing = 1,
  /**
   * つりざお
   */
  Fishing = 2,
  /**
   * 揺れる草むら（特殊エンカウント）
   */
  ShakingGrass = 3,
  /**
   * 砂煙（特殊エンカウント）
   */
  DustCloud = 4,
  /**
   * ポケモンの影（特殊エンカウント）
   */
  PokemonShadow = 5,
  /**
   * 水泡（なみのり版特殊エンカウント）
   */
  SurfingBubble = 6,
  /**
   * 水泡釣り（釣り版特殊エンカウント）
   */
  FishingBubble = 7,
  /**
   * 固定シンボル（レジェンダリー等）- シンクロ有効
   */
  StaticSymbol = 10,
  /**
   * 御三家受け取り - シンクロ無効
   */
  StaticStarter = 11,
  /**
   * 化石復元 - シンクロ無効
   */
  StaticFossil = 12,
  /**
   * イベント配布 - シンクロ無効
   */
  StaticEvent = 13,
  /**
   * 徘徊ポケモン（ドキュメント仕様準拠）
   */
  Roamer = 20,
}

export class EndianUtils {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 32bit値のバイトスワップ
   *
   * # Arguments
   * * `value` - 変換する32bit値
   *
   * # Returns
   * バイトスワップされた値
   */
  static swap_bytes_32(value: number): number;
  /**
   * 16bit値のバイトスワップ
   *
   * # Arguments
   * * `value` - 変換する16bit値
   *
   * # Returns
   * バイトスワップされた値
   */
  static swap_bytes_16(value: number): number;
  /**
   * 64bit値のバイトスワップ
   *
   * # Arguments
   * * `value` - 変換する64bit値
   *
   * # Returns
   * バイトスワップされた値
   */
  static swap_bytes_64(value: bigint): bigint;
  /**
   * ビッグエンディアン32bit値をリトルエンディアンに変換
   */
  static be32_to_le(value: number): number;
  /**
   * リトルエンディアン32bit値をビッグエンディアンに変換
   */
  static le32_to_be(value: number): number;
}

export class EnumeratedPokemonData {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 任意: 元の RawPokemonData を複製して取得
   */
  into_raw(): RawPokemonData;
  readonly get_advance: bigint;
  readonly get_seed: bigint;
  readonly get_pid: number;
  readonly get_nature: number;
  readonly get_sync_applied: boolean;
  readonly get_ability_slot: number;
  readonly get_gender_value: number;
  readonly get_encounter_slot_value: number;
  readonly get_encounter_type: number;
  readonly get_level_rand_value: bigint;
  readonly get_shiny_type: number;
}

export class EverstonePlanJs {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static fixed(nature_index: number): EverstonePlanJs;
  static readonly None: EverstonePlanJs;
}

export class ExtraResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 消費した乱数回数
   */
  advances: number;
  /**
   * 成功フラグ（重複回避完了）
   */
  success: boolean;
  /**
   * 最終的な3つの値
   */
  value1: number;
  value2: number;
  value3: number;
  readonly get_advances: number;
  readonly get_success: boolean;
  readonly get_value1: number;
  readonly get_value2: number;
  readonly get_value3: number;
}

/**
 * ゲームモード列挙型（仕様書準拠）
 */
export enum GameMode {
  /**
   * BW 始めから（セーブ有り）
   */
  BwNewGameWithSave = 0,
  /**
   * BW 始めから（セーブ無し）
   */
  BwNewGameNoSave = 1,
  /**
   * BW 続きから
   */
  BwContinue = 2,
  /**
   * BW2 始めから（思い出リンク済みセーブ有り）
   */
  Bw2NewGameWithMemoryLinkSave = 3,
  /**
   * BW2 始めから（思い出リンク無しセーブ有り）
   */
  Bw2NewGameNoMemoryLinkSave = 4,
  /**
   * BW2 始めから（セーブ無し）
   */
  Bw2NewGameNoSave = 5,
  /**
   * BW2 続きから（思い出リンク済み）
   */
  Bw2ContinueWithMemoryLink = 6,
  /**
   * BW2 続きから（思い出リンク無し）
   */
  Bw2ContinueNoMemoryLink = 7,
}

/**
 * ゲームバージョン列挙型
 */
export enum GameVersion {
  B = 0,
  W = 1,
  B2 = 2,
  W2 = 3,
}

export class GenderRatio {
  free(): void;
  [Symbol.dispose](): void;
  constructor(threshold: number, genderless: boolean);
  resolve(gender_value: number): number;
  threshold: number;
  genderless: boolean;
}

export class GenerationConditionsJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  set_everstone(plan: EverstonePlanJs): void;
  set_trainer_ids(ids: TrainerIds): void;
  set_gender_ratio(ratio: GenderRatio): void;
  has_nidoran_flag: boolean;
  uses_ditto: boolean;
  allow_hidden_ability: boolean;
  female_parent_has_hidden: boolean;
  reroll_count: number;
}

export class IndividualFilterJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  set_iv_range(stat_index: number, min: number, max: number): void;
  set_nature(nature_index: number): void;
  set_gender(gender: number): void;
  set_ability(ability: number): void;
  /**
   * Set shiny filter mode
   * 0 = All (no filter), 1 = Shiny (star OR square), 2 = Star, 3 = Square, 4 = NonShiny
   */
  set_shiny_filter_mode(mode: number): void;
  set_hidden_power_type(hp_type: number): void;
  set_hidden_power_power(power: number): void;
}

export class MtSeedBootTimingSearchIterator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * コンストラクタ
   *
   * # Arguments
   * - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
   * - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
   * - `time_range`: 時刻範囲パラメータ
   * - `search_range`: 検索範囲パラメータ
   * - `target_seeds`: 検索対象のMT Seed値（複数可）
   */
  constructor(ds_config: DSConfigJs, segment: SegmentParamsJs, time_range: TimeRangeParamsJs, search_range: SearchRangeParamsJs, target_seeds: Uint32Array);
  /**
   * 次のバッチを取得
   *
   * - max_results件見つかったら即return
   * - chunk_seconds秒分処理したら結果がなくても一旦return
   * - 検索範囲を全て処理したらfinished=trueになる
   */
  next_batch(max_results: number, chunk_seconds: number): MtSeedBootTimingSearchResults;
  /**
   * 検索が完了したかどうか
   */
  readonly isFinished: boolean;
  /**
   * 処理済み秒数
   */
  readonly processedSeconds: number;
  /**
   * 総秒数
   */
  readonly totalSeconds: number;
  /**
   * 進捗率 (0.0 - 1.0)
   */
  readonly progress: number;
}

export class MtSeedBootTimingSearchResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  readonly mtSeed: number;
  readonly mtSeedHex: string;
  readonly lcgSeedHigh: number;
  readonly lcgSeedLow: number;
  readonly lcgSeedHex: string;
  readonly year: number;
  readonly month: number;
  readonly day: number;
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
  readonly timer0: number;
  readonly vcount: number;
  readonly keyCode: number;
}

export class MtSeedBootTimingSearchResults {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 結果をJavaScript配列として取得
   */
  to_array(): Array<any>;
  /**
   * 指定インデックスの結果を取得
   */
  get(index: number): MtSeedBootTimingSearchResult | undefined;
  readonly length: number;
  readonly processedInChunk: number;
}

export class NumberUtils {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 16進数文字列を32bit整数に変換
   *
   * # Arguments
   * * `hex_str` - 16進数文字列（0xプレフィックス可）
   *
   * # Returns
   * 変換された整数値（エラー時は0）
   */
  static hex_string_to_u32(hex_str: string): number;
  /**
   * 32bit整数を16進数文字列に変換
   *
   * # Arguments
   * * `value` - 変換する整数値
   * * `uppercase` - 大文字で出力するか
   *
   * # Returns
   * 16進数文字列
   */
  static u32_to_hex_string(value: number, uppercase: boolean): string;
  /**
   * BCD（Binary Coded Decimal）エンコード
   *
   * # Arguments
   * * `value` - エンコードする値（0-99）
   *
   * # Returns
   * BCDエンコードされた値
   */
  static encode_bcd(value: number): number;
  /**
   * BCD（Binary Coded Decimal）デコード
   *
   * # Arguments
   * * `bcd_value` - デコードするBCD値
   *
   * # Returns
   * デコードされた値
   */
  static decode_bcd(bcd_value: number): number;
  /**
   * パーセンテージを乱数閾値に変換
   *
   * # Arguments
   * * `percentage` - パーセンテージ（0.0-100.0）
   *
   * # Returns
   * 32bit乱数閾値
   */
  static percentage_to_threshold(percentage: number): number;
  /**
   * 32bit乱数閾値をパーセンテージに変換
   *
   * # Arguments
   * * `threshold` - 32bit乱数閾値
   *
   * # Returns
   * パーセンテージ
   */
  static threshold_to_percentage(threshold: number): number;
}

export class OffsetCalculator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいOffsetCalculatorインスタンスを作成
   *
   * # Arguments
   * * `seed` - 初期Seed値
   */
  constructor(seed: bigint);
  /**
   * 次の32bit乱数値を取得（上位32bit）
   *
   * # Returns
   * 32bit乱数値
   */
  next_rand(): number;
  /**
   * 指定回数だけ乱数を消費（Rand×n）
   *
   * # Arguments
   * * `count` - 消費する回数
   */
  consume_random(count: number): void;
  /**
   * 計算器をリセット
   *
   * # Arguments
   * * `new_seed` - 新しいSeed値
   */
  reset(new_seed: bigint): void;
  /**
   * TID/SID決定処理（リファレンス実装準拠）
   *
   * # Returns
   * TidSidResult
   */
  calculate_tid_sid(): TidSidResult;
  /**
   * 表住人決定処理（BW：固定10回乱数消費）
   */
  determine_front_residents(): void;
  /**
   * 裏住人決定処理（BW：固定3回乱数消費）
   */
  determine_back_residents(): void;
  /**
   * 住人決定一括処理（BW専用）
   */
  determine_all_residents(): void;
  /**
   * Probability Table処理（仕様書準拠の6段階テーブル処理）
   */
  probability_table_process(): void;
  /**
   * PT操作×n回
   */
  probability_table_process_multiple(count: number): void;
  /**
   * Extra処理（BW2専用：重複値回避ループ）
   * 3つの値（0-14範囲）がすべて異なるまでループ
   */
  extra_process(): ExtraResult;
  /**
   * ゲーム初期化処理の総合実行（仕様書準拠）
   *
   * # Arguments
   * * `mode` - ゲームモード
   *
   * # Returns
   * 初期化完了時の進行回数
   */
  execute_game_initialization(mode: GameMode): number;
  /**
   * 現在の進行回数を取得
   *
   * # Returns
   * 進行回数
   */
  readonly get_advances: number;
  /**
   * 現在のSeed値を取得
   *
   * # Returns
   * 現在のSeed値
   */
  readonly get_current_seed: bigint;
}

export class PIDCalculator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいPIDCalculatorインスタンスを作成
   */
  constructor();
  /**
   * BW/BW2準拠 統一PID生成
   * 32bit乱数 ^ 0x10000 の計算（固定・野生共通）
   *
   * # Arguments
   * * `r1` - 乱数値1
   *
   * # Returns
   * 基本PID（ID補正前）
   */
  static generate_base_pid(r1: number): number;
  /**
   * ID補正処理
   * 性格値下位 ^ トレーナーID ^ 裏ID の奇偶性で最上位bitを調整
   *
   * # Arguments
   * * `pid` - 基本PID
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   *
   * # Returns
   * ID補正後PID
   */
  static apply_id_correction(pid: number, tid: number, sid: number): number;
  /**
   * BW/BW2準拠 野生ポケモンのPID生成
   * 32bit乱数 ^ 0x10000 + ID補正処理
   *
   * # Arguments
   * * `r1` - 乱数値1
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   *
   * # Returns
   * 生成されたPID（ID補正適用後）
   */
  static generate_wild_pid(r1: number, tid: number, sid: number): number;
  /**
   * BW/BW2準拠 固定シンボルポケモンのPID生成
   * 32bit乱数 ^ 0x10000 + ID補正処理
   *
   * # Arguments
   * * `r1` - 乱数値1
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   *
   * # Returns
   * 生成されたPID（ID補正適用後）
   */
  static generate_static_pid(r1: number, tid: number, sid: number): number;
  /**
   * BW/BW2準拠 徘徊ポケモンのPID生成
   * 32bit乱数 ^ 0x10000 + ID補正処理
   *
   * # Arguments
   * * `r1` - 乱数値1
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   *
   * # Returns
   * 生成されたPID（ID補正適用後）
   */
  static generate_roamer_pid(r1: number, tid: number, sid: number): number;
  /**
   * BW/BW2準拠 イベントポケモンのPID生成
   * 32bit乱数 ^ 0x10000（ID補正なし - 先頭特性無効）
   *
   * # Arguments
   * * `r1` - 乱数値1
   *
   * # Returns
   * 生成されたPID（ID補正なし）
   */
  static generate_event_pid(r1: number): number;
  /**
   * ギフトポケモンのPID生成
   * 特殊な計算式を使用
   *
   * # Arguments
   * * `r1` - 乱数値1
   * * `r2` - 乱数値2
   *
   * # Returns
   * 生成されたPID
   */
  static generate_gift_pid(r1: number, r2: number): number;
  /**
   * タマゴのPID生成
   * 特殊な計算式を使用
   *
   * # Arguments
   * * `r1` - 乱数値1
   * * `r2` - 乱数値2
   *
   * # Returns
   * 生成されたPID
   */
  static generate_egg_pid(r1: number, r2: number): number;
}

export class ParentsIVsJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  set male(value: Uint8Array);
  set female(value: Uint8Array);
}

export class PersonalityRNG {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいPersonalityRNGインスタンスを作成
   *
   * # Arguments
   * * `seed` - 初期Seed値（64bit）
   */
  constructor(seed: bigint);
  /**
   * 次の32bit乱数値を取得（上位32bit）
   *
   * # Returns
   * 上位32bitの乱数値
   */
  next(): number;
  /**
   * 次の64bit乱数値を取得
   *
   * # Returns
   * 64bit乱数値（内部状態そのもの）
   */
  next_u64(): bigint;
  /**
   * 指定回数だけ乱数を進める
   *
   * # Arguments
   * * `advances` - 進める回数
   */
  advance(advances: number): void;
  /**
   * Seedをリセット
   *
   * # Arguments
   * * `initial_seed` - リセット後のSeed値
   */
  reset(initial_seed: bigint): void;
  /**
   * 0x0からの進行度を計算
   *
   * # Arguments
   * * `seed` - 計算対象のSeed値
   *
   * # Returns
   * 0x0からの進行度
   */
  static get_index(seed: bigint): bigint;
  /**
   * 2つのSeed間の距離を計算
   *
   * # Arguments
   * * `from_seed` - 開始Seed
   * * `to_seed` - 終了Seed
   *
   * # Returns
   * from_seedからto_seedまでの距離
   */
  static distance_between(from_seed: bigint, to_seed: bigint): bigint;
  /**
   * 指定Seedから現在のSeedまでの距離
   *
   * # Arguments
   * * `source_seed` - 開始Seed
   *
   * # Returns
   * source_seedから現在のSeedまでの距離
   */
  distance_from(source_seed: bigint): bigint;
  /**
   * 現在のSeed値を取得
   *
   * # Returns
   * 現在の内部Seed値
   */
  readonly current_seed: bigint;
  /**
   * Seed値を設定
   *
   * # Arguments
   * * `new_seed` - 新しいSeed値
   */
  set seed(value: bigint);
}

export class PokemonGenerator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいPokemonGeneratorインスタンスを作成
   */
  constructor();
  /**
   * BW/BW2準拠 単体ポケモン生成（統括関数）
   *
   * # Arguments
   * * `seed` - 初期Seed値
   * * `config` - BW準拠設定
   *
   * # Returns
   * 生成されたポケモンデータ
   */
  static generate_single_pokemon_bw(seed: bigint, config: BWGenerationConfig): RawPokemonData;
  /**
   * オフセット適用後の生成開始Seedを計算
   */
  static calculate_generation_seed(initial_seed: bigint, offset: bigint): bigint;
  /**
   * BW/BW2準拠 バッチ生成（offsetのみ）
   *
   * # Arguments
   * * `base_seed` - 列挙の初期Seed
   * * `offset` - 最初の生成までの前進数（ゲーム内不定消費を含めた開始位置）
   * * `count` - 生成数（0なら空）
   * * `config` - BW準拠設定
   *
   * # Returns
   * 生成されたポケモンデータの配列
   */
  static generate_pokemon_batch_bw(base_seed: bigint, offset: bigint, count: number, config: BWGenerationConfig): RawPokemonData[];
}

export class RawPokemonData {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * getter methods for JavaScript access
   */
  readonly get_seed: bigint;
  readonly get_pid: number;
  readonly get_nature: number;
  readonly get_ability_slot: number;
  readonly get_gender_value: number;
  readonly get_encounter_slot_value: number;
  readonly get_level_rand_value: bigint;
  readonly get_shiny_type: number;
  readonly get_sync_applied: boolean;
  readonly get_encounter_type: number;
}

export class SearchRangeParamsJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor(start_year: number, start_month: number, start_day: number, range_seconds: number);
  readonly start_year: number;
  readonly start_month: number;
  readonly start_day: number;
  readonly range_seconds: number;
}

export class SeedEnumerator {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 列挙器を作成
   */
  constructor(base_seed: bigint, user_offset: bigint, count: number, config: BWGenerationConfig, game_mode: GameMode);
  /**
   * 次のポケモンを生成（残数0なら undefined を返す）
   */
  next_pokemon(): EnumeratedPokemonData | undefined;
  /**
   * 残数を取得
   */
  readonly remaining: number;
}

export class SegmentParamsJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor(timer0: number, vcount: number, key_code: number);
  readonly timer0: number;
  readonly vcount: number;
  readonly key_code: number;
}

export class ShinyChecker {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * 新しいShinyCheckerインスタンスを作成
   */
  constructor();
  /**
   * 色違い判定
   *
   * # Arguments
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   * * `pid` - ポケモンのPID
   *
   * # Returns
   * 色違いかどうか
   */
  static is_shiny(tid: number, sid: number, pid: number): boolean;
  /**
   * 色違い値の計算
   * TID ^ SID ^ PID上位16bit ^ PID下位16bit
   *
   * # Arguments
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   * * `pid` - ポケモンのPID
   *
   * # Returns
   * 色違い値
   */
  static get_shiny_value(tid: number, sid: number, pid: number): number;
  /**
   * 色違いタイプの判定
   *
   * # Arguments
   * * `shiny_value` - 色違い値
   *
   * # Returns
   * 色違いタイプ
   */
  static get_shiny_type(shiny_value: number): ShinyType;
  /**
   * 色違い判定とタイプを同時に取得
   *
   * # Arguments
   * * `tid` - トレーナーID
   * * `sid` - シークレットID
   * * `pid` - ポケモンのPID
   *
   * # Returns
   * 色違いタイプ
   */
  static check_shiny_type(tid: number, sid: number, pid: number): ShinyType;
  /**
   * 色違い確率の計算
   * 通常の色違い確率を計算（参考用）
   *
   * # Returns
   * 色違い確率（分母）
   */
  static shiny_probability(): number;
  /**
   * 光るお守り効果の確率計算
   *
   * # Arguments
   * * `has_shiny_charm` - 光るお守りを持っているか
   *
   * # Returns
   * 色違い確率（分母）
   */
  static shiny_probability_with_charm(has_shiny_charm: boolean): number;
}

/**
 * 色違いタイプ列挙型
 */
export enum ShinyType {
  /**
   * 通常（色違いでない）
   */
  Normal = 0,
  /**
   * 四角い色違い（一般的な色違い）
   */
  Square = 1,
  /**
   * 星形色違い（特殊な色違い）
   */
  Star = 2,
}

export class StatRange {
  free(): void;
  [Symbol.dispose](): void;
  constructor(min: number, max: number);
  contains(value: number): boolean;
  min: number;
  max: number;
}

export class TidSidResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * TID（トレーナーID下位16bit）
   */
  tid: number;
  /**
   * SID（シークレットID上位16bit）
   */
  sid: number;
  /**
   * 消費した乱数回数
   */
  advances_used: number;
  readonly get_tid: number;
  readonly get_sid: number;
  readonly get_advances_used: number;
}

export class TimeRangeParamsJs {
  free(): void;
  [Symbol.dispose](): void;
  constructor(hour_start: number, hour_end: number, minute_start: number, minute_end: number, second_start: number, second_end: number);
  readonly hour_start: number;
  readonly hour_end: number;
  readonly minute_start: number;
  readonly minute_end: number;
  readonly second_start: number;
  readonly second_end: number;
}

export class TrainerIds {
  free(): void;
  [Symbol.dispose](): void;
  constructor(tid: number, sid: number);
  tid: number;
  sid: number;
  tsv: number;
}

export class ValidationUtils {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * TIDの妥当性チェック
   *
   * # Arguments
   * * `tid` - トレーナーID
   *
   * # Returns
   * 妥当性
   */
  static is_valid_tid(_tid: number): boolean;
  /**
   * SIDの妥当性チェック
   *
   * # Arguments
   * * `sid` - シークレットID
   *
   * # Returns
   * 妥当性
   */
  static is_valid_sid(_sid: number): boolean;
  /**
   * 性格値の妥当性チェック
   *
   * # Arguments
   * * `nature` - 性格値
   *
   * # Returns
   * 妥当性
   */
  static is_valid_nature(nature: number): boolean;
  /**
   * 特性スロットの妥当性チェック
   *
   * # Arguments
   * * `ability_slot` - 特性スロット
   *
   * # Returns
   * 妥当性
   */
  static is_valid_ability_slot(ability_slot: number): boolean;
  /**
   * 16進数文字列の妥当性チェック
   *
   * # Arguments
   * * `hex_str` - 16進数文字列
   *
   * # Returns
   * 妥当性
   */
  static is_valid_hex_string(hex_str: string): boolean;
  /**
   * Seed値の妥当性チェック
   *
   * # Arguments
   * * `seed` - Seed値
   *
   * # Returns
   * 妥当性
   */
  static is_valid_seed(seed: bigint): boolean;
}

/**
 * オフセット計算統合API（仕様書準拠）
 */
export function calculate_game_offset(initial_seed: bigint, mode: GameMode): number;

/**
 * TID/SID決定処理統合API（仕様書準拠）
 */
export function calculate_tid_sid_from_seed(initial_seed: bigint, mode: GameMode): TidSidResult;

/**
 * WASM公開関数: IVコードデコード
 *
 * IVコードをIVセットにデコードする
 *
 * # Arguments
 * * `code` - IVコード (30bit)
 *
 * # Returns
 * IVセット [HP, Atk, Def, SpA, SpD, Spe]
 */
export function decode_iv_code_wasm(code: number): Uint8Array;

/**
 * WASM公開関数: IVセット導出
 *
 * MT SeedとMT消費数からIVセットを導出する
 *
 * # Arguments
 * * `mt_seed` - MT Seed
 * * `advances` - MT消費数
 *
 * # Returns
 * IVセット [HP, Atk, Def, SpA, SpD, Spe]
 */
export function derive_iv_set_wasm(mt_seed: number, advances: number): Uint8Array;

/**
 * WASM公開関数: IVコードエンコード
 *
 * IVセットをIVコードにエンコードする
 *
 * # Arguments
 * * `ivs` - IVセット [HP, Atk, Def, SpA, SpD, Spe]
 *
 * # Returns
 * IVコード (30bit)
 */
export function encode_iv_code_wasm(ivs: Uint8Array): number;

/**
 * WASM公開関数: MT Seed検索セグメント実行
 *
 * SIMD最適化版を使用して検索を実行
 *
 * # Arguments
 * * `start` - 検索開始Seed (inclusive)
 * * `end` - 検索終了Seed (inclusive)
 * * `advances` - MT消費数
 * * `target_codes` - 検索対象IVコードのスライス
 *
 * # Returns
 * フラット配列 [seed0, code0, seed1, code1, ...]
 */
export function mt_seed_search_segment(start: number, end: number, advances: number, target_codes: Uint32Array): Uint32Array;

/**
 * WebAssembly向けバッチSHA-1計算エントリポイント
 * `messages` は 16 ワード単位（512bit）で並ぶフラットな配列である必要がある
 */
export function sha1_hash_batch(messages: Uint32Array): Uint32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_eggboottimingsearchresult_free: (a: number, b: number) => void;
  readonly eggboottimingsearchresult_year: (a: number) => number;
  readonly eggboottimingsearchresult_month: (a: number) => number;
  readonly eggboottimingsearchresult_date: (a: number) => number;
  readonly eggboottimingsearchresult_hour: (a: number) => number;
  readonly eggboottimingsearchresult_minute: (a: number) => number;
  readonly eggboottimingsearchresult_second: (a: number) => number;
  readonly eggboottimingsearchresult_timer0: (a: number) => number;
  readonly eggboottimingsearchresult_vcount: (a: number) => number;
  readonly eggboottimingsearchresult_key_code: (a: number) => number;
  readonly eggboottimingsearchresult_lcg_seed_hex: (a: number, b: number) => void;
  readonly eggboottimingsearchresult_advance: (a: number) => bigint;
  readonly eggboottimingsearchresult_is_stable: (a: number) => number;
  readonly eggboottimingsearchresult_ivs: (a: number, b: number) => void;
  readonly eggboottimingsearchresult_nature: (a: number) => number;
  readonly eggboottimingsearchresult_gender: (a: number) => number;
  readonly eggboottimingsearchresult_ability: (a: number) => number;
  readonly eggboottimingsearchresult_shiny: (a: number) => number;
  readonly eggboottimingsearchresult_pid: (a: number) => number;
  readonly eggboottimingsearchresult_hp_type: (a: number) => number;
  readonly eggboottimingsearchresult_hp_power: (a: number) => number;
  readonly eggboottimingsearchresult_hp_known: (a: number) => number;
  readonly eggboottimingsearchresult_mt_seed: (a: number) => number;
  readonly eggboottimingsearchresult_mt_seed_hex: (a: number, b: number) => void;
  readonly __wbg_eggboottimingsearchiterator_free: (a: number, b: number) => void;
  readonly eggboottimingsearchiterator_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: bigint, l: number) => void;
  readonly eggboottimingsearchiterator_is_finished: (a: number) => number;
  readonly eggboottimingsearchiterator_processed_seconds: (a: number) => number;
  readonly eggboottimingsearchiterator_total_seconds: (a: number) => number;
  readonly eggboottimingsearchiterator_progress: (a: number) => number;
  readonly eggboottimingsearchiterator_next_batch: (a: number, b: number, c: number) => number;
  readonly __wbg_get_statrange_min: (a: number) => number;
  readonly __wbg_set_statrange_min: (a: number, b: number) => void;
  readonly statrange_new: (a: number, b: number) => number;
  readonly statrange_contains: (a: number, b: number) => number;
  readonly __wbg_get_genderratio_threshold: (a: number) => number;
  readonly __wbg_set_genderratio_threshold: (a: number, b: number) => void;
  readonly __wbg_get_genderratio_genderless: (a: number) => number;
  readonly __wbg_set_genderratio_genderless: (a: number, b: number) => void;
  readonly genderratio_new: (a: number, b: number) => number;
  readonly genderratio_resolve: (a: number, b: number) => number;
  readonly __wbg_everstoneplanjs_free: (a: number, b: number) => void;
  readonly everstoneplanjs_none: () => number;
  readonly everstoneplanjs_fixed: (a: number) => number;
  readonly __wbg_get_trainerids_tid: (a: number) => number;
  readonly __wbg_set_trainerids_tid: (a: number, b: number) => void;
  readonly __wbg_get_trainerids_sid: (a: number) => number;
  readonly __wbg_set_trainerids_sid: (a: number, b: number) => void;
  readonly __wbg_get_trainerids_tsv: (a: number) => number;
  readonly __wbg_set_trainerids_tsv: (a: number, b: number) => void;
  readonly trainerids_new: (a: number, b: number) => number;
  readonly __wbg_generationconditionsjs_free: (a: number, b: number) => void;
  readonly __wbg_get_generationconditionsjs_has_nidoran_flag: (a: number) => number;
  readonly __wbg_set_generationconditionsjs_has_nidoran_flag: (a: number, b: number) => void;
  readonly __wbg_get_generationconditionsjs_uses_ditto: (a: number) => number;
  readonly __wbg_set_generationconditionsjs_uses_ditto: (a: number, b: number) => void;
  readonly __wbg_get_generationconditionsjs_allow_hidden_ability: (a: number) => number;
  readonly __wbg_set_generationconditionsjs_allow_hidden_ability: (a: number, b: number) => void;
  readonly __wbg_get_generationconditionsjs_female_parent_has_hidden: (a: number) => number;
  readonly __wbg_set_generationconditionsjs_female_parent_has_hidden: (a: number, b: number) => void;
  readonly __wbg_get_generationconditionsjs_reroll_count: (a: number) => number;
  readonly __wbg_set_generationconditionsjs_reroll_count: (a: number, b: number) => void;
  readonly generationconditionsjs_new: () => number;
  readonly generationconditionsjs_set_everstone: (a: number, b: number) => void;
  readonly generationconditionsjs_set_trainer_ids: (a: number, b: number) => void;
  readonly generationconditionsjs_set_gender_ratio: (a: number, b: number) => void;
  readonly individualfilterjs_new: () => number;
  readonly individualfilterjs_set_iv_range: (a: number, b: number, c: number, d: number) => void;
  readonly individualfilterjs_set_nature: (a: number, b: number) => void;
  readonly individualfilterjs_set_gender: (a: number, b: number) => void;
  readonly individualfilterjs_set_ability: (a: number, b: number) => void;
  readonly individualfilterjs_set_shiny_filter_mode: (a: number, b: number) => void;
  readonly individualfilterjs_set_hidden_power_type: (a: number, b: number) => void;
  readonly individualfilterjs_set_hidden_power_power: (a: number, b: number) => void;
  readonly parentsivsjs_new: () => number;
  readonly parentsivsjs_set_male: (a: number, b: number, c: number) => void;
  readonly parentsivsjs_set_female: (a: number, b: number, c: number) => void;
  readonly __wbg_eggseedenumeratorjs_free: (a: number, b: number) => void;
  readonly eggseedenumeratorjs_new: (a: bigint, b: bigint, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly eggseedenumeratorjs_next_egg: (a: number) => number;
  readonly eggseedenumeratorjs_remaining: (a: number) => number;
  readonly encountercalculator_new: () => number;
  readonly encountercalculator_calculate_encounter_slot: (a: number, b: number, c: number) => number;
  readonly encountercalculator_slot_to_table_index: (a: number, b: number) => number;
  readonly encountercalculator_get_dust_cloud_content: (a: number) => number;
  readonly __wbg_mtseedboottimingsearchresult_free: (a: number, b: number) => void;
  readonly mtseedboottimingsearchresult_mt_seed_hex: (a: number, b: number) => void;
  readonly mtseedboottimingsearchresult_lcg_seed_hex: (a: number, b: number) => void;
  readonly mtseedboottimingsearchresult_day: (a: number) => number;
  readonly mtseedboottimingsearchresult_minute: (a: number) => number;
  readonly mtseedboottimingsearchresult_timer0: (a: number) => number;
  readonly mtseedboottimingsearchresult_key_code: (a: number) => number;
  readonly __wbg_mtseedboottimingsearchresults_free: (a: number, b: number) => void;
  readonly mtseedboottimingsearchresults_length: (a: number) => number;
  readonly mtseedboottimingsearchresults_to_array: (a: number) => number;
  readonly mtseedboottimingsearchresults_get: (a: number, b: number) => number;
  readonly __wbg_mtseedboottimingsearchiterator_free: (a: number, b: number) => void;
  readonly mtseedboottimingsearchiterator_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly mtseedboottimingsearchiterator_is_finished: (a: number) => number;
  readonly mtseedboottimingsearchiterator_processed_seconds: (a: number) => number;
  readonly mtseedboottimingsearchiterator_total_seconds: (a: number) => number;
  readonly mtseedboottimingsearchiterator_progress: (a: number) => number;
  readonly mtseedboottimingsearchiterator_next_batch: (a: number, b: number, c: number) => number;
  readonly mt_seed_search_segment: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly derive_iv_set_wasm: (a: number, b: number, c: number) => void;
  readonly encode_iv_code_wasm: (a: number, b: number) => number;
  readonly decode_iv_code_wasm: (a: number, b: number) => void;
  readonly __wbg_tidsidresult_free: (a: number, b: number) => void;
  readonly __wbg_get_tidsidresult_tid: (a: number) => number;
  readonly __wbg_set_tidsidresult_tid: (a: number, b: number) => void;
  readonly __wbg_get_tidsidresult_sid: (a: number) => number;
  readonly __wbg_set_tidsidresult_sid: (a: number, b: number) => void;
  readonly tidsidresult_get_tid: (a: number) => number;
  readonly tidsidresult_get_sid: (a: number) => number;
  readonly __wbg_extraresult_free: (a: number, b: number) => void;
  readonly __wbg_get_extraresult_advances: (a: number) => number;
  readonly __wbg_set_extraresult_advances: (a: number, b: number) => void;
  readonly __wbg_get_extraresult_success: (a: number) => number;
  readonly __wbg_set_extraresult_success: (a: number, b: number) => void;
  readonly __wbg_get_extraresult_value1: (a: number) => number;
  readonly __wbg_set_extraresult_value1: (a: number, b: number) => void;
  readonly __wbg_get_extraresult_value2: (a: number) => number;
  readonly __wbg_set_extraresult_value2: (a: number, b: number) => void;
  readonly __wbg_get_extraresult_value3: (a: number) => number;
  readonly __wbg_set_extraresult_value3: (a: number, b: number) => void;
  readonly extraresult_get_advances: (a: number) => number;
  readonly extraresult_get_success: (a: number) => number;
  readonly extraresult_get_value1: (a: number) => number;
  readonly extraresult_get_value2: (a: number) => number;
  readonly extraresult_get_value3: (a: number) => number;
  readonly offsetcalculator_new: (a: bigint) => number;
  readonly offsetcalculator_next_rand: (a: number) => number;
  readonly offsetcalculator_consume_random: (a: number, b: number) => void;
  readonly offsetcalculator_reset: (a: number, b: bigint) => void;
  readonly offsetcalculator_calculate_tid_sid: (a: number) => number;
  readonly offsetcalculator_determine_front_residents: (a: number) => void;
  readonly offsetcalculator_determine_back_residents: (a: number) => void;
  readonly offsetcalculator_determine_all_residents: (a: number) => void;
  readonly offsetcalculator_probability_table_process: (a: number) => void;
  readonly offsetcalculator_probability_table_process_multiple: (a: number, b: number) => void;
  readonly offsetcalculator_extra_process: (a: number) => number;
  readonly offsetcalculator_execute_game_initialization: (a: number, b: number) => number;
  readonly calculate_game_offset: (a: bigint, b: number) => number;
  readonly calculate_tid_sid_from_seed: (a: bigint, b: number) => number;
  readonly __wbg_personalityrng_free: (a: number, b: number) => void;
  readonly personalityrng_new: (a: bigint) => number;
  readonly personalityrng_next: (a: number) => number;
  readonly personalityrng_next_u64: (a: number) => bigint;
  readonly personalityrng_advance: (a: number, b: number) => void;
  readonly personalityrng_reset: (a: number, b: bigint) => void;
  readonly personalityrng_get_index: (a: bigint) => bigint;
  readonly personalityrng_distance_between: (a: bigint, b: bigint) => bigint;
  readonly personalityrng_distance_from: (a: number, b: bigint) => bigint;
  readonly pidcalculator_generate_base_pid: (a: number) => number;
  readonly pidcalculator_apply_id_correction: (a: number, b: number, c: number) => number;
  readonly pidcalculator_generate_roamer_pid: (a: number, b: number, c: number) => number;
  readonly pidcalculator_generate_gift_pid: (a: number, b: number) => number;
  readonly pidcalculator_generate_egg_pid: (a: number, b: number) => number;
  readonly shinychecker_is_shiny: (a: number, b: number, c: number) => number;
  readonly shinychecker_get_shiny_value: (a: number, b: number, c: number) => number;
  readonly shinychecker_get_shiny_type: (a: number) => number;
  readonly shinychecker_check_shiny_type: (a: number, b: number, c: number) => number;
  readonly shinychecker_shiny_probability: () => number;
  readonly shinychecker_shiny_probability_with_charm: (a: number) => number;
  readonly __wbg_rawpokemondata_free: (a: number, b: number) => void;
  readonly __wbg_enumeratedpokemondata_free: (a: number, b: number) => void;
  readonly enumeratedpokemondata_get_seed: (a: number) => bigint;
  readonly enumeratedpokemondata_get_nature: (a: number) => number;
  readonly enumeratedpokemondata_get_sync_applied: (a: number) => number;
  readonly enumeratedpokemondata_get_ability_slot: (a: number) => number;
  readonly enumeratedpokemondata_get_gender_value: (a: number) => number;
  readonly enumeratedpokemondata_get_encounter_slot_value: (a: number) => number;
  readonly enumeratedpokemondata_get_encounter_type: (a: number) => number;
  readonly enumeratedpokemondata_get_level_rand_value: (a: number) => bigint;
  readonly enumeratedpokemondata_get_shiny_type: (a: number) => number;
  readonly enumeratedpokemondata_into_raw: (a: number) => number;
  readonly rawpokemondata_get_nature: (a: number) => number;
  readonly rawpokemondata_get_ability_slot: (a: number) => number;
  readonly rawpokemondata_get_gender_value: (a: number) => number;
  readonly rawpokemondata_get_encounter_slot_value: (a: number) => number;
  readonly rawpokemondata_get_level_rand_value: (a: number) => bigint;
  readonly rawpokemondata_get_shiny_type: (a: number) => number;
  readonly rawpokemondata_get_sync_applied: (a: number) => number;
  readonly rawpokemondata_get_encounter_type: (a: number) => number;
  readonly __wbg_bwgenerationconfig_free: (a: number, b: number) => void;
  readonly bwgenerationconfig_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly bwgenerationconfig_get_version: (a: number) => number;
  readonly bwgenerationconfig_get_encounter_type: (a: number) => number;
  readonly bwgenerationconfig_get_tid: (a: number) => number;
  readonly bwgenerationconfig_get_sid: (a: number) => number;
  readonly bwgenerationconfig_get_sync_enabled: (a: number) => number;
  readonly bwgenerationconfig_get_sync_nature_id: (a: number) => number;
  readonly bwgenerationconfig_get_is_shiny_locked: (a: number) => number;
  readonly bwgenerationconfig_get_has_shiny_charm: (a: number) => number;
  readonly pokemongenerator_generate_single_pokemon_bw: (a: bigint, b: number) => number;
  readonly pokemongenerator_calculate_generation_seed: (a: bigint, b: bigint) => bigint;
  readonly pokemongenerator_generate_pokemon_batch_bw: (a: number, b: bigint, c: bigint, d: number, e: number) => void;
  readonly seedenumerator_new: (a: bigint, b: bigint, c: number, d: number, e: number) => number;
  readonly seedenumerator_next_pokemon: (a: number) => number;
  readonly __wbg_dsconfigjs_free: (a: number, b: number) => void;
  readonly dsconfigjs_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly dsconfigjs_mac: (a: number, b: number) => void;
  readonly dsconfigjs_nazo: (a: number, b: number) => void;
  readonly dsconfigjs_hardware: (a: number, b: number) => void;
  readonly segmentparamsjs_new: (a: number, b: number, c: number) => number;
  readonly __wbg_timerangeparamsjs_free: (a: number, b: number) => void;
  readonly timerangeparamsjs_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly searchrangeparamsjs_new: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly sha1_hash_batch: (a: number, b: number, c: number) => void;
  readonly endianutils_swap_bytes_16: (a: number) => number;
  readonly endianutils_swap_bytes_64: (a: bigint) => bigint;
  readonly endianutils_be32_to_le: (a: number) => number;
  readonly endianutils_le32_to_be: (a: number) => number;
  readonly bitutils_rotate_left_32: (a: number, b: number) => number;
  readonly bitutils_rotate_right_32: (a: number, b: number) => number;
  readonly bitutils_get_bit: (a: number, b: number) => number;
  readonly bitutils_set_bit: (a: number, b: number, c: number) => number;
  readonly bitutils_count_bits: (a: number) => number;
  readonly bitutils_extract_bits: (a: number, b: number, c: number) => number;
  readonly numberutils_hex_string_to_u32: (a: number, b: number) => number;
  readonly numberutils_u32_to_hex_string: (a: number, b: number, c: number) => void;
  readonly numberutils_encode_bcd: (a: number) => number;
  readonly numberutils_decode_bcd: (a: number) => number;
  readonly numberutils_percentage_to_threshold: (a: number) => number;
  readonly numberutils_threshold_to_percentage: (a: number) => number;
  readonly __wbg_arrayutils_free: (a: number, b: number) => void;
  readonly arrayutils_sum_u32_array: (a: number, b: number) => bigint;
  readonly arrayutils_average_u32_array: (a: number, b: number) => number;
  readonly arrayutils_max_u32_array: (a: number, b: number) => number;
  readonly arrayutils_min_u32_array: (a: number, b: number) => number;
  readonly arrayutils_deduplicate_u32_array: (a: number, b: number, c: number) => void;
  readonly validationutils_is_valid_sid: (a: number) => number;
  readonly validationutils_is_valid_nature: (a: number) => number;
  readonly validationutils_is_valid_ability_slot: (a: number) => number;
  readonly validationutils_is_valid_hex_string: (a: number, b: number) => number;
  readonly validationutils_is_valid_seed: (a: bigint) => number;
  readonly pidcalculator_new: () => number;
  readonly shinychecker_new: () => number;
  readonly pokemongenerator_new: () => number;
  readonly __wbg_set_tidsidresult_advances_used: (a: number, b: number) => void;
  readonly validationutils_is_valid_tid: (a: number) => number;
  readonly personalityrng_set_seed: (a: number, b: bigint) => void;
  readonly pidcalculator_generate_event_pid: (a: number) => number;
  readonly pidcalculator_generate_wild_pid: (a: number, b: number, c: number) => number;
  readonly pidcalculator_generate_static_pid: (a: number, b: number, c: number) => number;
  readonly __wbg_get_tidsidresult_advances_used: (a: number) => number;
  readonly mtseedboottimingsearchresult_month: (a: number) => number;
  readonly mtseedboottimingsearchresult_hour: (a: number) => number;
  readonly mtseedboottimingsearchresult_second: (a: number) => number;
  readonly mtseedboottimingsearchresult_vcount: (a: number) => number;
  readonly mtseedboottimingsearchresults_processed_in_chunk: (a: number) => number;
  readonly tidsidresult_get_advances_used: (a: number) => number;
  readonly mtseedboottimingsearchresult_mt_seed: (a: number) => number;
  readonly mtseedboottimingsearchresult_lcg_seed_high: (a: number) => number;
  readonly mtseedboottimingsearchresult_lcg_seed_low: (a: number) => number;
  readonly mtseedboottimingsearchresult_year: (a: number) => number;
  readonly offsetcalculator_get_advances: (a: number) => number;
  readonly offsetcalculator_get_current_seed: (a: number) => bigint;
  readonly personalityrng_current_seed: (a: number) => bigint;
  readonly enumeratedpokemondata_get_advance: (a: number) => bigint;
  readonly enumeratedpokemondata_get_pid: (a: number) => number;
  readonly rawpokemondata_get_seed: (a: number) => bigint;
  readonly rawpokemondata_get_pid: (a: number) => number;
  readonly seedenumerator_remaining: (a: number) => number;
  readonly segmentparamsjs_timer0: (a: number) => number;
  readonly segmentparamsjs_vcount: (a: number) => number;
  readonly segmentparamsjs_key_code: (a: number) => number;
  readonly timerangeparamsjs_hour_start: (a: number) => number;
  readonly timerangeparamsjs_hour_end: (a: number) => number;
  readonly timerangeparamsjs_minute_start: (a: number) => number;
  readonly timerangeparamsjs_minute_end: (a: number) => number;
  readonly timerangeparamsjs_second_start: (a: number) => number;
  readonly timerangeparamsjs_second_end: (a: number) => number;
  readonly searchrangeparamsjs_start_year: (a: number) => number;
  readonly searchrangeparamsjs_start_month: (a: number) => number;
  readonly searchrangeparamsjs_start_day: (a: number) => number;
  readonly searchrangeparamsjs_range_seconds: (a: number) => number;
  readonly __wbg_set_statrange_max: (a: number, b: number) => void;
  readonly __wbg_get_statrange_max: (a: number) => number;
  readonly endianutils_swap_bytes_32: (a: number) => number;
  readonly __wbg_statrange_free: (a: number, b: number) => void;
  readonly __wbg_genderratio_free: (a: number, b: number) => void;
  readonly __wbg_trainerids_free: (a: number, b: number) => void;
  readonly __wbg_individualfilterjs_free: (a: number, b: number) => void;
  readonly __wbg_pidcalculator_free: (a: number, b: number) => void;
  readonly __wbg_shinychecker_free: (a: number, b: number) => void;
  readonly __wbg_parentsivsjs_free: (a: number, b: number) => void;
  readonly __wbg_pokemongenerator_free: (a: number, b: number) => void;
  readonly __wbg_seedenumerator_free: (a: number, b: number) => void;
  readonly __wbg_segmentparamsjs_free: (a: number, b: number) => void;
  readonly __wbg_searchrangeparamsjs_free: (a: number, b: number) => void;
  readonly __wbg_endianutils_free: (a: number, b: number) => void;
  readonly __wbg_encountercalculator_free: (a: number, b: number) => void;
  readonly __wbg_numberutils_free: (a: number, b: number) => void;
  readonly __wbg_bitutils_free: (a: number, b: number) => void;
  readonly __wbg_validationutils_free: (a: number, b: number) => void;
  readonly __wbg_offsetcalculator_free: (a: number, b: number) => void;
  readonly __wbindgen_export: (a: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export3: (a: number, b: number) => number;
  readonly __wbindgen_export4: (a: number, b: number, c: number, d: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
