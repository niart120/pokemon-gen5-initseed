/* tslint:disable */
/* eslint-disable */
/**
 * オフセット計算統合API（仕様書準拠）
 */
export function calculate_game_offset(initial_seed: bigint, mode: GameMode): number;
/**
 * TID/SID決定処理統合API（仕様書準拠）
 */
export function calculate_tid_sid_from_seed(initial_seed: bigint, mode: GameMode): TidSidResult;
/**
 * WebAssembly向けバッチSHA-1計算エントリポイント
 * `messages` は 16 ワード単位（512bit）で並ぶフラットな配列である必要がある
 */
export function sha1_hash_batch(messages: Uint32Array): Uint32Array;
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
/**
 * 遭遇タイプ列挙型
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
  Roaming = 20,
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
/**
 * 配列操作ユーティリティ
 */
export class ArrayUtils {
  private constructor();
  free(): void;
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
/**
 * BW/BW2準拠設定構造体
 */
export class BWGenerationConfig {
  free(): void;
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
/**
 * ビット操作ユーティリティ
 */
export class BitUtils {
  private constructor();
  free(): void;
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
/**
 * 遭遇計算エンジン
 */
export class EncounterCalculator {
  free(): void;
  /**
   * 新しいEncounterCalculatorインスタンスを作成
   */
  constructor();
  /**
   * 遭遇スロットを計算
   *
   * # Arguments
   * * `version` - ゲームバージョン
   * * `encounter_type` - 遭遇タイプ
   * * `random_value` - 乱数値（32bit）
   *
   * # Returns
   * 遭遇スロット番号（0-11）
   */
  static calculate_encounter_slot(version: GameVersion, encounter_type: EncounterType, random_value: number): number;
  /**
   * スロット番号をテーブルインデックスに変換
   *
   * # Arguments
   * * `encounter_type` - 遭遇タイプ
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
 * エンディアン変換ユーティリティ
 */
export class EndianUtils {
  private constructor();
  free(): void;
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
/**
 * Extra処理結果（BW2専用）
 */
export class ExtraResult {
  private constructor();
  free(): void;
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
 * 統合シード探索器
 * 固定パラメータを事前計算し、日時範囲を高速探索する
 */
export class IntegratedSeedSearcher {
  free(): void;
  /**
   * コンストラクタ: 固定パラメータの事前計算
   */
  constructor(mac: Uint8Array, nazo: Uint32Array, hardware: string, key_input: number, frame: number);
  /**
   * 統合シード探索メイン関数
   * 日時範囲とTimer0/VCount範囲を指定して一括探索
   */
  search_seeds_integrated(year_start: number, month_start: number, date_start: number, hour_start: number, minute_start: number, second_start: number, range_seconds: number, timer0_min: number, timer0_max: number, vcount_min: number, vcount_max: number, target_seeds: Uint32Array): Array<any>;
  /**
   * 統合シード探索SIMD版
   * range_secondsを最内ループに配置してSIMD SHA-1計算を活用
   */
  search_seeds_integrated_simd(year_start: number, month_start: number, date_start: number, hour_start: number, minute_start: number, second_start: number, range_seconds: number, timer0_min: number, timer0_max: number, vcount_min: number, vcount_max: number, target_seeds: Uint32Array): Array<any>;
}
/**
 * 数値変換ユーティリティ
 */
export class NumberUtils {
  private constructor();
  free(): void;
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
/**
 * オフセット計算エンジン
 */
export class OffsetCalculator {
  free(): void;
  /**
   * 新しいOffsetCalculatorインスタンスを作成
   *
   * # Arguments
   * * `seed` - 初期シード値
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
   * * `new_seed` - 新しいシード値
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
   * 現在のシード値を取得
   *
   * # Returns
   * 現在のシード値
   */
  readonly get_current_seed: bigint;
}
/**
 * PID計算エンジン
 */
export class PIDCalculator {
  free(): void;
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
  static generate_roaming_pid(r1: number, tid: number, sid: number): number;
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
/**
 * PersonalityRNG構造体
 * BW仕様64bit線形合同法: S[n+1] = S[n] * 0x5D588B656C078965 + 0x269EC3
 */
export class PersonalityRNG {
  free(): void;
  /**
   * 新しいPersonalityRNGインスタンスを作成
   *
   * # Arguments
   * * `seed` - 初期シード値（64bit）
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
   * シードをリセット
   *
   * # Arguments
   * * `initial_seed` - リセット後のシード値
   */
  reset(initial_seed: bigint): void;
  /**
   * 0x0からの進行度を計算
   *
   * # Arguments
   * * `seed` - 計算対象のシード値
   *
   * # Returns
   * 0x0からの進行度
   */
  static get_index(seed: bigint): bigint;
  /**
   * 2つのシード間の距離を計算
   *
   * # Arguments
   * * `from_seed` - 開始シード
   * * `to_seed` - 終了シード
   *
   * # Returns
   * from_seedからto_seedまでの距離
   */
  static distance_between(from_seed: bigint, to_seed: bigint): bigint;
  /**
   * 指定シードから現在のシードまでの距離
   *
   * # Arguments
   * * `source_seed` - 開始シード
   *
   * # Returns
   * source_seedから現在のシードまでの距離
   */
  distance_from(source_seed: bigint): bigint;
  /**
   * 現在のシード値を取得
   *
   * # Returns
   * 現在の内部シード値
   */
  readonly current_seed: bigint;
  /**
   * シード値を設定
   *
   * # Arguments
   * * `new_seed` - 新しいシード値
   */
  set seed(value: bigint);
}
/**
 * ポケモン生成エンジン
 */
export class PokemonGenerator {
  free(): void;
  /**
   * 新しいPokemonGeneratorインスタンスを作成
   */
  constructor();
  /**
   * BW/BW2準拠 単体ポケモン生成（統括関数）
   *
   * # Arguments
   * * `seed` - 初期シード値
   * * `config` - BW準拠設定
   *
   * # Returns
   * 生成されたポケモンデータ
   */
  static generate_single_pokemon_bw(seed: bigint, config: BWGenerationConfig): RawPokemonData;
  /**
   * オフセット適用後の生成開始シードを計算
   */
  static calculate_generation_seed(initial_seed: bigint, offset: bigint): bigint;
  /**
   * BW/BW2準拠 バッチ生成（offsetのみ）
   *
   * # Arguments
   * * `base_seed` - 列挙の基準シード（初期シード）
   * * `offset` - 最初の生成までの前進数（ゲーム内不定消費を含めた開始位置）
   * * `count` - 生成数（0なら空）
   * * `config` - BW準拠設定
   *
   * # Returns
   * 生成されたポケモンデータの配列
   */
  static generate_pokemon_batch_bw(base_seed: bigint, offset: bigint, count: number, config: BWGenerationConfig): RawPokemonData[];
}
/**
 * 生ポケモンデータ構造体
 */
export class RawPokemonData {
  private constructor();
  free(): void;
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
/**
 * 探索結果構造体
 */
export class SearchResult {
  free(): void;
  constructor(seed: number, hash: string, year: number, month: number, date: number, hour: number, minute: number, second: number, timer0: number, vcount: number);
  readonly seed: number;
  readonly hash: string;
  readonly year: number;
  readonly month: number;
  readonly date: number;
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
  readonly timer0: number;
  readonly vcount: number;
}
/**
 * 連続列挙用のシード列挙器（offsetのみ）
 */
export class SeedEnumerator {
  free(): void;
  /**
   * 列挙器を作成
   */
  constructor(base_seed: bigint, offset: bigint, count: number, config: BWGenerationConfig);
  /**
   * 次のポケモンを生成（残数0なら undefined を返す）
   */
  next_pokemon(): RawPokemonData | undefined;
  /**
   * 残数を取得
   */
  readonly remaining: number;
}
/**
 * 色違い判定エンジン
 */
export class ShinyChecker {
  free(): void;
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
 * TID/SID決定結果
 */
export class TidSidResult {
  private constructor();
  free(): void;
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
/**
 * バリデーションユーティリティ
 */
export class ValidationUtils {
  private constructor();
  free(): void;
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
   * シード値の妥当性チェック
   *
   * # Arguments
   * * `seed` - シード値
   *
   * # Returns
   * 妥当性
   */
  static is_valid_seed(seed: bigint): boolean;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly encountercalculator_new: () => number;
  readonly encountercalculator_calculate_encounter_slot: (a: number, b: number, c: number) => number;
  readonly encountercalculator_slot_to_table_index: (a: number, b: number) => number;
  readonly encountercalculator_get_dust_cloud_content: (a: number) => number;
  readonly __wbg_searchresult_free: (a: number, b: number) => void;
  readonly searchresult_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => number;
  readonly searchresult_hash: (a: number, b: number) => void;
  readonly searchresult_year: (a: number) => number;
  readonly searchresult_month: (a: number) => number;
  readonly searchresult_date: (a: number) => number;
  readonly searchresult_hour: (a: number) => number;
  readonly searchresult_minute: (a: number) => number;
  readonly searchresult_second: (a: number) => number;
  readonly searchresult_timer0: (a: number) => number;
  readonly searchresult_vcount: (a: number) => number;
  readonly __wbg_integratedseedsearcher_free: (a: number, b: number) => void;
  readonly integratedseedsearcher_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly integratedseedsearcher_search_seeds_integrated: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => number;
  readonly integratedseedsearcher_search_seeds_integrated_simd: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => number;
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
  readonly __wbg_offsetcalculator_free: (a: number, b: number) => void;
  readonly offsetcalculator_new: (a: bigint) => number;
  readonly offsetcalculator_next_rand: (a: number) => number;
  readonly offsetcalculator_consume_random: (a: number, b: number) => void;
  readonly offsetcalculator_get_advances: (a: number) => number;
  readonly offsetcalculator_get_current_seed: (a: number) => bigint;
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
  readonly pidcalculator_generate_roaming_pid: (a: number, b: number, c: number) => number;
  readonly pidcalculator_generate_gift_pid: (a: number, b: number) => number;
  readonly pidcalculator_generate_egg_pid: (a: number, b: number) => number;
  readonly shinychecker_is_shiny: (a: number, b: number, c: number) => number;
  readonly shinychecker_get_shiny_value: (a: number, b: number, c: number) => number;
  readonly shinychecker_get_shiny_type: (a: number) => number;
  readonly shinychecker_check_shiny_type: (a: number, b: number, c: number) => number;
  readonly shinychecker_shiny_probability: () => number;
  readonly shinychecker_shiny_probability_with_charm: (a: number) => number;
  readonly __wbg_rawpokemondata_free: (a: number, b: number) => void;
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
  readonly seedenumerator_new: (a: bigint, b: bigint, c: number, d: number) => number;
  readonly seedenumerator_next_pokemon: (a: number) => number;
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
  readonly tidsidresult_get_advances_used: (a: number) => number;
  readonly searchresult_seed: (a: number) => number;
  readonly personalityrng_current_seed: (a: number) => bigint;
  readonly rawpokemondata_get_seed: (a: number) => bigint;
  readonly rawpokemondata_get_pid: (a: number) => number;
  readonly seedenumerator_remaining: (a: number) => number;
  readonly endianutils_swap_bytes_32: (a: number) => number;
  readonly __wbg_pidcalculator_free: (a: number, b: number) => void;
  readonly __wbg_shinychecker_free: (a: number, b: number) => void;
  readonly __wbg_pokemongenerator_free: (a: number, b: number) => void;
  readonly __wbg_seedenumerator_free: (a: number, b: number) => void;
  readonly __wbg_endianutils_free: (a: number, b: number) => void;
  readonly __wbg_encountercalculator_free: (a: number, b: number) => void;
  readonly __wbg_numberutils_free: (a: number, b: number) => void;
  readonly __wbg_bitutils_free: (a: number, b: number) => void;
  readonly __wbg_validationutils_free: (a: number, b: number) => void;
  readonly __wbindgen_export_0: (a: number, b: number) => number;
  readonly __wbindgen_export_1: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export_2: (a: number, b: number, c: number) => void;
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
