/**
 * UnresolvedPokemonData - WASM出力を取り込んだ直後の未解決データ型
 * wasm-bindgen 生成物からアダプトした、シリアラブルなプレーンデータ
 */
export interface UnresolvedPokemonData {
  /** 初期シード値 */
  seed: bigint;
  /** PID (32bit) */
  pid: number;
  /** 性格値 (0-24) */
  nature: number;
  /** シンクロ適用フラグ */
  sync_applied: boolean;
  /** 特性スロット (0-1) */
  ability_slot: number;
  /** 性別値 (0-255) */
  gender_value: number;
  /** 遭遇スロット値 */
  encounter_slot_value: number;
  /** エンカウントタイプ (0-7: 野生, 10-13: 固定, 20: 徘徊) */
  encounter_type: number;
  /** レベル乱数値 */
  level_rand_value: bigint;
  /** 色違いタイプ (0: NotShiny, 1: Square, 2: Star) */
  shiny_type: number;
}

// Domain 列挙体の再エクスポートは行わない（利用側は '@/types/domain' を直接参照）

/**
 * 性別比設定
 */
export interface GenderRatio {
  /** 雌判定の閾値 (0-256) - gender_value < threshold なら雌。固定雌は 256 を用いる */
  threshold: number;
  /** 性別不明かどうか */
  genderless: boolean;
}
