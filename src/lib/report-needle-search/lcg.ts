/**
 * LCG演算モジュール
 * ポケモンBW/BW2のレポート針方向計算に使用
 */

/**
 * LCG乗算定数
 */
export const LCG_MULTIPLIER = 0x5D588B656C078965n;

/**
 * LCG加算定数
 */
export const LCG_INCREMENT = 0x269EC3n;

/**
 * 64bitマスク
 */
export const MASK_64 = (1n << 64n) - 1n;

/**
 * LCGの次のシード値を計算する
 * next = (seed * LCG_MULTIPLIER + LCG_INCREMENT) & MASK_64
 */
export function nextSeed(seed: bigint): bigint {
  return ((seed * LCG_MULTIPLIER) + LCG_INCREMENT) & MASK_64;
}

/**
 * LCGの前のシード値を計算する
 * 逆演算: prev = (seed - LCG_INCREMENT) * LCG_MULTIPLIER_INV
 */
export const LCG_MULTIPLIER_INV = 0xDEDCEDAE9638806Dn;

export function prevSeed(seed: bigint): bigint {
  return ((seed - LCG_INCREMENT) * LCG_MULTIPLIER_INV) & MASK_64;
}

/**
 * LCGシードから針の方向(0-7)を計算する
 * Rustの PersonalityRNG::calc_report_needle_direction と同じ式
 * dir = ((next >> 32) * 8) >> 32
 */
export function calcNeedleDir(seed: bigint): number {
  const next = nextSeed(seed);
  const upper32 = next >> 32n;
  const dir = (upper32 * 8n) >> 32n;
  return Number(dir);
}

/**
 * LCGをn回進める
 */
export function advanceSeed(seed: bigint, n: number): bigint {
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = nextSeed(s);
  }
  return s;
}

/**
 * 16進文字列をBigIntにパース
 * 0xプレフィックスは任意
 */
export function parseSeedHex(hexStr: string): bigint | null {
  const cleaned = hexStr.replace(/^0x/i, '').trim();
  if (!/^[0-9a-fA-F]+$/.test(cleaned) || cleaned.length === 0) {
    return null;
  }
  try {
    const value = BigInt(`0x${cleaned}`);
    return value & MASK_64;
  } catch {
    return null;
  }
}

/**
 * BigIntを16進文字列に変換（16桁ゼロパディング）
 */
export function seedToHex(seed: bigint): string {
  return (seed & MASK_64).toString(16).toUpperCase().padStart(16, '0');
}
