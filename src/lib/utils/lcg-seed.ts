/**
 * LCG Seed計算ユーティリティ
 * SHA-1ハッシュ結果からLCG Seedを計算する
 */

/**
 * SHA-1ハッシュからLCG Seedを計算
 * @param h0 SHA-1ハッシュの第1ワード
 * @param h1 SHA-1ハッシュの第2ワード
 * @returns 64bit LCG Seed値
 */
export function calculateLcgSeed(h0: number, h1: number): bigint {
  // h0, h1をリトルエンディアンに変換
  const h0Le = swapBytes32(h0);
  const h1Le = swapBytes32(h1);

  // 64bit値を構築
  const lcgSeed = (BigInt(h1Le) << 32n) | BigInt(h0Le);

  return lcgSeed;
}

/**
 * 32bit値のバイトスワップ
 * @param value 変換する32bit値
 * @returns バイトスワップされた値
 */
function swapBytes32(value: number): number {
  return (
    ((value & 0xff) << 24) |
    (((value >> 8) & 0xff) << 16) |
    (((value >> 16) & 0xff) << 8) |
    ((value >> 24) & 0xff)
  ) >>> 0;
}

/**
 * LCG SeedをMT Seedに変換
 * @param lcgSeed 64bit LCG Seed値
 * @returns 32bit MT Seed値
 */
export function lcgSeedToMtSeed(lcgSeed: bigint): number {
  // LCG演算
  const multiplier = 0x5D588B656C078965n;
  const addValue = 0x269EC3n;
  
  const result = lcgSeed * multiplier + addValue;
  
  // 上位32bitを取得
  return Number((result >> 32n) & 0xFFFFFFFFn);
}

/**
 * LCG Seedを16進数文字列に変換
 * @param lcgSeed 64bit LCG Seed値
 * @returns 16進数文字列 (例: "0x123456789ABCDEF0")
 */
export function lcgSeedToHex(lcgSeed: bigint): string {
  return '0x' + lcgSeed.toString(16).toUpperCase().padStart(16, '0');
}
