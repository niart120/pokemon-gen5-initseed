/**
 * 既知のMT初期Seedテンプレート定義
 * ポケモンBW/BW2で特定条件で出現する固定Seed値のリスト
 */

export interface SeedTemplate {
  /** テンプレート表示名 */
  name: string;
  /** 対応するMT初期Seedのリスト (32bit整数) */
  seeds: number[];
  /** テンプレートの説明 (オプション) */
  description?: string;
}

/**
 * 定義済みのSeedテンプレート
 */
export const SEED_TEMPLATES: SeedTemplate[] = [
  {
    name: 'BW 固定・野生 6V',
    description: 'ブラック・ホワイト 標準（消費0） 6V（5種類）',
    seeds: [
      0x14B11BA6,
      0x8A30480D,
      0x9E02B0AE,
      0xADFA2178,
      0xFC4AA3AC,
    ],
  },
  {
    name: 'BW 固定・野生 5VA0',
    description: 'ブラック・ホワイト 標準（消費0） 5VA0（3種類）',
    seeds: [
      0x4BD26FC3,
      0xC59A441A,
      0xDFE7EBF2,
    ],
  },
  {
    name: 'BW 固定・野生 V0VVV0',
    description: 'ブラック・ホワイト 標準（消費0） V0VVV0（2種類）',
    seeds: [
      0x0B5A81F0,
      0x5D6F6D1D,
    ],
  },
  {
    name: 'BW 固定・野生 V2UVVV めざ氷',
    description: 'ブラック・ホワイト 標準（消費0） V2UVVV（めざ氷 7種類）',
    seeds: [
      0x01117891,
      0x2277228B,
      0xA38FBAAF,
      0xA49FDC53,
      0xAF3FFBBF,
      0xF0EE8F20,
      0xF62667EE,
    ],
  },

  {
    name: 'BW 徘徊 6V',
    description: 'ブラック・ホワイト 徘徊（消費なし） 6V（5種類）',
    seeds: [
      0x35652A5F,
      0x4707F449,
      0x7541AAD0,
      0xBEE598A7,
      0xEAA27A05,
    ],
  },
  {
    name: 'BW 徘徊 V2UVVV めざ氷',
    description: 'ブラック・ホワイト 徘徊（消費なし） V2UVVV（めざ氷 6種類）',
    seeds: [
      0x5F3DE7EF,
      0x7F1983D4,
      0xB8500799,
      0xC18AA384,
      0xC899E66E,
      0xD8BFC637,
    ],
  },
  {
    name: 'BW 徘徊 U2UUUV めざ飛',
    description: 'ブラック・ホワイト 徘徊（消費なし） U2UUUV（めざ飛 5種類）',
    seeds: [
      0x4A28CBE0,
      0x5B41C530,
      0xA359C930,
      0xC8175B8B,
      0xDAFA8540,
    ],
  },

  {
    name: 'BW2 固定・野生 6V',
    description: 'ブラック2・ホワイト2（消費2） 6V（6種類）',
    seeds: [
      0x31C26DE4,
      0x519A0C07,
      0xC28A882E,
      0xDFE7EBF2,
      0xE34372AE,
      0xED01C9C2,
    ],
  },
  {
    name: 'BW2 固定・野生 5VA0',
    description: 'ブラック2・ホワイト2（消費2） 5VA0（10種類）',
    seeds: [
      0x14719922,
      0x634CC2B0,
      0x71AFC896,
      0x88EFDEC2,
      0xAA333835,
      0xABD93E44,
      0xADD877C4,
      0xB32B6B02,
      0xC31DDEF7,
      0xD286653C,
    ],
  },
  {
    name: 'BW2 固定・野生 V0VVV0',
    description: 'ブラック2・ホワイト2（消費2） V0VVV0（4種類）',
    seeds: [
      0x54F39E0F,
      0x6338DDED,
      0x7BF8CD77,
      0xF9C432EB,
    ],
  },
  {
    name: 'BW2 固定・野生 V2UVVV めざ氷',
    description: 'ブラック2・ホワイト2（消費2） V2UVVV（めざ氷 8種類）',
    seeds: [
      0x03730F34,
      0x2C9D32BF,
      0x3F37A9B9,
      0x440CB317,
      0x6728FDBF,
      0x7240A4AE,
      0x9BFB3D33,
      0xFF1DF7DC,
    ],
  },

  // 既存のテストサンプルはそのまま残す
  {
    name: 'テストサンプル',
    description: 'デモ・テスト用のサンプルSeed',
    seeds: [
      0x12345678,
      0x9ABCDEF0,
      0x11111111,
      0xFFFFFFFF,
    ],
  },
];
