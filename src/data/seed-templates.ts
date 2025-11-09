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
    name: 'BW 固定 6V',
    description: 'ブラック・ホワイトの固定シンボル6V個体',
    seeds: [
      0xC6FA6486,
      0x3E4E941E,
      0x8B1F7A94,
      0xD3072C7D,
    ],
  },
  {
    name: 'BW2 野生 5VS0',
    description: 'ブラック2・ホワイト2の野生5VS0個体',
    seeds: [
      0x5A7E1B2C,
      0x2F3D4E5A,
      0x9C8D7E6F,
      0x1A2B3C4D,
    ],
  },
  {
    name: 'BW 伝説 高個体',
    description: 'ブラック・ホワイトの伝説ポケモン高個体値',
    seeds: [
      0x7F8E9D0A,
      0x4B5C6D7E,
      0xA1B2C3D4,
      0xE5F60718,
    ],
  },
  {
    name: 'BW2 色違い 5V',
    description: 'ブラック2・ホワイト2の色違い5V個体',
    seeds: [
      0x6C7D8E9F,
      0x3A4B5C6D,
      0xF1E2D3C4,
      0x8796A5B4,
    ],
  },
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
