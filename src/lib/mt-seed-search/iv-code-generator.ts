/**
 * IVコード変換ロジック
 * 検索条件からIVコードリストを生成する
 */

import type { IvSet } from '@/types/egg';
import type {
  IvCode,
  IvSearchFilter,
  IvCodeGenerationResult,
  StatRange,
} from '@/types/mt-seed-search';
import { encodeIvCode, MAX_IV_CODES } from '@/types/mt-seed-search';

// === めざめるパワー計算 ===

/**
 * めざめるパワータイプ名
 */
export const HIDDEN_POWER_TYPE_NAMES = [
  'かくとう', // 0
  'ひこう', // 1
  'どく', // 2
  'じめん', // 3
  'いわ', // 4
  'むし', // 5
  'ゴースト', // 6
  'はがね', // 7
  'ほのお', // 8
  'みず', // 9
  'くさ', // 10
  'でんき', // 11
  'エスパー', // 12
  'こおり', // 13
  'ドラゴン', // 14
  'あく', // 15
] as const;

/**
 * めざめるパワー情報
 */
export interface HiddenPowerResult {
  type: number; // 0-15
  power: number; // 30-70
}

/**
 * IVセットからめざめるパワーを計算
 * BW/BW2のめざパ計算式に準拠
 */
export function calculateHiddenPower(ivs: IvSet): HiddenPowerResult {
  // タイプ計算: 各IVの最下位ビットを使用
  // HP, Atk, Def, Spe, SpA, SpD の順で bit0-5
  let typeBits = 0;
  typeBits |= (ivs[0] & 1) << 0; // HP
  typeBits |= (ivs[1] & 1) << 1; // Atk
  typeBits |= (ivs[2] & 1) << 2; // Def
  typeBits |= (ivs[5] & 1) << 3; // Spe
  typeBits |= (ivs[3] & 1) << 4; // SpA
  typeBits |= (ivs[4] & 1) << 5; // SpD

  const type = Math.floor((typeBits * 15) / 63);

  // 威力計算: 各IVの bit1 を使用
  let powerBits = 0;
  powerBits |= ((ivs[0] >> 1) & 1) << 0; // HP
  powerBits |= ((ivs[1] >> 1) & 1) << 1; // Atk
  powerBits |= ((ivs[2] >> 1) & 1) << 2; // Def
  powerBits |= ((ivs[5] >> 1) & 1) << 3; // Spe
  powerBits |= ((ivs[3] >> 1) & 1) << 4; // SpA
  powerBits |= ((ivs[4] >> 1) & 1) << 5; // SpD

  const power = Math.floor((powerBits * 40) / 63) + 30;

  return { type, power };
}

// === IVコード生成 ===

/**
 * 徘徊ポケモン用IV並び替え
 * 通常: HABCDS (HP, Atk, Def, SpA, SpD, Spe)
 * 徘徊: HABDSC (HP, Atk, Def, Spe, SpA, SpD) → MT取得順に合わせる
 *
 * IVコードは HABCDS 順でエンコードされているため、
 * 徘徊検索時は SpD と Spe を入れ替えたコードを生成する必要がある
 */
export function reorderIvCodeForRoamer(ivCode: IvCode): IvCode {
  // 現在の配置: [HP:5][Atk:5][Def:5][SpA:5][SpD:5][Spe:5]
  const hp = (ivCode >> 25) & 0x1f;
  const atk = (ivCode >> 20) & 0x1f;
  const def = (ivCode >> 15) & 0x1f;
  const spa = (ivCode >> 10) & 0x1f;
  const spd = (ivCode >> 5) & 0x1f;
  const spe = ivCode & 0x1f;

  // 徘徊用配置: [HP:5][Atk:5][Def:5][Spe:5][SpA:5][SpD:5]
  // ※IVコードのエンコード順序は変更せず、検索対象の値を入れ替える
  // MTから取得される順: HP, Atk, Def, Spe, SpA, SpD
  // これを標準IVコード形式 (HABCDS) で表現:
  return (hp << 25) | (atk << 20) | (def << 15) | (spe << 10) | (spa << 5) | spd;
}

/**
 * めざパフィルターとのマッチング判定
 */
function matchesHiddenPowerFilter(
  ivs: IvSet,
  filter: IvSearchFilter
): boolean {
  if (
    filter.hiddenPowerType === undefined &&
    filter.hiddenPowerPower === undefined
  ) {
    return true;
  }

  const hp = calculateHiddenPower(ivs);

  if (filter.hiddenPowerType !== undefined && hp.type !== filter.hiddenPowerType) {
    return false;
  }
  if (filter.hiddenPowerPower !== undefined && hp.power !== filter.hiddenPowerPower) {
    return false;
  }

  return true;
}

/**
 * フィルター条件から組み合わせ数を見積もる
 */
function estimateTotalCombinations(filter: IvSearchFilter): number {
  let total = 1;
  for (const range of filter.ivRanges) {
    total *= range.max - range.min + 1;
  }
  return total;
}

/**
 * StatRangeのバリデーション
 */
function validateStatRange(range: StatRange, index: number): string | null {
  if (range.min < 0 || range.min > 31) {
    return `ivRanges[${index}].min must be 0-31`;
  }
  if (range.max < 0 || range.max > 31) {
    return `ivRanges[${index}].max must be 0-31`;
  }
  if (range.min > range.max) {
    return `ivRanges[${index}].min must be <= max`;
  }
  return null;
}

/**
 * IvSearchFilterのバリデーション
 */
export function validateIvSearchFilter(filter: IvSearchFilter): string[] {
  const errors: string[] = [];

  for (let i = 0; i < 6; i++) {
    const error = validateStatRange(filter.ivRanges[i], i);
    if (error) {
      errors.push(error);
    }
  }

  if (filter.hiddenPowerType !== undefined) {
    if (filter.hiddenPowerType < 0 || filter.hiddenPowerType > 15) {
      errors.push('hiddenPowerType must be 0-15');
    }
  }

  if (filter.hiddenPowerPower !== undefined) {
    if (filter.hiddenPowerPower < 30 || filter.hiddenPowerPower > 70) {
      errors.push('hiddenPowerPower must be 30-70');
    }
  }

  return errors;
}

/**
 * 検索条件からIVコードリストを生成
 *
 * @param filter - 検索フィルター
 * @param options - オプション
 * @param options.isRoamer - 徘徊ポケモンモード（IV順序を HABDSC に変換）
 * @returns 成功時はIVコード配列、失敗時はエラー情報
 */
export function generateIvCodes(
  filter: IvSearchFilter,
  options?: { isRoamer?: boolean }
): IvCodeGenerationResult {
  const candidates: IvCode[] = [];
  const isRoamer = options?.isRoamer ?? false;

  const [hpRange, atkRange, defRange, spaRange, spdRange, speRange] =
    filter.ivRanges;

  // 6重ループで全組み合わせを列挙
  for (let hp = hpRange.min; hp <= hpRange.max; hp++) {
    for (let atk = atkRange.min; atk <= atkRange.max; atk++) {
      for (let def = defRange.min; def <= defRange.max; def++) {
        for (let spa = spaRange.min; spa <= spaRange.max; spa++) {
          for (let spd = spdRange.min; spd <= spdRange.max; spd++) {
            for (let spe = speRange.min; spe <= speRange.max; spe++) {
              const ivs: IvSet = [hp, atk, def, spa, spd, spe];

              // めざパフィルター適用
              if (!matchesHiddenPowerFilter(ivs, filter)) {
                continue;
              }

              // IVコードを生成（徘徊モード時は順序変換）
              let ivCode = encodeIvCode(ivs);
              if (isRoamer) {
                ivCode = reorderIvCodeForRoamer(ivCode);
              }
              candidates.push(ivCode);

              // 上限チェック（早期終了）
              if (candidates.length > MAX_IV_CODES) {
                return {
                  success: false,
                  error: 'TOO_MANY_COMBINATIONS',
                  estimatedCount: estimateTotalCombinations(filter),
                };
              }
            }
          }
        }
      }
    }
  }

  return { success: true, ivCodes: candidates };
}

// === エラーメッセージ ===

/**
 * IVコード生成エラーメッセージ
 */
export function getIvCodeGenerationErrorMessage(
  result: IvCodeGenerationResult
): string | null {
  if (result.success) {
    return null;
  }

  switch (result.error) {
    case 'TOO_MANY_COMBINATIONS':
      return `検索条件が広すぎます。個体値の組み合わせが${result.estimatedCount.toLocaleString()}件あります（上限: ${MAX_IV_CODES}件）。条件を絞り込んでください。`;
    default:
      return '不明なエラーが発生しました';
  }
}

// === デフォルト値 ===

/**
 * デフォルトのStatRange（全範囲）
 */
export function createDefaultStatRange(): StatRange {
  return { min: 0, max: 31 };
}

/**
 * デフォルトのIvSearchFilter（全範囲、めざパ指定なし）
 */
export function createDefaultIvSearchFilter(): IvSearchFilter {
  return {
    ivRanges: [
      createDefaultStatRange(),
      createDefaultStatRange(),
      createDefaultStatRange(),
      createDefaultStatRange(),
      createDefaultStatRange(),
      createDefaultStatRange(),
    ],
  };
}

/**
 * 6V固定のIvSearchFilter
 */
export function create6VIvSearchFilter(): IvSearchFilter {
  return {
    ivRanges: [
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
      { min: 31, max: 31 },
    ],
  };
}

/**
 * 5V0A (A抜け) のIvSearchFilter
 */
export function create5V0AIvSearchFilter(): IvSearchFilter {
  return {
    ivRanges: [
      { min: 31, max: 31 }, // HP
      { min: 0, max: 0 }, // Atk
      { min: 31, max: 31 }, // Def
      { min: 31, max: 31 }, // SpA
      { min: 31, max: 31 }, // SpD
      { min: 31, max: 31 }, // Spe
    ],
  };
}

/**
 * 5V0S (S抜け) のIvSearchFilter（トリル用）
 */
export function create5V0SIvSearchFilter(): IvSearchFilter {
  return {
    ivRanges: [
      { min: 31, max: 31 }, // HP
      { min: 31, max: 31 }, // Atk
      { min: 31, max: 31 }, // Def
      { min: 31, max: 31 }, // SpA
      { min: 31, max: 31 }, // SpD
      { min: 0, max: 0 }, // Spe
    ],
  };
}
