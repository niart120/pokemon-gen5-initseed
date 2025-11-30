/**
 * ROM Parameters Helper Functions
 * 新しいタプル型ROMParametersデータ構造のアクセス関数
 */
import type { ROMParameters } from '@/types/rom';
import romParameters from '@/data/rom-parameters';
import type { ROMVersion, ROMRegion } from '@/types/rom';

/**
 * ROMParametersを取得（型安全）
 * @param version ROM version
 * @param region ROM region
 * @returns ROMParameters or null if not found
 */
export function getROMParameters(version: string, region: string): ROMParameters | null {
  // romParameters の型を構築
  type ROMParamsTree = Record<ROMVersion, Record<ROMRegion, ROMParameters>>;
  const tree = romParameters as unknown as ROMParamsTree;
  const versionKey = version as ROMVersion;
  const regionKey = region as ROMRegion;
  const versionData = tree[versionKey];
  if (!versionData) return null;

  const regionData = versionData[regionKey];
  if (!regionData) return null;

  return regionData;
}

/**
 * 指定されたVCOUNT値に対応するTimer0範囲を取得
 * @param version ROM version
 * @param region ROM region  
 * @param vcount VCOUNT value
 * @returns Timer0範囲、または該当なしの場合null
 */
export function getTimer0Range(version: string, region: string, vcount: number): 
  { min: number; max: number } | null {
  
  const params = getROMParameters(version, region);
  if (!params) return null;
  
  for (const [vcountValue, timer0Min, timer0Max] of params.vcountTimerRanges) {
    if (vcountValue === vcount) {
      return { min: timer0Min, max: timer0Max };
    }
  }
  
  return null;
}

/**
 * 指定されたROMで有効なVCOUNT値の一覧を取得
 * @param version ROM version
 * @param region ROM region
 * @returns 有効なVCOUNT値の配列
 */
export function getValidVCounts(version: string, region: string): number[] {
  const params = getROMParameters(version, region);
  if (!params) return [];
  
  return params.vcountTimerRanges.map(([vcount]) => vcount);
}

/**
 * 指定されたVCOUNT値が有効かチェック
 * @param version ROM version
 * @param region ROM region
 * @param vcount VCOUNT value to validate
 * @returns true if valid
 */
export function isValidVCount(version: string, region: string, vcount: number): boolean {
  const validVCounts = getValidVCounts(version, region);
  return validVCounts.includes(vcount);
}

/**
 * Timer0値から対応するVCOUNT値を逆引き
 * @param version ROM version
 * @param region ROM region
 * @param timer0 Timer0 value
 * @returns 対応するVCOUNT値、または該当なしの場合null
 */
export function getVCountFromTimer0(version: string, region: string, timer0: number): 
  number | null {
  
  const params = getROMParameters(version, region);
  if (!params) return null;
  
  for (const [vcount, timer0Min, timer0Max] of params.vcountTimerRanges) {
    if (timer0 >= timer0Min && timer0 <= timer0Max) {
      return vcount;
    }
  }
  
  return null;
}

/**
 * Timer0範囲から対応するVCount範囲を計算（Auto mode用）
 * vcountTimerRangesを順方向で使用し、Timer0からVCountへの逆引きを回避
 * 
 * @param version ROM version
 * @param region ROM region
 * @param timer0Min Timer0 minimum
 * @param timer0Max Timer0 maximum
 * @returns VCount範囲 (min/max)
 */
export function computeVCountRangeFromTimer0Range(
  version: string,
  region: string,
  timer0Min: number,
  timer0Max: number
): { min: number; max: number } {
  const params = getROMParameters(version, region);
  if (!params || params.vcountTimerRanges.length === 0) {
    return { min: 0x60, max: 0x60 };
  }

  const vcounts: number[] = [];

  // vcountTimerRangesを順方向で走査（逆引き不要）
  for (const [vcount, rangeMin, rangeMax] of params.vcountTimerRanges) {
    // 指定されたTimer0範囲との交差があるかチェック
    const effectiveMin = Math.max(timer0Min, rangeMin);
    const effectiveMax = Math.min(timer0Max, rangeMax);

    if (effectiveMin <= effectiveMax) {
      vcounts.push(vcount);
    }
  }

  if (vcounts.length === 0) {
    return { min: 0x60, max: 0x60 };
  }

  return {
    min: Math.min(...vcounts),
    max: Math.max(...vcounts),
  };
}

/**
 * 指定されたROMで利用可能なTimer0の全範囲を取得
 * @param version ROM version
 * @param region ROM region
 * @returns Timer0の最小・最大値、または該当なしの場合null
 */
export function getFullTimer0Range(version: string, region: string): 
  { min: number; max: number } | null {
  
  const params = getROMParameters(version, region);
  if (!params || params.vcountTimerRanges.length === 0) return null;
  
  let min = Number.MAX_SAFE_INTEGER;
  let max = Number.MIN_SAFE_INTEGER;
  
  for (const [, timer0Min, timer0Max] of params.vcountTimerRanges) {
    min = Math.min(min, timer0Min);
    max = Math.max(max, timer0Max);
  }
  
  return { min, max };
}

/**
 * VCOUNTずれ対応バージョンかどうかを判定
 * @param version ROM version
 * @param region ROM region
 * @returns true if VCOUNT offset version
 */
export function hasVCountOffset(version: string, region: string): boolean {
  const params = getROMParameters(version, region);
  if (!params) return false;
  
  return params.vcountTimerRanges.length > 1;
}

/**
 * Timer0とVCountの組み合わせが有効かチェック
 * @param version ROM version
 * @param region ROM region
 * @param timer0 Timer0 value
 * @param vcount VCount value
 * @returns true if the combination is valid for the ROM
 */
export function isValidTimer0VCountPair(version: string, region: string, timer0: number, vcount: number): boolean {
  const params = getROMParameters(version, region);
  if (!params) return false;
  
  for (const [validVCount, timer0Min, timer0Max] of params.vcountTimerRanges) {
    if (vcount === validVCount && timer0 >= timer0Min && timer0 <= timer0Max) {
      return true;
    }
  }
  
  return false;
}
