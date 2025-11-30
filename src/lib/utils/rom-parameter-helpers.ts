/**
 * ROM Parameters Helper Functions
 * Timer0VCountSegment形式のROMParametersデータ構造のアクセス関数
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
 * 指定されたROMで有効なVCOUNT値の一覧を取得
 * @param version ROM version
 * @param region ROM region
 * @returns 有効なVCOUNT値の配列
 */
export function getValidVCounts(version: string, region: string): number[] {
  const params = getROMParameters(version, region);
  if (!params) return [];
  
  return params.vcountTimerRanges.map((segment) => segment.vcount);
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
  
  for (const segment of params.vcountTimerRanges) {
    min = Math.min(min, segment.timer0Min);
    max = Math.max(max, segment.timer0Max);
  }
  
  return { min, max };
}
