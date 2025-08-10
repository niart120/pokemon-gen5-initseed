/**
 * ROM and hardware related types
 */

export type ROMVersion = 'B' | 'W' | 'B2' | 'W2';
export type ROMRegion = 'JPN' | 'KOR' | 'USA' | 'GER' | 'FRA' | 'SPA' | 'ITA';
export type Hardware = 'DS' | 'DS_LITE' | '3DS';

export interface ROMParameters {
  nazo: readonly [number, number, number, number, number];
  // 各要素: [vcount, timer0Min, timer0Max]
  // 通常版: 1要素、VCOUNTずれ版: 複数要素
  vcountTimerRanges: readonly (readonly [number, number, number])[];
}
