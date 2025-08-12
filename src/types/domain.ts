/**
 * Domain-wide enum definitions (single source of truth for app-level concepts)
 *
 * Note:
 * - Numeric values are aligned with WASM enums but kept separate from runtime WASM exports.
 * - Use these in application code; conversions at the WASM boundary live in lib/generation.
 */

export enum DomainGameVersion {
  B = 0,
  W = 1,
  B2 = 2,
  W2 = 3,
}

// EncounterType: enum から const オブジェクトへ移行（逆引きはヘルパーで提供）
export const DomainEncounterType = {
  Normal: 0,
  Surfing: 1,
  Fishing: 2,
  ShakingGrass: 3,
  DustCloud: 4,
  PokemonShadow: 5,
  SurfingBubble: 6,
  FishingBubble: 7,
  StaticSymbol: 10,
  StaticStarter: 11,
  StaticFossil: 12,
  StaticEvent: 13,
  Roaming: 20,
} as const;
export type DomainEncounterType = typeof DomainEncounterType[keyof typeof DomainEncounterType];

// EncounterType の名前ユニオン（データスキーマやJSONとの境界で使用）
export const DomainEncounterTypeNames = [
  'Normal',
  'Surfing',
  'Fishing',
  'ShakingGrass',
  'DustCloud',
  'PokemonShadow',
  'SurfingBubble',
  'FishingBubble',
  'StaticSymbol',
  'StaticStarter',
  'StaticFossil',
  'StaticEvent',
  'Roaming',
] as const;
export type DomainEncounterTypeName = typeof DomainEncounterTypeNames[number];

// 逆引き (数値 -> 名前) マップを事前構築
const _DomainEncounterTypeReverse: Record<number, DomainEncounterTypeName> = (() => {
  const r: Record<number, DomainEncounterTypeName> = {} as Record<number, DomainEncounterTypeName>;
  for (const name of DomainEncounterTypeNames) {
    const v = (DomainEncounterType as Record<string, number>)[name];
    r[v] = name as DomainEncounterTypeName;
  }
  return r;
})();

export function getDomainEncounterTypeName(value: number): DomainEncounterTypeName | undefined {
  return _DomainEncounterTypeReverse[value];
}

export enum DomainShinyType {
  Normal = 0,
  Square = 1,
  Star = 2,
}

export enum DomainGameMode {
  BwNewGameWithSave = 0,
  BwNewGameNoSave = 1,
  BwContinue = 2,
  Bw2NewGameWithMemoryLinkSave = 3,
  Bw2NewGameNoMemoryLinkSave = 4,
  Bw2NewGameNoSave = 5,
  Bw2ContinueWithMemoryLink = 6,
  Bw2ContinueNoMemoryLink = 7,
}

// Optional: dust cloud content is a domain concept too, keep for completeness
export enum DomainDustCloudContent {
  Pokemon = 0,
  Jewel = 1,
  EvolutionStone = 2,
}

// Nature names (EN) - single source of truth for nature display
export const DomainNatureNames = [
  'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
  'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
  'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
  'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
  'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
] as const;
