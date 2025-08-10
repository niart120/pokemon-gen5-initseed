/**
 * Domain-wide enum definitions (single source of truth for app-level concepts)
 *
 * Note:
 * - Numeric values are aligned with WASM enums but kept separate from runtime WASM exports.
 * - Use these in application code; conversions at the WASM boundary live in lib/integration.
 */

export enum DomainGameVersion {
  B = 0,
  W = 1,
  B2 = 2,
  W2 = 3,
}

export enum DomainEncounterType {
  Normal = 0,
  Surfing = 1,
  Fishing = 2,
  ShakingGrass = 3,
  DustCloud = 4,
  PokemonShadow = 5,
  SurfingBubble = 6,
  FishingBubble = 7,
  StaticSymbol = 10,
  StaticStarter = 11,
  StaticFossil = 12,
  StaticEvent = 13,
  Roaming = 20,
}

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
