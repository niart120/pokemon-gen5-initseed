import type { SupportedLocale } from './i18n';
import { resolveEncounterCategoryName, resolveEncounterTypeName } from '@/data/encounters/i18n/display-name-resolver';

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
  StaticLegendary: 14,
  Roamer: 20,
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
  'StaticLegendary',
  'StaticStarter',
  'StaticFossil',
  'StaticEvent',
  'Roamer',
] as const;
export type DomainEncounterTypeName = typeof DomainEncounterTypeNames[number];

export const DomainEncounterTypeDisplayNames = {
  Normal: 'Grass/Cave',
  Surfing: 'Surfing',
  Fishing: 'Fishing',
  ShakingGrass: 'Shaking Grass',
  DustCloud: 'Dust Cloud',
  PokemonShadow: 'Pokemon Shadow',
  SurfingBubble: 'Surfing (Ripples)',
  FishingBubble: 'Fishing (Ripples)',
  StaticSymbol: 'Stationary',
  StaticLegendary: 'Legendary',
  StaticStarter: 'Starter',
  StaticFossil: 'Fossil',
  StaticEvent: 'Event',
  Roamer: 'Roamer',
} as const satisfies Record<DomainEncounterTypeName, string>;

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

export type DomainEncounterTypeCategoryKey = 'wild' | 'static' | 'gift';

export interface DomainEncounterCategoryOption {
  key: DomainEncounterTypeCategoryKey;
  label: string;
  disabled?: boolean;
  typeNames: readonly DomainEncounterTypeName[];
}

const DomainEncounterCategoryDisplayNames: Record<DomainEncounterTypeCategoryKey, string> = {
  wild: 'Wild',
  static: 'Static',
  gift: 'Gift (WIP)',
};

const WILD_ENCOUNTER_TYPE_NAMES = [
  'Normal',
  'Surfing',
  'Fishing',
  'ShakingGrass',
  'DustCloud',
  'PokemonShadow',
  'SurfingBubble',
  'FishingBubble',
] as const satisfies readonly DomainEncounterTypeName[];

const STATIC_ENCOUNTER_TYPE_NAMES = [
  'StaticSymbol',
  'StaticLegendary',
  'StaticStarter',
  'StaticFossil',
  'StaticEvent',
  'Roamer',
] as const satisfies readonly DomainEncounterTypeName[];

export const DomainEncounterCategoryOptions: readonly DomainEncounterCategoryOption[] = [
  { key: 'wild', label: DomainEncounterCategoryDisplayNames.wild, typeNames: WILD_ENCOUNTER_TYPE_NAMES },
  { key: 'static', label: DomainEncounterCategoryDisplayNames.static, typeNames: STATIC_ENCOUNTER_TYPE_NAMES },
  { key: 'gift', label: DomainEncounterCategoryDisplayNames.gift, typeNames: [], disabled: true },
] as const;

const DEFAULT_ENCOUNTER_CATEGORY: DomainEncounterTypeCategoryKey = 'wild';

const _EncounterCategoryOptionByKey: Record<DomainEncounterTypeCategoryKey, DomainEncounterCategoryOption> = (() => {
  const map: Partial<Record<DomainEncounterTypeCategoryKey, DomainEncounterCategoryOption>> = {};
  for (const option of DomainEncounterCategoryOptions) {
    map[option.key] = option;
  }
  return map as Record<DomainEncounterTypeCategoryKey, DomainEncounterCategoryOption>;
})();

const _EncounterCategoryByName: Record<DomainEncounterTypeName, DomainEncounterTypeCategoryKey> = (() => {
  const map: Partial<Record<DomainEncounterTypeName, DomainEncounterTypeCategoryKey>> = {};
  for (const option of DomainEncounterCategoryOptions) {
    for (const name of option.typeNames) {
      map[name] = option.key;
    }
  }
  return map as Record<DomainEncounterTypeName, DomainEncounterTypeCategoryKey>;
})();

export function getDomainEncounterTypeCategoryByName(name: DomainEncounterTypeName): DomainEncounterTypeCategoryKey {
  return _EncounterCategoryByName[name] ?? DEFAULT_ENCOUNTER_CATEGORY;
}

export function getDomainEncounterTypeCategory(value: DomainEncounterType): DomainEncounterTypeCategoryKey {
  const name = getDomainEncounterTypeName(value);
  if (!name) return DEFAULT_ENCOUNTER_CATEGORY;
  return getDomainEncounterTypeCategoryByName(name);
}

export function listDomainEncounterTypeNamesByCategory(category: DomainEncounterTypeCategoryKey): DomainEncounterTypeName[] {
  const option = _EncounterCategoryOptionByKey[category];
  if (!option) return [];
  return [...option.typeNames];
}

export function getDomainEncounterTypeDisplayName(
  value: DomainEncounterType | DomainEncounterTypeName | undefined,
  locale?: SupportedLocale,
): string {
  if (value === undefined) return '';
  const name = typeof value === 'string' ? value : getDomainEncounterTypeName(value);
  if (!name) return '';
  const fallback = DomainEncounterTypeDisplayNames[name] ?? name;
  if (!locale) return fallback;
  return resolveEncounterTypeName(name, locale, fallback);
}

export function getDomainEncounterCategoryDisplayName(
  key: DomainEncounterTypeCategoryKey,
  locale?: SupportedLocale,
): string {
  const fallback = DomainEncounterCategoryDisplayNames[key] ?? key;
  if (!locale) return fallback;
  return resolveEncounterCategoryName(key, locale, fallback);
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

// Nature names (localized) - single source of truth for nature display
export const DomainNatureNames = {
  en: [
    'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
    'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
    'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
    'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
    'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
  ],
  ja: [
    'がんばりや', 'さみしがり', 'ゆうかん', 'いじっぱり', 'やんちゃ',
    'ずぶとい', 'すなお', 'のんき', 'わんぱく', 'のうてんき',
    'おくびょう', 'せっかち', 'まじめ', 'ようき', 'むじゃき',
    'ひかえめ', 'おっとり', 'れいせい', 'てれや', 'うっかりや',
    'おだやか', 'おとなしい', 'なまいき', 'しんちょう', 'きまぐれ'
  ],
} as const satisfies Record<SupportedLocale, readonly string[]>;

export const DOMAIN_NATURE_COUNT = DomainNatureNames.en.length;
