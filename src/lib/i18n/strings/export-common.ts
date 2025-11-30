/**
 * Common export i18n strings shared across exporters
 */

import type { SupportedLocale } from '@/types/i18n';
import type { LocaleText } from './types';

/**
 * Ability label lookup by slot (0: first, 1: second, 2: hidden)
 */
export const ABILITY_LABELS: Record<0 | 1 | 2, LocaleText> = {
  0: { ja: '特性1', en: 'Ability 1' },
  1: { ja: '特性2', en: 'Ability 2' },
  2: { ja: '夢特性', en: 'Hidden' },
};

/**
 * Gender label lookup
 */
export const GENDER_LABELS: Record<'male' | 'female' | 'genderless', LocaleText> = {
  male: { ja: '♂', en: '♂' },
  female: { ja: '♀', en: '♀' },
  genderless: { ja: '-', en: '-' },
};

/**
 * Shiny label lookup (0: none, 1: square shiny, 2: star shiny)
 */
export const SHINY_LABELS: Record<0 | 1 | 2, LocaleText> = {
  0: { ja: '-', en: '-' },
  1: { ja: '◇', en: '◇' },
  2: { ja: '☆', en: '☆' },
};

/**
 * Hidden Power type names (indexed by type 0-15)
 */
export const HP_TYPE_NAMES: Record<SupportedLocale, readonly string[]> = {
  ja: ['かくとう', 'ひこう', 'どく', 'じめん', 'いわ', 'むし', 'ゴースト', 'はがね', 'ほのお', 'みず', 'くさ', 'でんき', 'エスパー', 'こおり', 'ドラゴン', 'あく'],
  en: ['Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel', 'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark'],
};

/**
 * Helper function to resolve ability label
 */
export function resolveAbilityLabel(slot: 0 | 1 | 2, locale: SupportedLocale): string {
  return ABILITY_LABELS[slot][locale];
}

/**
 * Helper function to resolve gender label
 */
export function resolveGenderLabel(gender: 'male' | 'female' | 'genderless', locale: SupportedLocale): string {
  return GENDER_LABELS[gender][locale];
}

/**
 * Helper function to resolve shiny label
 */
export function resolveShinyLabel(shinyType: 0 | 1 | 2, locale: SupportedLocale): string {
  return SHINY_LABELS[shinyType][locale];
}

/**
 * Helper function to resolve hidden power type name
 */
export function resolveHiddenPowerTypeName(hpType: number, locale: SupportedLocale): string {
  return HP_TYPE_NAMES[locale][hpType] ?? '?';
}
