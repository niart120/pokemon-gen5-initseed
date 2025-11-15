import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

type StatKey = 'hp' | 'attack' | 'defense' | 'specialAttack' | 'specialDefense' | 'speed';

enum FieldKey {
  Species = 'species',
  Ability = 'ability',
  Gender = 'gender',
  Nature = 'nature',
  Shiny = 'shiny',
  Level = 'level',
}

export const generationResultsControlTitle: LocaleText = {
  ja: '結果フィルター',
  en: 'Results Control',
};

export const generationResultsControlFiltersHeading: LocaleText = {
  ja: 'フィルター',
  en: 'Filters',
};

export const generationResultsControlResetFiltersLabel: LocaleText = {
  ja: 'フィルターをリセット',
  en: 'Reset Filters',
};

export const generationResultsControlClearResultsLabel: LocaleText = {
  ja: '結果をクリア',
  en: 'Clear Results',
};

export const generationResultsControlFieldLabels: LocaleMap<Record<FieldKey, string>> = {
  ja: {
    species: '種族',
    ability: '特性',
    gender: '性別',
    nature: '性格',
    shiny: '色違い',
    level: 'Lv',
  },
  en: {
    species: 'Species',
    ability: 'Ability',
    gender: 'Gender',
    nature: 'Nature',
    shiny: 'Shiny',
    level: 'Lv',
  },
};

export const generationResultsControlStatLabels: LocaleMap<Record<StatKey, string>> = {
  ja: {
    hp: 'HP',
    attack: 'Atk',
    defense: 'Def',
    specialAttack: 'SpA',
    specialDefense: 'SpD',
    speed: 'Spe',
  },
  en: {
    hp: 'HP',
    attack: 'Atk',
    defense: 'Def',
    specialAttack: 'SpA',
    specialDefense: 'SpD',
    speed: 'Spe',
  },
};

export const generationResultsControlLevelAriaLabel: LocaleText = {
  ja: 'レベルの指定値',
  en: 'Level exact value',
};

export const generationResultsControlStatExactValueSuffix: LocaleText = {
  ja: 'の指定値',
  en: 'exact value',
};

export const generationResultsControlAbilityPreviewEllipsis: LocaleText = {
  ja: '...',
  en: '...',
};

export function resolveGenerationResultsControlFieldLabel(key: FieldKey, locale: SupportedLocale): string {
  const labels = resolveLocaleValue(generationResultsControlFieldLabels, locale);
  return labels[key];
}

export function formatGenerationResultsControlStatAria(statLabel: string, locale: SupportedLocale): string {
  const suffix = resolveLocaleValue(generationResultsControlStatExactValueSuffix, locale);
  return locale === 'ja' ? `${statLabel}${suffix}` : `${statLabel} ${suffix}`;
}

export { FieldKey as GenerationResultsControlFieldKey };
