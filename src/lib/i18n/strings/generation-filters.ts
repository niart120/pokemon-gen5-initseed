import type { ShinyFilterMode } from '@/store/generation-store';
import type { LocaleMap, LocaleText } from './types';

export const noAbilitySelectionLabel: LocaleText = {
  ja: '特性なし',
  en: 'No abilities',
};

export const noGenderSelectionLabel: LocaleText = {
  ja: '性別不明のみ',
  en: 'No genders',
};

export const shinyModeOptionLabels: LocaleMap<Record<ShinyFilterMode, string>> = {
  ja: {
    all: '指定なし',
    shiny: '☆&◇',
    star: '☆',
    square: '◇',
    'non-shiny': '通常色',
  },
  en: {
    all: 'Any',
    shiny: '☆&◇',
    star: '☆',
    square: '◇',
    'non-shiny': 'Normal',
  },
};

export const genderOptionLabels: LocaleMap<Record<'M' | 'F' | 'N', string>> = {
  ja: {
    M: 'オス',
    F: 'メス',
    N: '性別不明',
  },
  en: {
    M: 'Male',
    F: 'Female',
    N: 'Genderless',
  },
};

export const abilitySlotLabels: LocaleMap<Record<0 | 1 | 2, string>> = {
  ja: {
    0: '特性1',
    1: '特性2',
    2: '夢特性',
  },
  en: {
    0: 'Primary',
    1: 'Secondary',
    2: 'Hidden',
  },
};

export const abilityPreviewJoiner: LocaleText = {
  ja: '、',
  en: ', ',
};
