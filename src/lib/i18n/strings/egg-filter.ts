import type { LocaleText, LocaleMap } from './types';

export const eggFilterPanelTitle: LocaleText = {
  ja: 'フィルター設定',
  en: 'Filter Settings',
};

export const eggFilterDisabledLabel: LocaleText = {
  ja: 'フィルターを無効にする',
  en: 'Disable Filter',
};

export const eggFilterIvRangeTitle: LocaleText = {
  ja: '個体値範囲',
  en: 'IV Range',
};

export const eggFilterNatureLabel: LocaleText = {
  ja: '性格',
  en: 'Nature',
};

export const eggFilterGenderLabel: LocaleText = {
  ja: '性別',
  en: 'Gender',
};

export const eggFilterAbilityLabel: LocaleText = {
  ja: '特性',
  en: 'Ability',
};

export const eggFilterShinyLabel: LocaleText = {
  ja: '色違い',
  en: 'Shiny',
};

export const eggFilterHpTypeLabel: LocaleText = {
  ja: 'めざパタイプ',
  en: 'Hidden Power Type',
};

export const eggFilterHpPowerLabel: LocaleText = {
  ja: 'めざパ威力 (30-70)',
  en: 'Hidden Power Power (30-70)',
};

export const eggFilterNoSelection: LocaleText = {
  ja: '指定なし',
  en: 'Any',
};

export const eggFilterGenderOptions: LocaleMap<Record<'none' | 'male' | 'female' | 'genderless', string>> = {
  ja: {
    'none': '指定なし',
    male: '♂',
    female: '♀',
    genderless: '無性別',
  },
  en: {
    'none': 'Any',
    male: '♂',
    female: '♀',
    genderless: 'Genderless',
  },
};

export const eggFilterAbilityOptions: LocaleMap<Record<'none' | '0' | '1' | '2', string>> = {
  ja: {
    'none': '指定なし',
    '0': '特性1',
    '1': '特性2',
    '2': '夢特性',
  },
  en: {
    'none': 'Any',
    '0': 'Ability 1',
    '1': 'Ability 2',
    '2': 'Hidden',
  },
};

export const eggFilterShinyOptions: LocaleMap<Record<'none' | '0' | '1' | '2', string>> = {
  ja: {
    'none': '指定なし',
    '0': '通常',
    '1': '正方形色違い',
    '2': '星型色違い',
  },
  en: {
    'none': 'Any',
    '0': 'Normal',
    '1': 'Square Shiny',
    '2': 'Star Shiny',
  },
};

export const eggFilterIvUnknownLabel: LocaleText = {
  ja: '任意',
  en: 'Any',
};
