import type { LocaleText, LocaleMap } from './types';

export const eggFilterPanelTitle: LocaleText = {
  ja: 'Filter',
  en: 'Filter',
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
  ja: 'めざパ(タイプ)',
  en: 'Hidden Power (Type)',
};

export const eggFilterHpPowerLabel: LocaleText = {
  ja: 'めざパ(威力)',
  en: 'Hidden Power (Power)',
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
    '0': '-',
    '1': '◇',
    '2': '☆',
  },
  en: {
    'none': 'Any',
    '0': '-',
    '1': '◇',
    '2': '☆',
  },
};

export const eggFilterIvUnknownLabel: LocaleText = {
  ja: '任意',
  en: 'Any',
};

export const eggFilterTimer0Label: LocaleText = {
  ja: 'Timer0',
  en: 'Timer0',
};

export const eggFilterVcountLabel: LocaleText = {
  ja: 'VCount',
  en: 'VCount',
};

export const eggFilterTimer0Placeholder: LocaleText = {
  ja: '例: 10A0',
  en: 'e.g. 10A0',
};

export const eggFilterVcountPlaceholder: LocaleText = {
  ja: '例: 5C',
  en: 'e.g. 5C',
};

export const eggFilterBootTimingDisabledHint: LocaleText = {
  ja: 'Boot-Timing時のみ有効',
  en: 'Available in Boot-Timing mode',
};
