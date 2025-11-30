import type { SupportedLocale } from '@/types/i18n';
import type { LocaleText, LocaleMap } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const eggResultsPanelTitle: LocaleText = {
  ja: 'Results',
  en: 'Results',
};

export const eggResultsEmptyMessage: LocaleText = {
  ja: '結果がありません',
  en: 'No results',
};

/**
 * Format result count in unified format: "x result(s)"
 */
export function formatEggResultCount(count: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const value = formatter.format(count);
  return `${value} result${count === 1 ? '' : 's'}`;
}

/**
 * Format processing duration: "Generation completed in X.Xs"
 */
export function formatEggProcessingDuration(durationMs: number): string {
  const seconds = durationMs / 1000;
  return `Generation completed in ${seconds.toFixed(1)}s`;
}

type EggResultHeaderKey =
  | 'advance'
  | 'dir'
  | 'dirValue'
  | 'hp'
  | 'atk'
  | 'def'
  | 'spa'
  | 'spd'
  | 'spe'
  | 'nature'
  | 'gender'
  | 'ability'
  | 'shiny'
  | 'hiddenPower'
  | 'pid'
  | 'stable';

type HeaderDefinition = {
  label: LocaleText;
  sr?: LocaleText;
};

const HEADER_DEFINITIONS: Record<EggResultHeaderKey, HeaderDefinition> = {
  advance: {
    label: {
      ja: 'Adv',
      en: 'Adv',
    },
    sr: {
      ja: 'advance',
      en: 'advance',
    },
  },
  dir: {
    label: {
      ja: 'Dir',
      en: 'Dir',
    },
  },
  dirValue: {
    label: {
      ja: 'v',
      en: 'v',
    },
  },
  hp: {
    label: {
      ja: 'H',
      en: 'HP',
    },
  },
  atk: {
    label: {
      ja: 'A',
      en: 'Atk',
    },
  },
  def: {
    label: {
      ja: 'B',
      en: 'Def',
    },
  },
  spa: {
    label: {
      ja: 'C',
      en: 'SpA',
    },
  },
  spd: {
    label: {
      ja: 'D',
      en: 'SpD',
    },
  },
  spe: {
    label: {
      ja: 'S',
      en: 'Spe',
    },
  },
  nature: {
    label: {
      ja: '性格',
      en: 'Nature',
    },
  },
  gender: {
    label: {
      ja: '性別',
      en: 'Gender',
    },
  },
  ability: {
    label: {
      ja: '特性',
      en: 'Ability',
    },
  },
  shiny: {
    label: {
      ja: '色',
      en: 'Shiny',
    },
  },
  hiddenPower: {
    label: {
      ja: 'めざパ',
      en: 'HP',
    },
  },
  pid: {
    label: {
      ja: 'PID',
      en: 'PID',
    },
  },
  stable: {
    label: {
      ja: '安定',
      en: 'Stable',
    },
  },
};

export function getEggResultHeader(key: EggResultHeaderKey, locale: keyof LocaleText): string {
  return HEADER_DEFINITIONS[key].label[locale];
}

export function getEggResultSrLabel(key: EggResultHeaderKey, locale: keyof LocaleText): string | undefined {
  return HEADER_DEFINITIONS[key].sr?.[locale];
}

export const eggResultShinyLabels: LocaleMap<Record<0 | 1 | 2, string>> = {
  ja: {
    0: '-',
    1: '◇',
    2: '☆',
  },
  en: {
    0: '-',
    1: '◇',
    2: '☆',
  },
};

export const eggResultGenderLabels: LocaleMap<Record<'male' | 'female' | 'genderless', string>> = {
  ja: {
    male: '♂',
    female: '♀',
    genderless: '-',
  },
  en: {
    male: '♂',
    female: '♀',
    genderless: '-',
  },
};

export const eggResultAbilityLabels: LocaleMap<Record<0 | 1 | 2, string>> = {
  ja: {
    0: '特性1',
    1: '特性2',
    2: '夢特性',
  },
  en: {
    0: 'Ability 1',
    1: 'Ability 2',
    2: 'Hidden',
  },
};

export const eggResultStableLabels: LocaleMap<Record<'yes' | 'no', string>> = {
  ja: {
    yes: '○',
    no: '-',
  },
  en: {
    yes: '○',
    no: '-',
  },
};

export const eggResultUnknownIv: LocaleText = {
  ja: '?',
  en: '?',
};

export const eggResultUnknownHp: LocaleText = {
  ja: '?/?',
  en: '?/?',
};
