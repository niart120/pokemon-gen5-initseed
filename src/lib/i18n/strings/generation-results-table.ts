import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

type HeaderKey =
  | 'advance'
  | 'direction'
  | 'directionValue'
  | 'species'
  | 'ability'
  | 'gender'
  | 'nature'
  | 'shiny'
  | 'level'
  | 'hp'
  | 'attack'
  | 'defense'
  | 'specialAttack'
  | 'specialDefense'
  | 'speed'
  | 'seed'
  | 'pid';

type HeaderDefinition = {
  label: LocaleText;
  sr?: LocaleText;
};

const HEADER_DEFINITIONS: Record<HeaderKey, HeaderDefinition> = {
  advance: {
    label: {
      ja: '消費',
      en: 'Adv',
    },
    sr: {
      ja: '',
      en: 'advance',
    },
  },
  direction: {
    label: {
      ja: '針',
      en: 'Dir',
    },
  },
  directionValue: {
    label: {
      ja: 'v',
      en: 'v',
    },
  },
  species: {
    label: {
      ja: 'ポケモン',
      en: 'Species',
    },
  },
  ability: {
    label: {
      ja: '特性',
      en: 'Ability',
    },
  },
  gender: {
    label: {
      ja: '性別',
      en: 'Gender',
    },
  },
  nature: {
    label: {
      ja: '性格',
      en: 'Nature',
    },
  },
  shiny: {
    label: {
      ja: '色違い',
      en: 'Shiny',
    },
  },
  level: {
    label: {
      ja: 'Lv',
      en: 'Lv',
    },
  },
  hp: {
    label: {
      ja: 'HP',
      en: 'HP',
    },
  },
  attack: {
    label: {
      ja: 'Atk',
      en: 'Atk',
    },
  },
  defense: {
    label: {
      ja: 'Def',
      en: 'Def',
    },
  },
  specialAttack: {
    label: {
      ja: 'SpA',
      en: 'SpA',
    },
  },
  specialDefense: {
    label: {
      ja: 'SpD',
      en: 'SpD',
    },
  },
  speed: {
    label: {
      ja: 'Spe',
      en: 'Spe',
    },
  },
  seed: {
    label: {
      ja: 'Seed',
      en: 'Seed',
    },
  },
  pid: {
    label: {
      ja: 'PID',
      en: 'PID',
    },
  },
};

export const generationResultsTableCaption: LocaleText = {
  ja: 'フィルター済み生成結果の一覧です。',
  en: 'Filtered generation results list.',
};

export const generationResultsTableUnknownLabel: LocaleText = {
  ja: '不明',
  en: 'Unknown',
};

export const generationResultsTableFilteredLabel: LocaleText = {
  ja: '結果',
  en: 'Results',
};

export const generationResultsTableTotalLabel: LocaleText = {
  ja: '総件数',
  en: 'Total',
};

export function resolveGenerationResultsTableHeaders(locale: SupportedLocale): Record<HeaderKey, { label: string; sr?: string }> {
  const entries = Object.entries(HEADER_DEFINITIONS) as Array<[HeaderKey, HeaderDefinition]>;
  const result: Partial<Record<HeaderKey, { label: string; sr?: string }>> = {};
  for (const [key, def] of entries) {
    const label = resolveLocaleValue(def.label, locale);
    const sr = def.sr ? resolveLocaleValue(def.sr, locale) : undefined;
    result[key] = { label, sr };
  }
  return result as Record<HeaderKey, { label: string; sr?: string }>;
}

export function formatGenerationResultsTableTitle(filteredCount: number, totalCount: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const filteredLabel = resolveLocaleValue(generationResultsTableFilteredLabel, locale);
  const totalLabel = resolveLocaleValue(generationResultsTableTotalLabel, locale);
  const filteredText = formatter.format(filteredCount);
  const totalText = formatter.format(totalCount);
  return `${filteredLabel} (${filteredText}) / ${totalLabel} ${totalText}`;
}

export function formatGenerationResultsTableAnnouncement(filteredCount: number, totalCount: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const filteredText = formatter.format(filteredCount);
  const totalText = formatter.format(totalCount);
  if (locale === 'ja') {
    return `フィルター済み結果 ${filteredText} 件 / 総計 ${totalText} 件を表示。`;
  }
  const resultWord = filteredCount === 1 ? 'result' : 'results';
  return `Showing ${filteredText} filtered ${resultWord} of ${totalText} total.`;
}
