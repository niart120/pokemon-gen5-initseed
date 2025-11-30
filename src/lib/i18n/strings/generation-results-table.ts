import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

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
  | 'pid'
  | 'timer0'
  | 'vcount'
  | 'bootTimestamp'
  | 'keyInput';

type HeaderDefinition = {
  label: LocaleText;
  sr?: LocaleText;
};

const HEADER_DEFINITIONS: Record<HeaderKey, HeaderDefinition> = {
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
  direction: {
    label: {
      ja: 'Dir',
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
      ja: '種族',
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
      ja: '色',
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
      ja: 'H',
      en: 'HP',
    },
  },
  attack: {
    label: {
      ja: 'A',
      en: 'Atk',
    },
  },
  defense: {
    label: {
      ja: 'B',
      en: 'Def',
    },
  },
  specialAttack: {
    label: {
      ja: 'C',
      en: 'SpA',
    },
  },
  specialDefense: {
    label: {
      ja: 'D',
      en: 'SpD',
    },
  },
  speed: {
    label: {
      ja: 'S',
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
  timer0: {
    label: {
      ja: 'Timer0',
      en: 'Timer0',
    },
  },
  vcount: {
    label: {
      ja: 'VCount',
      en: 'VCount',
    },
  },
  bootTimestamp: {
    label: {
      ja: '起動時刻',
      en: 'Boot Time',
    },
  },
  keyInput: {
    label: {
      ja: 'キー入力',
      en: 'Key Input',
    },
  },
};

export const generationResultsTableCaption: LocaleText = {
  ja: 'フィルター済み生成結果の一覧です。',
  en: 'Filtered generation results list.',
};

export const generationResultsTableUnknownLabel: LocaleText = {
  ja: 'Unknown',
  en: 'Unknown',
};

export const generationResultsTableEmptyMessage: LocaleText = {
  ja: '結果がありません',
  en: 'No results',
};

export const generationResultsTableInitialMessage: LocaleText = {
  ja: '生成を実行すると結果が表示されます',
  en: 'Run generation to see results',
};

export const generationResultsTableFilteredLabel: LocaleText = {
  ja: 'Results',
  en: 'Results',
};

export const generationResultsTableTotalLabel: LocaleText = {
  ja: 'Total',
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
  const filteredLabel = resolveLocaleValue(generationResultsTableFilteredLabel, locale);
  return `${filteredLabel}`;
}

/**
 * Format filtered result count in unified format: "x result(s)"
 */
export function formatGenerationResultsCount(filteredCount: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const value = formatter.format(filteredCount);
  return `${value} result${filteredCount === 1 ? '' : 's'}`;
}

/**
 * Format processing duration: "Generation completed in X.Xs"
 */
export function formatGenerationProcessingDuration(durationMs: number): string {
  const seconds = durationMs / 1000;
  return `Generation completed in ${seconds.toFixed(1)}s`;
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
