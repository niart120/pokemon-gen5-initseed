import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const searchResultsTitle: LocaleText = {
  ja: 'Results',
  en: 'Results',
};

export const searchResultsInitialMessage: LocaleText = {
  ja: '検索結果はまだありません。検索を実行するとここに表示されます。',
  en: 'No results yet. Run a search to see results here.',
};

export const searchResultsFilteredEmptyMessage: LocaleText = {
  ja: '現在のフィルター条件に一致する結果はありません。',
  en: 'No results match the current filter criteria.',
};

export const searchResultsHeaders = {
  action: {
    ja: '',
    en: '',
  } satisfies LocaleText,
  lcgSeed: {
    ja: 'LCG Seed',
    en: 'LCG Seed',
  } satisfies LocaleText,
  dateTime: {
    ja: 'Date/Time',
    en: 'Date/Time',
  } satisfies LocaleText,
  mtSeed: {
    ja: 'MT Seed',
    en: 'MT Seed',
  } satisfies LocaleText,
  timer0: {
    ja: 'Timer0',
    en: 'Timer0',
  } satisfies LocaleText,
  vcount: {
    ja: 'VCount',
    en: 'VCount',
  } satisfies LocaleText,
};

export const viewDetailsLabel: LocaleText = {
  ja: '詳細を見る',
  en: 'View Details',
};

export const viewDetailsAriaLabel: LocaleText = {
  ja: '検索結果の詳細を見る',
  en: 'View search result details',
};

export const detailsButtonLabel: LocaleText = {
  ja: '詳細',
  en: 'Details',
};

export const resultDetailsTitle: LocaleText = {
  ja: 'Seed Details',
  en: 'Seed Details',
};

export const lcgSeedLabel: LocaleText = {
  ja: 'LCG Seed',
  en: 'LCG Seed',
};

export const mtSeedLabel: LocaleText = {
  ja: 'MT Seed',
  en: 'MT Seed',
};

export const dateTimeLabel: LocaleText = {
  ja: 'Date/Time',
  en: 'Date/Time',
};

export const timer0Label: LocaleText = {
  ja: 'Timer0',
  en: 'Timer0',
};

export const vcountLabel: LocaleText = {
  ja: 'VCount',
  en: 'VCount',
};

export const romLabel: LocaleText = {
  ja: 'ROM',
  en: 'ROM',
};

export const hardwareLabel: LocaleText = {
  ja: 'Hardware',
  en: 'Hardware',
};

export const keyInputLabel: LocaleText = {
  ja: 'Key Input',
  en: 'Key Input',
};

export const sha1HashLabel: LocaleText = {
  ja: 'SHA-1 Hash',
  en: 'SHA-1 Hash',
};

export const generatedMessageLabel: LocaleText = {
  ja: 'Generated Message',
  en: 'Generated Message',
};

export const copyToGenerationPanelHint: LocaleText = {
  ja: 'クリックで生成パネルへコピー',
  en: 'Click to copy to Generation Panel',
};

export const copyMtSeedHint: LocaleText = {
  ja: 'クリックでMT Seedをコピー',
  en: 'Click to copy MT Seed',
};

export const lcgSeedCopySuccess: LocaleText = {
  ja: 'LCG Seedを生成パネルにコピーしました',
  en: 'LCG Seed copied to Generation Panel',
};

export const mtSeedCopySuccess: LocaleText = {
  ja: 'MT Seedをクリップボードにコピーしました',
  en: 'MT Seed copied to clipboard',
};

export const mtSeedCopyFailure: LocaleText = {
  ja: 'MT Seedのコピーに失敗しました',
  en: 'Failed to copy MT Seed',
};

export const clipboardUnavailable: LocaleText = {
  ja: 'クリップボードを使用できません',
  en: 'Clipboard is not available',
};

export const keyInputUnavailableLabel: LocaleText = {
  ja: '利用不可',
  en: 'Unavailable',
};

export const keyInputNoneLabel: LocaleText = {
  ja: '-',
  en: '-',
};

export const keyInputJoiner: LocaleText = {
  ja: '、',
  en: ', ',
};

export function formatResultCount(count: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const value = formatter.format(count);
  return `${value} result${count === 1 ? '' : 's'}`;
}

export function formatSearchDuration(durationMs: number, _locale: SupportedLocale): string {
  const seconds = durationMs / 1000;
  const formatted = seconds.toFixed(1);
  return `Search completed in ${formatted}s`;
}

export function formatResultDateTime(date: Date, locale: SupportedLocale): string {
  const formatter = new Intl.DateTimeFormat(BCP47_BY_LOCALE[locale], {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
  return formatter.format(date);
}

export function formatKeyInputDisplay(keyNames: string[], locale: SupportedLocale): string {
  if (keyNames.length === 0) {
    return resolveLocaleValue(keyInputNoneLabel, locale);
  }
  const joiner = resolveLocaleValue(keyInputJoiner, locale);
  return keyNames.join(joiner);
}
