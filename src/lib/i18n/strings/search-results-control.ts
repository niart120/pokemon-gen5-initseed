import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

export type SearchResultsSortKey = 'datetime' | 'seed' | 'timer0' | 'vcount';

export const searchResultsControlTitle: LocaleText = {
  ja: '結果コントロール',
  en: 'Results Control',
};

export const searchResultsControlClearButton: LocaleText = {
  ja: '結果をクリア',
  en: 'Clear Results',
};

export const searchResultsControlFilterLabel: LocaleText = {
  ja: 'Seedフィルタ',
  en: 'Filter by Seed',
};

export const searchResultsControlFilterPlaceholder: LocaleText = {
  ja: 'Seed値を入力 (16進)',
  en: 'Enter seed value (hex)',
};

export const searchResultsControlSortLabel: LocaleText = {
  ja: '並び替え',
  en: 'Sort by',
};

export const searchResultsControlSortPlaceholder: LocaleText = {
  ja: '項目を選択',
  en: 'Select field',
};

export const searchResultsControlSortOptionLabels: LocaleMap<Record<SearchResultsSortKey, string>> = {
  ja: {
    datetime: '日時',
    seed: 'MT Seed',
    timer0: 'Timer0',
    vcount: 'VCount',
  },
  en: {
    datetime: 'Date/Time',
    seed: 'MT Seed',
    timer0: 'Timer0',
    vcount: 'VCount',
  },
};

export function formatSearchResultsSortOption(key: SearchResultsSortKey, locale: SupportedLocale): string {
  const labels = resolveLocaleValue(searchResultsControlSortOptionLabels, locale);
  return labels[key];
}
