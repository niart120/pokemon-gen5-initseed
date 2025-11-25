/**
 * i18n strings for Egg Search Panel
 * Based on: spec/agent/pr_egg_boot_timing_search/I18N_DESIGN.md
 */

import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';
import type { EggBootTimingSearchStatus } from '@/store/egg-boot-timing-search-store';

// NOTE: BCP47_BY_LOCALE should be moved to common.ts in a future refactor
const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

// === Panel Titles ===

export const eggSearchPanelTitle: LocaleText = {
  ja: 'Search(Egg)',
  en: 'Search(Egg)',
};

// === Run Card ===

export const eggSearchRunCardTitle: LocaleText = {
  ja: '検索制御',
  en: 'Search Control',
};

export const eggSearchStatusPrefix: LocaleText = {
  ja: 'ステータス',
  en: 'Status',
};

export const eggSearchFoundLabel: LocaleText = {
  ja: '発見数',
  en: 'Found',
};

export const eggSearchElapsedLabel: LocaleText = {
  ja: '経過時間',
  en: 'Elapsed',
};

export const eggSearchProgressLabel: LocaleText = {
  ja: '進捗',
  en: 'Progress',
};

export const eggSearchButtonLabels = {
  start: {
    ja: '検索開始',
    en: 'Start Search',
  } satisfies LocaleText,
  stop: {
    ja: '停止',
    en: 'Stop',
  } satisfies LocaleText,
  stopping: {
    ja: '停止中...',
    en: 'Stopping...',
  } satisfies LocaleText,
};

export const eggSearchStatusLabels: LocaleMap<Record<EggBootTimingSearchStatus, string>> = {
  ja: {
    idle: 'アイドル',
    starting: '開始中',
    running: '検索中',
    stopping: '停止中',
    completed: '完了',
    error: 'エラー',
  },
  en: {
    idle: 'Idle',
    starting: 'Starting',
    running: 'Searching',
    stopping: 'Stopping',
    completed: 'Completed',
    error: 'Error',
  },
};

// === Params Card ===

export const eggSearchParamsCardTitle: LocaleText = {
  ja: '検索条件',
  en: 'Search Parameters',
};

export const eggSearchParamsLabels = {
  startDatetime: {
    ja: '開始日時',
    en: 'Start Date/Time',
  } satisfies LocaleText,
  rangeSeconds: {
    ja: '検索範囲（秒）',
    en: 'Search Range (seconds)',
  } satisfies LocaleText,
  frame: {
    ja: 'フレーム',
    en: 'Frame',
  } satisfies LocaleText,
  userOffset: {
    ja: '開始Advance',
    en: 'Start Advance',
  } satisfies LocaleText,
  advanceCount: {
    ja: '検索Advance数',
    en: 'Advance Count',
  } satisfies LocaleText,
  keyInput: {
    ja: 'キー入力マスク',
    en: 'Key Input Mask',
  } satisfies LocaleText,
};

// === Filter Card ===

export const eggSearchFilterCardTitle: LocaleText = {
  ja: 'フィルター',
  en: 'Filter',
};

export const eggSearchFilterLabels = {
  shinyOnly: {
    ja: '色違いのみ',
    en: 'Shiny Only',
  } satisfies LocaleText,
  shinyHint: {
    ja: '色違いの結果のみ表示',
    en: 'Show only shiny results',
  } satisfies LocaleText,
};

// === Results Card ===

export const eggSearchResultsCardTitle: LocaleText = {
  ja: '検索結果',
  en: 'Search Results',
};

export const eggSearchResultsEmpty: LocaleText = {
  ja: '結果がありません',
  en: 'No results',
};

export const eggSearchResultsCountLabel: LocaleText = {
  ja: '件',
  en: 'results',
};

export const eggSearchResultsTableHeaders = {
  bootTime: {
    ja: '起動時間',
    en: 'Boot Time',
  } satisfies LocaleText,
  timer0: {
    ja: 'Timer0',
    en: 'Timer0',
  } satisfies LocaleText,
  vcount: {
    ja: 'VCount',
    en: 'VCount',
  } satisfies LocaleText,
  lcgSeed: {
    ja: 'LCG Seed',
    en: 'LCG Seed',
  } satisfies LocaleText,
  advance: {
    ja: 'Advance',
    en: 'Advance',
  } satisfies LocaleText,
  nature: {
    ja: '性格',
    en: 'Nature',
  } satisfies LocaleText,
  ivs: {
    ja: '個体値',
    en: 'IVs',
  } satisfies LocaleText,
  shiny: {
    ja: '色違い',
    en: 'Shiny',
  } satisfies LocaleText,
  stable: {
    ja: '安定',
    en: 'Stable',
  } satisfies LocaleText,
};

// === Helper Functions ===

function getNumberFormatter(locale: SupportedLocale): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
}

export function getEggSearchStatusLabel(
  status: EggBootTimingSearchStatus,
  locale: SupportedLocale
): string {
  const labels = resolveLocaleValue(eggSearchStatusLabels, locale);
  return labels[status] ?? status;
}

export function formatEggSearchElapsed(ms: number, locale: SupportedLocale): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return locale === 'ja' ? `${seconds}秒` : `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return locale === 'ja'
    ? `${minutes}分${remainingSeconds}秒`
    : `${minutes}m ${remainingSeconds}s`;
}

export function formatEggSearchResultsCount(
  count: number,
  locale: SupportedLocale
): string {
  const formatter = getNumberFormatter(locale);
  const countStr = formatter.format(count);
  const suffix = eggSearchResultsCountLabel[locale];
  // Japanese doesn't use space before counters (e.g., '100件' not '100 件')
  const separator = locale === 'ja' ? '' : ' ';
  return `${countStr}${separator}${suffix}`;
}
