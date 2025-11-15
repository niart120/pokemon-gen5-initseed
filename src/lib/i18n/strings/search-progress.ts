import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const searchProgressTitle: LocaleText = {
  ja: 'Search Progress',
  en: 'Search Progress',
};

export const searchProgressReadyMessage: LocaleText = {
  ja: '検索を開始できます',
  en: 'Ready to search',
};

export const searchProgressProgressLabel: LocaleText = {
  ja: 'Progress',
  en: 'Progress',
};

export const searchProgressMatchesLabel: LocaleText = {
  ja: 'Matches',
  en: 'Matches',
};

export const searchProgressTimeElapsedLabel: LocaleText = {
  ja: 'Elapsed',
  en: 'Elapsed',
};

export const searchProgressTimeRemainingLabel: LocaleText = {
  ja: 'Remaining',
  en: 'Remaining',
};

export const searchProgressTimeSpeedLabel: LocaleText = {
  ja: 'Speed',
  en: 'Speed',
};

export const searchProgressWorkerBadgeLabel: LocaleText = {
  ja: 'Workers',
  en: 'Workers',
};

const searchProgressWorkerSummaryTemplate: LocaleText = {
  ja: 'Workers: {active} active, {completed} completed',
  en: 'Workers: {active} active, {completed} completed',
};

const searchProgressWorkerTotalTemplate: LocaleText = {
  ja: 'Total: {total}',
  en: 'Total: {total}',
};

export const searchProgressWorkerListLabel: LocaleText = {
  ja: 'Individual Worker Progress',
  en: 'Individual Worker Progress',
};

export const searchProgressWorkerToggleLabel: LocaleText = {
  ja: 'Toggle worker details',
  en: 'Toggle worker details',
};

export const searchProgressWorkerOverviewLabel: LocaleText = {
  ja: 'Worker Overview',
  en: 'Worker Overview',
};

const searchProgressWorkerOverviewSuffix: LocaleText = {
  ja: ' ({count} workers)',
  en: ' ({count} workers)',
};

const searchProgressWorkerOverviewMatchesTemplate: LocaleText = {
  ja: 'Total Matches: {matches}',
  en: 'Total Matches: {matches}',
};

const searchProgressWorkerFooterTemplate: LocaleText = {
  ja: 'Running: {running}, Completed: {completed}, Total Matches: {matches}',
  en: 'Running: {running}, Completed: {completed}, Total Matches: {matches}',
};

const searchProgressWorkerCompletionTemplate: LocaleText = {
  ja: '完了: {completed} / {total}',
  en: '{completed} / {total} completed',
};

export const searchProgressWorkerStatusLabels: LocaleMap<Record<'initializing' | 'running' | 'paused' | 'completed' | 'error', string>> = {
  ja: {
    initializing: 'Init',
    running: 'Run',
    paused: 'Pause',
    completed: 'Done',
    error: 'Error',
  },
  en: {
    initializing: 'Init',
    running: 'Run',
    paused: 'Paused',
    completed: 'Done',
    error: 'Error',
  },
};

function numberFormatter(locale: SupportedLocale, options?: Intl.NumberFormatOptions): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale], options);
}

function replaceTokens(template: string, replacements: Record<string, string>): string {
  return Object.keys(replacements).reduce((result, key) => {
    return result.replace(`{${key}}`, replacements[key]);
  }, template);
}

export function formatSearchProgressCount(value: number, locale: SupportedLocale): string {
  return numberFormatter(locale).format(Number.isFinite(value) ? value : 0);
}

export function formatSearchProgressPercent(value: number, locale: SupportedLocale, fractionDigits = 1): string {
  const safeValue = Number.isFinite(value) ? value : 0;
  const formatter = numberFormatter(locale, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
  return `${formatter.format(safeValue)}%`;
}

export function formatSearchProgressWorkerBadge(count: number, locale: SupportedLocale): string {
  const label = resolveLocaleValue(searchProgressWorkerBadgeLabel, locale);
  return `${formatSearchProgressCount(count, locale)} ${label}`;
}

export function formatSearchProgressWorkerSummary(active: number, completed: number, locale: SupportedLocale): string {
  const template = resolveLocaleValue(searchProgressWorkerSummaryTemplate, locale);
  return replaceTokens(template, {
    active: formatSearchProgressCount(active, locale),
    completed: formatSearchProgressCount(completed, locale),
  });
}

export function formatSearchProgressWorkerTotal(total: number, locale: SupportedLocale): string {
  const template = resolveLocaleValue(searchProgressWorkerTotalTemplate, locale);
  return replaceTokens(template, {
    total: formatSearchProgressCount(total, locale),
  });
}

export function formatSearchProgressWorkerOverview(count: number, locale: SupportedLocale): string {
  const label = resolveLocaleValue(searchProgressWorkerOverviewLabel, locale);
  const suffix = resolveLocaleValue(searchProgressWorkerOverviewSuffix, locale);
  return `${label}${replaceTokens(suffix, { count: formatSearchProgressCount(count, locale) })}`;
}

export function formatSearchProgressWorkerMatches(matches: number, locale: SupportedLocale): string {
  const template = resolveLocaleValue(searchProgressWorkerOverviewMatchesTemplate, locale);
  return replaceTokens(template, {
    matches: formatSearchProgressCount(matches, locale),
  });
}

export function formatSearchProgressWorkerFooter(
  running: number,
  completed: number,
  matches: number,
  locale: SupportedLocale,
): string {
  const template = resolveLocaleValue(searchProgressWorkerFooterTemplate, locale);
  return replaceTokens(template, {
    running: formatSearchProgressCount(running, locale),
    completed: formatSearchProgressCount(completed, locale),
    matches: formatSearchProgressCount(matches, locale),
  });
}

export function formatSearchProgressWorkerCompletion(
  completed: number,
  total: number,
  locale: SupportedLocale,
): string {
  const template = resolveLocaleValue(searchProgressWorkerCompletionTemplate, locale);
  return replaceTokens(template, {
    completed: formatSearchProgressCount(completed, locale),
    total: formatSearchProgressCount(total, locale),
  });
}

export function getSearchProgressWorkerStatusLabel(
  status: 'initializing' | 'running' | 'paused' | 'completed' | 'error',
  locale: SupportedLocale,
): string {
  const labels = resolveLocaleValue(searchProgressWorkerStatusLabels, locale);
  return labels[status] ?? status;
}
