/**
 * i18n strings for Report Needle Search
 */

import type { SupportedLocale } from '@/types/i18n';
import type { LocaleMap, LocaleText } from './types';
import type { ReportNeedleSearchStatus, ReportNeedleSearchMode } from '@/types/report-needle-search';

export const reportNeedleCardTitle: LocaleText = {
  ja: 'レポ針検索',
  en: 'Report Needle Search',
};

export const reportNeedleStatusPrefix: LocaleText = {
  ja: 'ステータス',
  en: 'Status',
};

export const reportNeedleStatusLabels: LocaleMap<Record<ReportNeedleSearchStatus, string>> = {
  ja: {
    idle: 'Idle',
    starting: 'Starting...',
    running: 'Running',
    paused: 'Paused',
    stopping: 'Stopping...',
    completed: 'Completed',
    error: 'Error',
  },
  en: {
    idle: 'Idle',
    starting: 'Starting...',
    running: 'Running',
    paused: 'Paused',
    stopping: 'Stopping...',
    completed: 'Completed',
    error: 'Error',
  },
};

export const reportNeedleModeLabels: LocaleMap<Record<ReportNeedleSearchMode, string>> = {
  ja: {
    startup: 'Boot-Timing',
    'initial-seed': 'LCG Seed',
  },
  en: {
    startup: 'Boot-Timing',
    'initial-seed': 'LCG Seed',
  },
};

export const reportNeedleButtonLabels = {
  start: { ja: 'Search', en: 'Search' } satisfies LocaleText,
  starting: { ja: 'Starting...', en: 'Starting...' } satisfies LocaleText,
  pause: { ja: 'Pause', en: 'Pause' } satisfies LocaleText,
  resume: { ja: 'Resume', en: 'Resume' } satisfies LocaleText,
  stop: { ja: 'Stop', en: 'Stop' } satisfies LocaleText,
  reset: { ja: 'Reset', en: 'Reset' } satisfies LocaleText,
  copy: { ja: 'Copy', en: 'Copy' } satisfies LocaleText,
};

export const reportNeedleParamLabels = {
  mode: { ja: '計算モード', en: 'Mode' } satisfies LocaleText,
  searchWindow: { ja: '検索対象期間', en: 'Search Window' } satisfies LocaleText,
  startDateTime: { ja: '起動日時', en: 'Boot Time' } satisfies LocaleText,
  keyInput: { ja: 'キー入力', en: 'Key Input' } satisfies LocaleText,
  timer0Range: { ja: 'Timer0', en: 'Timer0' } satisfies LocaleText,
  vcountRange: { ja: 'VCount', en: 'VCount' } satisfies LocaleText,
  needleValue: { ja: '針入力', en: 'Needle Input' } satisfies LocaleText,
  needleHelper: { ja: '0-7の針を入力', en: 'Enter needles 0-7' } satisfies LocaleText,
  initialSeed: { ja: 'LCG Seed', en: 'LCG Seed' } satisfies LocaleText,
  advanceRange: { ja: '消費探索', en: 'Advance Range' } satisfies LocaleText,
};

export const reportNeedleKeyLabels = {
  configure: { ja: 'キー入力を設定', en: 'Configure Keys' } satisfies LocaleText,
  dialogTitle: { ja: 'キー入力の設定', en: 'Configure Key Input' } satisfies LocaleText,
  reset: { ja: 'リセット', en: 'Reset' } satisfies LocaleText,
  apply: { ja: '適用', en: 'Apply' } satisfies LocaleText,
  displayPlaceholder: { ja: '-', en: '-' } satisfies LocaleText,
};

export const reportNeedleStartupPlaceholders = {
  bootDate: { ja: '起動日', en: 'Boot Date' } satisfies LocaleText,
};

export const reportNeedleResultsLabel: LocaleText = {
  ja: '結果',
  en: 'Results',
};

export const reportNeedleResultsEmpty: LocaleText = {
  ja: '一致なし',
  en: 'No matches',
};

export const reportNeedleResultsSearching: LocaleText = {
  ja: '検索中...',
  en: 'Searching...',
};

export const reportNeedleTableHeaders: LocaleMap<{ position: string; timer0: string; vcount: string }> = {
  ja: {
    position: '消費位置',
    timer0: 'Timer0',
    vcount: 'VCount',
  },
  en: {
    position: 'Advance',
    timer0: 'Timer0',
    vcount: 'VCount',
  },
};

export function getReportNeedleStatusLabel(status: ReportNeedleSearchStatus, locale: SupportedLocale): string {
  return reportNeedleStatusLabels[locale][status] ?? status;
}

export function getReportNeedleModeLabel(mode: ReportNeedleSearchMode, locale: SupportedLocale): string {
  return reportNeedleModeLabels[locale][mode] ?? mode;
}
