/**
 * i18n strings for ID Adjustment Search
 */

import type { SupportedLocale } from '@/types/i18n';
import type { LocaleMap, LocaleText } from './types';

// === Status Types ===

export type IdAdjustmentSearchStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'completed'
  | 'error';

// === Card Titles ===

export const idAdjustmentCardTitle: LocaleText = {
  ja: 'ID調整検索',
  en: 'ID Adjustment Search',
};

// === Status Labels ===

export const idAdjustmentStatusPrefix: LocaleText = {
  ja: 'Status',
  en: 'Status',
};

export const idAdjustmentStatusLabels: LocaleMap<Record<IdAdjustmentSearchStatus, string>> = {
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

// === Button Labels ===

export const idAdjustmentButtonLabels = {
  startSearch: {
    ja: '検索開始',
    en: 'Start Search',
  } satisfies LocaleText,
  starting: {
    ja: '開始中...',
    en: 'Starting...',
  } satisfies LocaleText,
  pause: {
    ja: '一時停止',
    en: 'Pause',
  } satisfies LocaleText,
  resume: {
    ja: '再開',
    en: 'Resume',
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

// === Controls Labels ===

export const idAdjustmentControlsLabel: LocaleText = {
  ja: 'Search Controls',
  en: 'Search Controls',
};

export const idAdjustmentResultsLabel: LocaleText = {
  ja: '検索結果',
  en: 'Results',
};

export const idAdjustmentBasicSettingLabel: LocaleText = {
  ja: '基本設定',
  en: 'Basic Setting',
};

export const idAdjustmentIdSettingLabel: LocaleText = {
  ja: 'ID設定',
  en: 'ID Setting',
};

// === Parameter Labels ===

export const idAdjustmentParamLabels = {
  tid: {
    ja: '表ID',
    en: 'TID',
  } satisfies LocaleText,
  sid: {
    ja: '裏ID',
    en: 'SID',
  } satisfies LocaleText,
  shinyPid: {
    ja: '色違いPID',
    en: 'Shiny PID',
  } satisfies LocaleText,
  startDate: {
    ja: '開始日',
    en: 'Start Date',
  } satisfies LocaleText,
  endDate: {
    ja: '終了日',
    en: 'End Date',
  } satisfies LocaleText,
  timeRange: {
    ja: '時刻範囲',
    en: 'Time Range',
  } satisfies LocaleText,
  hour: {
    ja: '時',
    en: 'Hour',
  } satisfies LocaleText,
  minute: {
    ja: '分',
    en: 'Min',
  } satisfies LocaleText,
  second: {
    ja: '秒',
    en: 'Sec',
  } satisfies LocaleText,
  keyInput: {
    ja: 'キー入力',
    en: 'Key Input',
  } satisfies LocaleText,
  configure: {
    ja: '設定',
    en: 'Configure',
  } satisfies LocaleText,
};

// === Key Input Dialog Labels ===

export const idAdjustmentKeyDialogLabels = {
  title: {
    ja: 'キー入力設定',
    en: 'Key Input Settings',
  } satisfies LocaleText,
  reset: {
    ja: 'リセット',
    en: 'Reset',
  } satisfies LocaleText,
  apply: {
    ja: '適用',
    en: 'Apply',
  } satisfies LocaleText,
};

// === Results Table Labels ===

export const idAdjustmentResultsTableHeaders = {
  dateTime: {
    ja: '日時',
    en: 'DateTime',
  } satisfies LocaleText,
  lcgSeed: {
    ja: 'LCG Seed',
    en: 'LCG Seed',
  } satisfies LocaleText,
  tid: {
    ja: '表ID',
    en: 'TID',
  } satisfies LocaleText,
  sid: {
    ja: '裏ID',
    en: 'SID',
  } satisfies LocaleText,
  shiny: {
    ja: '色違い',
    en: 'Shiny',
  } satisfies LocaleText,
  timer0: {
    ja: 'Timer0',
    en: 'Timer0',
  } satisfies LocaleText,
  vcount: {
    ja: 'VCount',
    en: 'VCount',
  } satisfies LocaleText,
  keyInput: {
    ja: 'キー入力',
    en: 'Key Input',
  } satisfies LocaleText,
};

// === Results Filter Labels ===

export const idAdjustmentFilterLabels = {
  timer0: {
    ja: 'Timer0',
    en: 'Timer0',
  } satisfies LocaleText,
  vcount: {
    ja: 'VCount',
    en: 'VCount',
  } satisfies LocaleText,
  shinyOnly: {
    ja: '色違いのみ',
    en: 'Shiny Only',
  } satisfies LocaleText,
};

// === Empty/Searching States ===

export const idAdjustmentResultsEmpty: LocaleText = {
  ja: '検索結果がありません',
  en: 'No results found',
};

export const idAdjustmentResultsSearching: LocaleText = {
  ja: '検索中...',
  en: 'Searching...',
};

// === Shiny Type Display ===

export const idAdjustmentShinyTypeLabels: LocaleMap<Record<'square' | 'star' | 'normal', string>> = {
  ja: {
    square: '◇',
    star: '☆',
    normal: '—',
  },
  en: {
    square: '◇',
    star: '☆',
    normal: '—',
  },
};

// === Helper Functions ===

export function getIdAdjustmentStatusLabel(
  status: IdAdjustmentSearchStatus,
  locale: SupportedLocale
): string {
  return idAdjustmentStatusLabels[locale][status] ?? status;
}

export function getIdAdjustmentShinyTypeLabel(
  shinyType: 'square' | 'star' | 'normal',
  locale: SupportedLocale
): string {
  return idAdjustmentShinyTypeLabels[locale][shinyType];
}
