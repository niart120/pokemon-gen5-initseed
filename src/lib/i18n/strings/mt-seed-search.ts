/**
 * i18n strings for MT Seed Search
 */

import type { SupportedLocale } from '@/types/i18n';
import type { LocaleMap, LocaleText } from './types';

// === Status Types ===

export type MtSeedSearchStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'completed'
  | 'error';

export type MtSeedSearchMode = 'gpu' | 'cpu';

// === Card Title ===

export const mtSeedSearchCardTitle: LocaleText = {
  ja: 'Search Control',
  en: 'Search Control',
};

// === Status Labels ===

export const mtSeedSearchStatusPrefix: LocaleText = {
  ja: 'ステータス',
  en: 'Status',
};

export const mtSeedSearchStatusLabels: LocaleMap<Record<MtSeedSearchStatus, string>> = {
  ja: {
    idle: '待機中',
    starting: '開始中...',
    running: '検索中',
    paused: '一時停止中',
    stopping: '停止中...',
    completed: '完了',
    error: 'エラー',
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

export const mtSeedSearchModeLabels: LocaleMap<Record<MtSeedSearchMode, string>> = {
  ja: {
    gpu: 'GPU',
    cpu: 'CPU',
  },
  en: {
    gpu: 'GPU',
    cpu: 'CPU',
  },
};

// === Button Labels ===

export const mtSeedSearchButtonLabels = {
  start: {
    ja: 'Search',
    en: 'Search',
  } satisfies LocaleText,
  starting: {
    ja: 'Starting...',
    en: 'Starting...',
  } satisfies LocaleText,
  pause: {
    ja: 'Pause',
    en: 'Pause',
  } satisfies LocaleText,
  resume: {
    ja: 'Resume',
    en: 'Resume',
  } satisfies LocaleText,
  stop: {
    ja: 'Stop',
    en: 'Stop',
  } satisfies LocaleText,
  reset: {
    ja: 'Reset',
    en: 'Reset',
  } satisfies LocaleText,
  copy: {
    ja: 'Copy',
    en: 'Copy',
  } satisfies LocaleText,
};

// === Parameter Labels ===

export const mtSeedSearchParamLabels = {
  mtAdvances: {
    ja: 'MT消費数',
    en: 'MT Advances',
  } satisfies LocaleText,
  roamer: {
    ja: '徘徊',
    en: 'Roamer',
  } satisfies LocaleText,
  ivRanges: {
    ja: 'IV範囲',
    en: 'IV Ranges',
  } satisfies LocaleText,
  hpType: {
    ja: 'めざパタイプ',
    en: 'HP Type',
  } satisfies LocaleText,
  hpPower: {
    ja: 'めざパ威力',
    en: 'HP Power',
  } satisfies LocaleText,
  noSelection: {
    ja: '指定なし',
    en: 'None',
  } satisfies LocaleText,
  results: {
    ja: '検索結果',
    en: 'Results',
  } satisfies LocaleText,
};

// === Stat Names ===

export const mtSeedSearchStatNames: LocaleMap<string[]> = {
  ja: ['H', 'A', 'B', 'C', 'D', 'S'],
  en: ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'],
};

// === Helper Functions ===

export function getMtSeedSearchStatusLabel(
  status: MtSeedSearchStatus,
  locale: SupportedLocale
): string {
  return mtSeedSearchStatusLabels[locale][status] ?? status;
}

export function getMtSeedSearchModeLabel(
  mode: MtSeedSearchMode,
  locale: SupportedLocale
): string {
  return mtSeedSearchModeLabels[locale][mode];
}
