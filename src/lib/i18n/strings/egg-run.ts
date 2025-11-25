import type { EggStatus } from '@/store/egg-store';
import type { EggCompletion } from '@/types/egg';
import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const eggRunPanelTitle: LocaleText = {
  ja: '実行制御',
  en: 'Execution Control',
};

export const eggRunControlsLabel: LocaleText = {
  ja: '制御ボタン',
  en: 'Control buttons',
};

export const eggRunStatusPrefix: LocaleText = {
  ja: 'ステータス:',
  en: 'Status:',
};

export const eggRunResultsLabel: LocaleText = {
  ja: '結果:',
  en: 'Results:',
};

export const eggRunProcessedLabel: LocaleText = {
  ja: '処理済み:',
  en: 'Processed:',
};

export const eggRunFilteredLabel: LocaleText = {
  ja: 'フィルター後:',
  en: 'After Filter:',
};

export const eggRunElapsedLabel: LocaleText = {
  ja: '実行時間:',
  en: 'Elapsed:',
};

export const eggRunButtonLabels = {
  start: {
    ja: '開始',
    en: 'Start',
  } satisfies LocaleText,
  starting: {
    ja: '開始中...',
    en: 'Starting...',
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

export const eggRunStatusLabels: LocaleMap<Record<EggStatus, string>> = {
  ja: {
    idle: 'アイドル',
    starting: '開始中',
    running: '実行中',
    stopping: '停止中',
    completed: '完了',
    error: 'エラー',
  },
  en: {
    idle: 'Idle',
    starting: 'Starting',
    running: 'Running',
    stopping: 'Stopping',
    completed: 'Completed',
    error: 'Error',
  },
};

export const eggRunCompletionReasonLabels: LocaleMap<Record<EggCompletion['reason'], string>> = {
  ja: {
    'max-count': '上限到達',
    stopped: 'ユーザー停止',
    error: 'エラー',
  },
  en: {
    'max-count': 'Reached limit',
    stopped: 'Stopped by user',
    error: 'Error',
  },
};

function getNumberFormatter(locale: SupportedLocale): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
}

export function getEggRunStatusLabel(status: EggStatus, locale: SupportedLocale): string {
  const labels = resolveLocaleValue(eggRunStatusLabels, locale);
  return labels[status] ?? status;
}

export function formatEggRunProgress(
  results: number,
  total: number,
  locale: SupportedLocale
): string {
  const formatter = getNumberFormatter(locale);
  const pct = total > 0 ? ((results / total) * 100).toFixed(1) : '0.0';
  return `${formatter.format(results)} / ${formatter.format(total)} (${pct}%)`;
}

export function formatEggRunElapsed(ms: number, locale: SupportedLocale): string {
  const formatter = getNumberFormatter(locale);
  return `${formatter.format(Math.round(ms))}ms`;
}
