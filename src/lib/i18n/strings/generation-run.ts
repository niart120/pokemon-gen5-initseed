import type { GenerationStatus } from '@/store/generation-store';
import type { GenerationCompletion } from '@/types/generation';
import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const generationRunPanelTitle: LocaleText = {
  ja: '生成実行',
  en: 'Generation Run',
};

export const generationRunControlsLabel: LocaleText = {
  ja: '生成実行の操作',
  en: 'Generation execution controls',
};

export const generationRunProgressLabel: LocaleText = {
  ja: '生成進捗',
  en: 'Generation progress',
};

export const generationRunProgressBarLabel: LocaleText = {
  ja: '生成進捗バー',
  en: 'Generation progress bar',
};

export const generationRunStatusPrefix: LocaleText = {
  ja: 'ステータス:',
  en: 'Status:',
};

export const generationRunAdvanceUnit: LocaleText = {
  ja: '消費',
  en: 'adv',
};

export const generationRunButtonLabels = {
  start: {
    ja: '開始',
    en: 'Start',
  } satisfies LocaleText,
  starting: {
    ja: '起動中...',
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
};

export const generationRunStatusLabels: LocaleMap<Record<GenerationStatus, string>> = {
  ja: {
    idle: '待機中',
    starting: '準備中',
    running: '実行中',
    paused: '一時停止中',
    stopping: '停止処理中',
    completed: '完了',
    error: 'エラー',
  },
  en: {
    idle: 'Idle',
    starting: 'Starting',
    running: 'Running',
    paused: 'Paused',
    stopping: 'Stopping',
    completed: 'Completed',
    error: 'Error',
  },
};

export const generationRunCompletionReasonLabels: LocaleMap<Record<GenerationCompletion['reason'], string>> = {
  ja: {
    'max-advances': '最大消費数到達',
    'max-results': '結果件数上限到達',
    'first-shiny': '最初の色違いで停止',
    stopped: 'ユーザー停止',
    error: 'エラー終了',
  },
  en: {
    'max-advances': 'Reached max advances',
    'max-results': 'Reached max results',
    'first-shiny': 'Stopped at first shiny',
    stopped: 'Stopped by user',
    error: 'Error',
  },
};

function getNumberFormatter(locale: SupportedLocale): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
}

export function getGenerationRunStatusLabel(status: GenerationStatus, locale: SupportedLocale): string {
  const labels = resolveLocaleValue(generationRunStatusLabels, locale);
  return labels[status] ?? status;
}

export function getGenerationRunCompletionReasonLabel(
  reason: GenerationCompletion['reason'],
  locale: SupportedLocale,
): string {
  const labels = resolveLocaleValue(generationRunCompletionReasonLabels, locale);
  return labels[reason] ?? reason;
}

export function formatGenerationRunStatusDisplay(
  status: GenerationStatus,
  reason: GenerationCompletion['reason'] | null | undefined,
  locale: SupportedLocale,
): string {
  const statusLabel = getGenerationRunStatusLabel(status, locale);
  if (!reason) {
    return statusLabel;
  }
  const reasonLabel = getGenerationRunCompletionReasonLabel(reason, locale);
  return locale === 'ja' ? `${statusLabel}（${reasonLabel}）` : `${statusLabel} (${reasonLabel})`;
}

export function formatGenerationRunAdvancesDisplay(
  done: number,
  total: number,
  locale: SupportedLocale,
): string {
  const formatter = getNumberFormatter(locale);
  const doneText = formatter.format(done);
  const totalText = formatter.format(total);
  const unit = resolveLocaleValue(generationRunAdvanceUnit, locale);
  return `${doneText}/${totalText} ${unit}`;
}

export function formatGenerationRunPercentDisplay(pct: number, locale: SupportedLocale): string {
  const value = Number.isFinite(pct) ? pct : 0;
  const rounded = value.toFixed(1);
  return `${rounded}%`;
}

export function formatGenerationRunScreenReaderSummary(
  status: string,
  advances: string,
  percent: string,
  locale: SupportedLocale,
): string {
  if (locale === 'ja') {
    return `${status}。${advances}。進捗 ${percent}。`;
  }
  return `${status}. ${advances}. ${percent} complete.`;
}
