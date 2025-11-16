import type { GenerationStatus } from '@/store/generation-store';
import type { GenerationCompletion } from '@/types/generation';
import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const generationRunPanelTitle: LocaleText = {
  ja: 'Generation Control',
  en: 'Generation Control',
};

export const generationRunControlsLabel: LocaleText = {
  ja: '生成実行の操作',
  en: 'Generation execution controls',
};

export const generationRunProgressLabel: LocaleText = {
  ja: 'Generation progress',
  en: 'Generation progress',
};

export const generationRunProgressBarLabel: LocaleText = {
  ja: 'Generation progress bar',
  en: 'Generation progress bar',
};

export const generationRunStatusPrefix: LocaleText = {
  ja: 'Status:',
  en: 'Status:',
};

export const generationRunAdvanceUnit: LocaleText = {
  ja: 'Adv',
  en: 'adv',
};

export const generationRunButtonLabels = {
  start: {
    ja: 'Generate',
    en: 'Generate',
  } satisfies LocaleText,
  starting: {
    ja: 'Generating...',
    en: 'Generating...',
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
};

export const generationRunStatusLabels: LocaleMap<Record<GenerationStatus, string>> = {
  ja: {
    idle: 'Idle',
    starting: 'Starting',
    running: 'Running',
    paused: 'Paused',
    stopping: 'Stopping',
    completed: 'Completed',
    error: 'Error',
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
    'max-advances': 'Reached max advances',
    'max-results': 'Reached max results',
    'first-shiny': 'Stopped at first shiny',
    stopped: 'Stopped by user',
    error: 'Error',
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
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale], {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  return `${formatter.format(value)}%`;
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

const generationRunProfileSyncDialogTexts = {
  title: {
    ja: 'プロフィールを保存して生成を開始',
    en: 'Save profile before generation',
  } satisfies LocaleText,
  description: {
    ja: 'ProfilePanel の変更を生成条件に反映するため、保存してから生成を開始します。',
    en: 'Save the pending ProfilePanel changes so they are applied to the generation conditions.',
  } satisfies LocaleText,
  cancel: {
    ja: 'キャンセル',
    en: 'Cancel',
  } satisfies LocaleText,
  confirm: {
    ja: '保存して生成',
    en: 'Save and generate',
  } satisfies LocaleText,
  validationError: {
    ja: 'プロフィールの入力内容に不備があります。修正してから再度生成を実行してください。',
    en: 'Profile inputs contain validation errors. Please fix them before starting generation.',
  } satisfies LocaleText,
  missingProfile: {
    ja: '有効なプロフィールが選択されていません。',
    en: 'No active profile is selected.',
  } satisfies LocaleText,
} as const;

export function resolveGenerationRunProfileSyncDialog(locale: SupportedLocale): {
  title: string;
  description: string;
  cancel: string;
  confirm: string;
  validationError: string;
  missingProfile: string;
} {
  return {
    title: resolveLocaleValue(generationRunProfileSyncDialogTexts.title, locale),
    description: resolveLocaleValue(generationRunProfileSyncDialogTexts.description, locale),
    cancel: resolveLocaleValue(generationRunProfileSyncDialogTexts.cancel, locale),
    confirm: resolveLocaleValue(generationRunProfileSyncDialogTexts.confirm, locale),
    validationError: resolveLocaleValue(generationRunProfileSyncDialogTexts.validationError, locale),
    missingProfile: resolveLocaleValue(generationRunProfileSyncDialogTexts.missingProfile, locale),
  };
}
