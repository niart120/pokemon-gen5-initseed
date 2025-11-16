import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const searchControlPanelTitle: LocaleText = {
  ja: 'Search Control',
  en: 'Search Control',
};

export const searchControlWakeLockLabel: LocaleText = {
  ja: 'Keep Screen On',
  en: 'Keep Screen On',
};

export const searchControlExecutionModeAriaLabel: LocaleText = {
  ja: '検索実行モード',
  en: 'Search execution mode',
};

export const searchControlExecutionModeLabels = {
  cpuParallel: {
    ja: 'CPU Parallel',
    en: 'CPU Parallel',
  } satisfies LocaleText,
  gpu: {
    ja: 'GPU',
    en: 'GPU',
  } satisfies LocaleText,
} as const;

export const searchControlExecutionModeHints = {
  cpuParallelUnavailable: {
    ja: 'このデバイスでは並列ワーカーを利用できません',
    en: 'Parallel workers are not available on this device',
  } satisfies LocaleText,
  gpuUnavailable: {
    ja: 'このブラウザではWebGPUを利用できません',
    en: 'WebGPU is not available in this browser',
  } satisfies LocaleText,
} as const;

export const searchControlButtonLabels = {
  start: {
    ja: 'Search',
    en: 'Search',
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
} as const;

export const searchControlWorkerThreadsLabel: LocaleText = {
  ja: 'Worker Threads',
  en: 'Worker Threads',
};

export const searchControlWorkerMinLabel: LocaleText = {
  ja: '1 worker',
  en: '1 worker',
};

export function formatSearchControlCpuCoresLabel(cores: number, locale: SupportedLocale): string {
  const formatted = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(cores);
  return `CPU cores: ${formatted}`;
}

export function formatSearchControlMaxWorkersLabel(maxWorkers: number, locale: SupportedLocale): string {
  const formatted = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(maxWorkers);
  return `${formatted} max`;
}

export function formatSearchControlMissingTargetsAlert(locale: SupportedLocale): string {
  return locale === 'ja'
    ? '検索を開始する前に目標Seedを追加してください。'
    : 'Add target seeds before starting the search.';
}

export function formatSearchControlNoMatchesAlert(totalSteps: number, locale: SupportedLocale): string {
  const steps = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(totalSteps);
  if (locale === 'ja') {
    return `検索が完了しました。${steps}通りを調べましたが一致するSeedは見つかりませんでした。\n\n次の点を確認してください:\n- 日付範囲を広げる\n- Timer0/VCountの範囲を見直す\n- 目標Seedの形式を確認する\n\n詳細なデバッグ情報はブラウザのコンソールを参照してください。`;
  }
  return `Search completed. No matches found in ${steps} combinations.\n\nTry:\n- Expanding the date range\n- Checking Timer0/VCount ranges\n- Verifying target seed format\n\nCheck browser console for detailed debug information.`;
}

export function formatSearchControlChangeModeWhileRunningAlert(locale: SupportedLocale): string {
  return locale === 'ja'
    ? '検索実行中は実行モードを変更できません。'
    : 'Cannot change execution mode while search is running.';
}

export function formatSearchControlSearchErrorAlert(error: string, locale: SupportedLocale): string {
  return locale === 'ja'
    ? `検索に失敗しました: ${error}`
    : `Search failed: ${error}`;
}

export function formatSearchControlStartErrorAlert(error: string, locale: SupportedLocale): string {
  return locale === 'ja'
    ? `検索を開始できませんでした: ${error}`
    : `Failed to start search: ${error}`;
}

export function resolveSearchControlButtonLabel(
  key: keyof typeof searchControlButtonLabels,
  locale: SupportedLocale,
): string {
  return resolveLocaleValue(searchControlButtonLabels[key], locale);
}

export function resolveSearchControlExecutionModeLabel(
  key: keyof typeof searchControlExecutionModeLabels,
  locale: SupportedLocale,
): string {
  return resolveLocaleValue(searchControlExecutionModeLabels[key], locale);
}

export function resolveSearchControlExecutionModeHint(
  key: keyof typeof searchControlExecutionModeHints,
  locale: SupportedLocale,
): string {
  return resolveLocaleValue(searchControlExecutionModeHints[key], locale);
}
