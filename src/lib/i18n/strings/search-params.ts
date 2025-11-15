import type { SupportedLocale } from '@/types/i18n';
import { type LocaleText } from './types';

export const searchParamsPanelTitle: LocaleText = {
  ja: 'Search Parameters',
  en: 'Search Parameters',
};

export const searchParamsStartDateLabel: LocaleText = {
  ja: '開始',
  en: 'Start Date',
};

export const searchParamsEndDateLabel: LocaleText = {
  ja: '終了',
  en: 'End Date',
};

export const searchParamsCurrentRangePrefix: LocaleText = {
  ja: '現在の範囲',
  en: 'Current range',
};

export const searchParamsKeyInputLabel: LocaleText = {
  ja: 'キー入力',
  en: 'Key Input',
};

export const searchParamsConfigureButtonLabel: LocaleText = {
  ja: 'Configure',
  en: 'Configure',
};

export const searchParamsDialogTitle: LocaleText = {
  ja: 'キー入力の設定',
  en: 'Key Input Configuration',
};

export const searchParamsResetButtonLabel: LocaleText = {
  ja: 'Reset',
  en: 'Reset',
};

export const searchParamsApplyButtonLabel: LocaleText = {
  ja: 'Apply',
  en: 'Apply',
};

export function formatSearchParamsCurrentRange(start: string, end: string, locale: SupportedLocale): string {
  return locale === 'ja'
    ? `${searchParamsCurrentRangePrefix.ja}：${start}〜${end}`
    : `${searchParamsCurrentRangePrefix.en}: ${start} to ${end}`;
}
