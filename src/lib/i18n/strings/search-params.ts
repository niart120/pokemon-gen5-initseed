import type { SupportedLocale } from '@/types/i18n';
import { type LocaleText } from './types';

export const searchParamsPanelTitle: LocaleText = {
  ja: '検索フィルター',
  en: 'Search Filters',
};

export const searchParamsStartDateLabel: LocaleText = {
  ja: '開始日',
  en: 'Start Date',
};

export const searchParamsEndDateLabel: LocaleText = {
  ja: '終了日',
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
  ja: '設定',
  en: 'Configure',
};

export const searchParamsDialogTitle: LocaleText = {
  ja: 'キー入力の設定',
  en: 'Key Input Configuration',
};

export const searchParamsResetButtonLabel: LocaleText = {
  ja: 'すべてリセット',
  en: 'Reset All',
};

export const searchParamsApplyButtonLabel: LocaleText = {
  ja: '適用',
  en: 'Apply',
};

export function formatSearchParamsCurrentRange(start: string, end: string, locale: SupportedLocale): string {
  return locale === 'ja'
    ? `${searchParamsCurrentRangePrefix.ja}：${start}〜${end}`
    : `${searchParamsCurrentRangePrefix.en}: ${start} to ${end}`;
}
