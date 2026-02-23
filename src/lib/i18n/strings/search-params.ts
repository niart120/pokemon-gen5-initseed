import type { SupportedLocale } from '@/types/i18n';
import { type LocaleText } from './types';

export const searchParamsPanelTitle: LocaleText = {
  ja: 'Search Parameters',
  en: 'Search Parameters',
};

export const searchParamsCurrentRangePrefix: LocaleText = {
  ja: '現在の範囲',
  en: 'Current range',
};

export function formatSearchParamsCurrentRange(start: string, end: string, locale: SupportedLocale): string {
  return locale === 'ja'
    ? `${searchParamsCurrentRangePrefix.ja}：${start}〜${end}`
    : `${searchParamsCurrentRangePrefix.en}: ${start} to ${end}`;
}
