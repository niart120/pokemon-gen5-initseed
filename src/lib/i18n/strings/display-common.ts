import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

export type ShinyLabelKey = 'normal' | 'square' | 'star' | 'unknown';

export const displayUnknownLabel: LocaleText = {
  ja: '不明',
  en: 'Unknown',
};

const shinyLabelMap: Record<ShinyLabelKey, LocaleText> = {
  normal: {
    ja: '-',
    en: '-',
  },
  square: {
    ja: '◇',
    en: '◇',
  },
  star: {
    ja: '☆',
    en: '☆',
  },
  unknown: {
    ja: 'Unknown',
    en: 'Unknown',
  },
};

export function resolveDisplayUnknownLabel(locale: SupportedLocale): string {
  return resolveLocaleValue(displayUnknownLabel, locale);
}

export function resolveShinyLabel(key: ShinyLabelKey, locale: SupportedLocale): string {
  return resolveLocaleValue(shinyLabelMap[key], locale);
}
