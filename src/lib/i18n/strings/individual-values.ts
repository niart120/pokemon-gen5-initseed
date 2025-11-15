import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

export type IvTooltipContextKey = 'wild' | 'roamer' | 'bw2';

const ivTooltipLabels: Record<IvTooltipContextKey, LocaleText> = {
  wild: {
    ja: 'BW/BW2 固定・野生 (消費0)',
    en: 'BW/BW2 Stationary/Wild (offset 0)',
  },
  roamer: {
    ja: 'BW 徘徊 (消費1)',
    en: 'BW Roamer (offset 1)',
  },
  bw2: {
    ja: 'BW2 固定・野生 (消費2)',
    en: 'BW2 Stationary/Wild (offset 2)',
  },
};

export function resolveIvTooltipLabel(key: IvTooltipContextKey, locale: SupportedLocale): string {
  return resolveLocaleValue(ivTooltipLabels[key], locale);
}
