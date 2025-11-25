import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

type ProfileTimerFieldKey = 'timer0Min' | 'timer0Max' | 'vcountMin' | 'vcountMax' | 'frame';

export const profileTimerFieldLabels: Record<ProfileTimerFieldKey, LocaleText> = {
  timer0Min: {
    ja: 'Timer0(Min)',
    en: 'Timer0(Min)',
  },
  timer0Max: {
    ja: 'Timer0(Max)',
    en: 'Timer0(Max)',
  },
  vcountMin: {
    ja: 'VCount(Min)',
    en: 'VCount(Min)',
  },
  vcountMax: {
    ja: 'VCount(Max)',
    en: 'VCount(Max)',
  },
  frame: {
    ja: 'Frame',
    en: 'Frame',
  },
};

export const profileTimerAutoLabel: LocaleText = {
  ja: 'Auto',
  en: 'Auto',
};

export const profileTimerAutoAria: LocaleText = {
  ja: 'Timer0自動設定を切り替え',
  en: 'Toggle timer auto settings',
};

export function resolveProfileTimerFieldLabel(key: ProfileTimerFieldKey, locale: SupportedLocale): string {
  return resolveLocaleValue(profileTimerFieldLabels[key], locale);
}

export function resolveProfileTimerAutoLabel(locale: SupportedLocale): string {
  return resolveLocaleValue(profileTimerAutoLabel, locale);
}

export function resolveProfileTimerAutoAria(locale: SupportedLocale): string {
  return resolveLocaleValue(profileTimerAutoAria, locale);
}
