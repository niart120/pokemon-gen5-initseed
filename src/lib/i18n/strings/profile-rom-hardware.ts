import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

export const profileRomLabels = {
  version: {
    ja: 'Version',
    en: 'Version',
  } satisfies LocaleText,
  region: {
    ja: 'Region',
    en: 'Region',
  } satisfies LocaleText,
  hardware: {
    ja: 'Hardware',
    en: 'Hardware',
  } satisfies LocaleText,
  macAddress: {
    ja: 'MAC Address',
    en: 'MAC Address',
  } satisfies LocaleText,
} as const;

export function resolveProfileRomLabel(key: keyof typeof profileRomLabels, locale: SupportedLocale): string {
  return resolveLocaleValue(profileRomLabels[key], locale);
}

export function formatProfileMacSegmentAria(index: number, locale: SupportedLocale): string {
  const formattedIndex = index + 1;
  return locale === 'ja'
    ? `MACセグメント${formattedIndex}`
    : `MAC segment ${formattedIndex}`;
}
