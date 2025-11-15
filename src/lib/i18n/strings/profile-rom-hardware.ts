import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

export const profileRomLabels = {
  version: {
    ja: 'バージョン',
    en: 'Version',
  } satisfies LocaleText,
  region: {
    ja: 'リージョン',
    en: 'Region',
  } satisfies LocaleText,
  hardware: {
    ja: 'ハードウェア',
    en: 'Hardware',
  } satisfies LocaleText,
  macAddress: {
    ja: 'MACアドレス',
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
