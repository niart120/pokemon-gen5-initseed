import { DEFAULT_LOCALE, type SupportedLocale } from '@/types/i18n';

export type LocaleMap<T> = Record<SupportedLocale, T>;
export type LocaleText = LocaleMap<string>;

export function resolveLocaleValue<T>(map: LocaleMap<T>, locale: SupportedLocale, fallback: SupportedLocale = DEFAULT_LOCALE): T {
  return map[locale] ?? map[fallback];
}
