export const SUPPORTED_LOCALES = ['ja', 'en'] as const;
export type SupportedLocale = typeof SUPPORTED_LOCALES[number];

export const DEFAULT_LOCALE: SupportedLocale = 'ja';
