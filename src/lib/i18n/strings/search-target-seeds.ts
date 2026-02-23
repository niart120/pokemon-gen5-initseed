import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const targetSeedsPanelTitle: LocaleText = {
  ja: 'Target Seeds',
  en: 'Target Seeds',
};

export const targetSeedsAriaLabel: LocaleText = {
  ja: '目標Seedの操作',
  en: 'Target seeds operations',
};

export const targetSeedsTemplateButtonLabel: LocaleText = {
  ja: 'Template',
  en: 'Template',
};

export const targetSeedsClearButtonLabel: LocaleText = {
  ja: 'Clear',
  en: 'Clear',
};

export const targetSeedsSupportsHexHint: LocaleText = {
  ja: '1行に1つのSeedを入力してください。0xプレフィックスの有無は任意です。',
  en: 'Enter one seed per line. Supports hex format with or without 0x prefix.',
};

export const targetSeedsPlaceholderHeading: LocaleText = {
  ja: '16進数のSeed値を入力してください:',
  en: 'Enter seed values in hexadecimal format:',
};

export const targetSeedsValidSeedsLabel: LocaleText = {
  ja: 'Valid Seeds',
  en: 'Valid Seeds',
};

export const targetSeedsParseErrorSummary: LocaleText = {
  ja: '以下の行でSeedの形式が無効です。',
  en: 'Invalid seed format on the following lines:',
};

export function formatTargetSeedsPlaceholder(exampleSeeds: string[], locale: SupportedLocale): string {
  const heading = resolveLocaleValue(targetSeedsPlaceholderHeading, locale);
  return `${heading}\n${exampleSeeds.join('\n')}`;
}

export function formatTargetSeedsErrorBadge(count: number, locale: SupportedLocale): string {
  const formatted = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(count);
  if (locale === 'ja') {
    return `${formatted}件のエラー`;
  }
  return `${formatted} error${count === 1 ? '' : 's'}`;
}

export function formatTargetSeedsErrorLine(line: number, value: string, error: string, locale: SupportedLocale): string {
  const formattedLine = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(line);
  return locale === 'ja'
    ? `行${formattedLine}: "${value}" - ${error}`
    : `Line ${formattedLine}: "${value}" - ${error}`;
}
