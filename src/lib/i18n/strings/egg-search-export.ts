/**
 * Egg search export i18n strings
 */

import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const eggSearchExportTriggerLabel: LocaleText = {
  ja: 'Export',
  en: 'Export',
};

export const eggSearchExportDialogTitle: LocaleText = {
  ja: 'タマゴ検索結果のエクスポート',
  en: 'Export Egg Search Results',
};

export const eggSearchExportFormatLabel: LocaleText = {
  ja: 'エクスポート形式',
  en: 'Export Format',
};

export const eggSearchExportFormatOptions = {
  csv: {
    ja: 'CSV (カンマ区切り)',
    en: 'CSV (Comma Separated Values)',
  } satisfies LocaleText,
  json: {
    ja: 'JSON',
    en: 'JSON',
  } satisfies LocaleText,
  txt: {
    ja: 'TXT (プレーンテキスト)',
    en: 'TXT (Plain Text)',
  } satisfies LocaleText,
};

export const eggSearchExportDownloadLabel: LocaleText = {
  ja: 'ファイルをダウンロード',
  en: 'Download File',
};

export const eggSearchExportCopyLabel: LocaleText = {
  ja: 'クリップボードにコピー',
  en: 'Copy to Clipboard',
};

export const eggSearchExportCopiedLabel: LocaleText = {
  ja: 'コピーしました',
  en: 'Copied!',
};

export function formatEggSearchExportTriggerLabel(resultCount: number, locale: SupportedLocale): string {
  const base = resolveLocaleValue(eggSearchExportTriggerLabel, locale);
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const countText = formatter.format(resultCount);
  return `${base} (${countText})`;
}

export function formatEggSearchExportSummary(resultCount: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const countText = formatter.format(resultCount);
  if (locale === 'ja') {
    return `${countText} 件をエクスポート`;
  }
  const unit = resultCount === 1 ? 'result' : 'results';
  return `Exporting ${countText} ${unit}`;
}

export function formatEggSearchExportCopyLabel(copied: boolean, locale: SupportedLocale): string {
  if (copied) {
    return resolveLocaleValue(eggSearchExportCopiedLabel, locale);
  }
  return resolveLocaleValue(eggSearchExportCopyLabel, locale);
}
