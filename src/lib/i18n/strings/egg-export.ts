/**
 * Egg generation export i18n strings
 */

import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const eggExportTriggerLabel: LocaleText = {
  ja: 'Export',
  en: 'Export',
};

export const eggExportDialogTitle: LocaleText = {
  ja: 'タマゴ生成結果のエクスポート',
  en: 'Export Egg Results',
};

export const eggExportFormatLabel: LocaleText = {
  ja: 'エクスポート形式',
  en: 'Export Format',
};

export const eggExportFormatOptions = {
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

export const eggExportDownloadLabel: LocaleText = {
  ja: 'ファイルをダウンロード',
  en: 'Download File',
};

export const eggExportCopyLabel: LocaleText = {
  ja: 'クリップボードにコピー',
  en: 'Copy to Clipboard',
};

export const eggExportCopiedLabel: LocaleText = {
  ja: 'コピーしました',
  en: 'Copied!',
};

export const eggExportAdditionalDataLabel: LocaleText = {
  ja: '追加情報を含める',
  en: 'Include Additional Data',
};

export const eggExportIncludeBootTimingLabel: LocaleText = {
  ja: '起動時間情報を含める',
  en: 'Include boot timing info',
};

export function formatEggExportTriggerLabel(resultCount: number, locale: SupportedLocale): string {
  const base = resolveLocaleValue(eggExportTriggerLabel, locale);
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const countText = formatter.format(resultCount);
  return `${base} (${countText})`;
}

export function formatEggExportSummary(resultCount: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const countText = formatter.format(resultCount);
  if (locale === 'ja') {
    return `${countText} 件をエクスポート`;
  }
  const unit = resultCount === 1 ? 'result' : 'results';
  return `Exporting ${countText} ${unit}`;
}

export function formatEggExportCopyLabel(copied: boolean, locale: SupportedLocale): string {
  if (copied) {
    return resolveLocaleValue(eggExportCopiedLabel, locale);
  }
  return resolveLocaleValue(eggExportCopyLabel, locale);
}
