import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const generationExportTriggerLabel: LocaleText = {
  ja: 'Export',
  en: 'Export',
};

export const generationExportDialogTitle: LocaleText = {
  ja: '生成結果のエクスポート',
  en: 'Export Generation Results',
};

export const generationExportFormatLabel: LocaleText = {
  ja: 'エクスポート形式',
  en: 'Export Format',
};

export const generationExportFormatOptions = {
  csv: {
    ja: 'CSV (カンマ区切り)',
    en: 'CSV (Comma Separated Values)',
  } satisfies LocaleText,
  json: {
    ja: 'JSON (JavaScript オブジェクト表記)',
    en: 'JSON (JavaScript Object Notation)',
  } satisfies LocaleText,
  txt: {
    ja: 'TXT (プレーンテキスト)',
    en: 'TXT (Plain Text)',
  } satisfies LocaleText,
};

export const generationExportDownloadLabel: LocaleText = {
  ja: 'ファイルをダウンロード',
  en: 'Download File',
};

export const generationExportCopyLabel: LocaleText = {
  ja: 'クリップボードにコピー',
  en: 'Copy to Clipboard',
};

export const generationExportCopiedLabel: LocaleText = {
  ja: 'コピーしました',
  en: 'Copied!',
};

export function formatGenerationExportTriggerLabel(resultCount: number, locale: SupportedLocale): string {
  const base = resolveLocaleValue(generationExportTriggerLabel, locale);
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const countText = formatter.format(resultCount);
  return `${base} (${countText})`;
}

export function formatGenerationExportSummary(resultCount: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const countText = formatter.format(resultCount);
  if (locale === 'ja') {
    return `結果 ${countText} 件をエクスポート`;
  }
  const unit = resultCount === 1 ? 'result' : 'results';
  return `Exporting ${countText} ${unit}`;
}
