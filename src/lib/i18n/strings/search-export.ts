import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const searchExportDialogTitle: LocaleText = {
  ja: '検索結果のエクスポート',
  en: 'Export Search Results',
};

export const searchExportFormatLabel: LocaleText = {
  ja: 'エクスポート形式',
  en: 'Export Format',
};

export const searchExportFormatPlaceholder: LocaleText = {
  ja: '形式を選択',
  en: 'Select format',
};

export const searchExportFormatOptions = {
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

export const searchExportAdditionalDataLabel: LocaleText = {
  ja: '追加情報を含める',
  en: 'Include Additional Data',
};

export const searchExportIncludeDetailsLabel: LocaleText = {
  ja: 'MACアドレスとキー入力を含める',
  en: 'Include MAC address and key input',
};

export const searchExportIncludeHashLabel: LocaleText = {
  ja: 'SHA-1ハッシュを含める',
  en: 'Include SHA-1 hash',
};

export const searchExportIncludeMessageLabel: LocaleText = {
  ja: 'メッセージデータを含める (技術向け)',
  en: 'Include raw message data (technical)',
};

export const searchExportDownloadLabel: LocaleText = {
  ja: 'ファイルをダウンロード',
  en: 'Download File',
};

export const searchExportCopyLabel: LocaleText = {
  ja: 'クリップボードにコピー',
  en: 'Copy to Clipboard',
};

export const searchExportCopiedLabel: LocaleText = {
  ja: 'コピーしました',
  en: 'Copied!',
};

export function formatSearchExportTriggerLabel(count: number, locale: SupportedLocale): string {
  const base = locale === 'ja' ? 'エクスポート' : 'Export';
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const suffix = locale === 'ja' ? ` (${formatter.format(count)}件)` : ` (${formatter.format(count)})`;
  return `${base}${suffix}`;
}

export function formatSearchExportSummary(count: number, locale: SupportedLocale): string {
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
  const formattedCount = formatter.format(count);
  if (locale === 'ja') {
    return `結果 ${formattedCount} 件をエクスポート`;
  }
  return `Exporting ${formattedCount} result${count === 1 ? '' : 's'}`;
}

export function formatSearchExportCopyLabel(copied: boolean, locale: SupportedLocale): string {
  return resolveLocaleValue(copied ? searchExportCopiedLabel : searchExportCopyLabel, locale);
}
