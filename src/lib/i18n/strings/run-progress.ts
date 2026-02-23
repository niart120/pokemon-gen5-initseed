/**
 * run-progress.ts
 * xxxRunCard 共通の進捗表示フォーマット関数
 * 統一フォーマット: 12.3%  xxx / yyy results
 */

import type { SupportedLocale } from '@/types/i18n';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

/**
 * 数値フォーマッター取得
 */
function getNumberFormatter(locale: SupportedLocale): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
}

/**
 * 進捗率フォーマット: 12.3%
 */
export function formatRunProgressPercent(percent: number, locale: SupportedLocale): string {
  const value = Number.isFinite(percent) ? percent : 0;
  const formatter = new Intl.NumberFormat(BCP47_BY_LOCALE[locale], {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  return `${formatter.format(value)}%`;
}

/**
 * 件数フォーマット: xxx / yyy results
 */
export function formatRunProgressCount(
  current: number,
  total: number,
  locale: SupportedLocale,
): string {
  const formatter = getNumberFormatter(locale);
  const currentText = formatter.format(current);
  const totalText = formatter.format(total);
  return `${currentText} / ${totalText} results`;
}
