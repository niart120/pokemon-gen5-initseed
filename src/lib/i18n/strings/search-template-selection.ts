import type { SupportedLocale } from '@/types/i18n';
import { type LocaleText } from './types';

const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

export const templateSelectionDialogTitle: LocaleText = {
  ja: 'Seedテンプレートの選択',
  en: 'Select Seed Templates',
};

export const templateSelectionDialogDescription: LocaleText = {
  ja: '読み込みたいテンプレートを選択してください。選択したテンプレートのSeedは統合されます。',
  en: 'Select one or more templates to load pre-defined seed lists. Selected templates will be merged.',
};

export const templateSelectionCancelButtonLabel: LocaleText = {
  ja: 'キャンセル',
  en: 'Cancel',
};

export function formatTemplateSelectionApplyButtonLabel(count: number, locale: SupportedLocale): string {
  const formatted = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(count);
  if (locale === 'ja') {
    return `選択を適用 (${formatted})`;
  }
  return `Apply Selected (${formatted})`;
}

export function formatTemplateSelectionSeedCount(count: number, locale: SupportedLocale): string {
  const formatted = new Intl.NumberFormat(BCP47_BY_LOCALE[locale]).format(count);
  if (locale === 'ja') {
    return `${formatted}件のSeed`;
  }
  return `${formatted} seed${count === 1 ? '' : 's'}`;
}
