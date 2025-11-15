import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

export const profileManagementSelectLabel: LocaleText = {
  ja: 'プロファイル',
  en: 'Profile',
};

export const profileManagementSelectPlaceholder: LocaleText = {
  ja: 'プロファイルを選択',
  en: 'Select a profile',
};

export const profileManagementNewProfileLabel: LocaleText = {
  ja: '＋ 新規プロファイル',
  en: '+ New profile',
};

export const profileManagementImportCurrentLabel: LocaleText = {
  ja: '現在の設定をインポート',
  en: 'Import current settings',
};

export const profileManagementButtons = {
  rename: {
    ja: '名前変更',
    en: 'Rename',
  } satisfies LocaleText,
  save: {
    ja: '保存',
    en: 'Save',
  } satisfies LocaleText,
  delete: {
    ja: '削除',
    en: 'Delete',
  } satisfies LocaleText,
} as const;

export const profileManagementDirtyBadge: LocaleText = {
  ja: '未保存',
  en: 'Unsaved',
};

export function resolveProfileManagementButtonLabel(
  key: keyof typeof profileManagementButtons,
  locale: SupportedLocale,
): string {
  return resolveLocaleValue(profileManagementButtons[key], locale);
}
