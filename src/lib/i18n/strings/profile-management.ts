import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

export const profileManagementSelectLabel: LocaleText = {
  ja: 'Profile',
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
    ja: 'Rename',
    en: 'Rename',
  } satisfies LocaleText,
  delete: {
    ja: 'Delete',
    en: 'Delete',
  } satisfies LocaleText,
} as const;

const profileManagementLockedBadge: LocaleText = {
  ja: '検索/世代実行中は編集できません',
  en: 'Editing is locked while searches or generation run',
};

export function resolveProfileManagementButtonLabel(
  key: keyof typeof profileManagementButtons,
  locale: SupportedLocale,
): string {
  return resolveLocaleValue(profileManagementButtons[key], locale);
}

export function resolveProfileManagementLockLabel(locale: SupportedLocale): string {
  return resolveLocaleValue(profileManagementLockedBadge, locale);
}
