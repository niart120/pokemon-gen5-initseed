import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

export const profileRenameDialogTitle: LocaleText = {
  ja: 'プロファイル名を変更',
  en: 'Rename Profile',
};

export const profileRenameDialogDescription: LocaleText = {
  ja: '選択中のプロファイル名を更新します。',
  en: 'Update the display name for the active profile.',
};

export const profileRenameFieldLabel: LocaleText = {
  ja: 'Profile Name',
  en: 'Profile Name',
};

export const profileRenameFieldPlaceholder: LocaleText = {
  ja: 'My Profile',
  en: 'My profile',
};

export const profileRenameCancelButton: LocaleText = {
  ja: 'Cancel',
  en: 'Cancel',
};

export const profileRenameSubmitButton: LocaleText = {
  ja: 'Rename',
  en: 'Rename',
};

export function resolveProfileRenameValue(map: LocaleText, locale: SupportedLocale): string {
  return resolveLocaleValue(map, locale);
}
