import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

export const profileDefaultName: LocaleText = {
  ja: 'デフォルトデバイス',
  en: 'Default Device',
};

export const profileNewName: LocaleText = {
  ja: '新規プロファイル',
  en: 'New Profile',
};

export const profileImportedName: LocaleText = {
  ja: 'インポートしたプロファイル',
  en: 'Imported Profile',
};

const profileCreatedToast: LocaleText = {
  ja: 'プロファイル「{name}」を作成しました',
  en: 'Profile "{name}" created',
};

const profileImportedToast: LocaleText = {
  ja: '現在の設定をフォームに取り込みました',
  en: 'Current settings imported into form',
};

const profileSavedToast: LocaleText = {
  ja: 'プロファイルを保存しました',
  en: 'Profile saved',
};

const profileDeletedToast: LocaleText = {
  ja: 'プロファイルを削除しました',
  en: 'Profile deleted',
};

const profileDeleteConfirm: LocaleText = {
  ja: 'プロファイル「{name}」を削除しますか?',
  en: 'Delete profile "{name}"?',
};

const profileMinimumError: LocaleText = {
  ja: '少なくとも1件のプロファイルが必要です',
  en: 'At least one profile is required',
};

export function resolveProfileDefaultName(locale: SupportedLocale): string {
  return resolveLocaleValue(profileDefaultName, locale);
}

export function resolveProfileNewName(locale: SupportedLocale): string {
  return resolveLocaleValue(profileNewName, locale);
}

export function resolveProfileImportedName(locale: SupportedLocale): string {
  return resolveLocaleValue(profileImportedName, locale);
}

export function formatProfileCreatedToast(name: string, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileCreatedToast, locale);
  return template.replace('{name}', name);
}

export function resolveProfileImportedToast(locale: SupportedLocale): string {
  return resolveLocaleValue(profileImportedToast, locale);
}

export function resolveProfileSavedToast(locale: SupportedLocale): string {
  return resolveLocaleValue(profileSavedToast, locale);
}

export function resolveProfileDeletedToast(locale: SupportedLocale): string {
  return resolveLocaleValue(profileDeletedToast, locale);
}

export function formatProfileDeleteConfirm(name: string, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileDeleteConfirm, locale);
  return template.replace('{name}', name);
}

export function resolveProfileMinimumError(locale: SupportedLocale): string {
  return resolveLocaleValue(profileMinimumError, locale);
}
