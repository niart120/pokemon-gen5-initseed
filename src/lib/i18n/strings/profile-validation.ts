import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleText } from './types';

const profileNameRequired: LocaleText = {
  ja: 'プロファイル名は必須です',
  en: 'Profile name is required',
};

const profileHexInvalid: LocaleText = {
  ja: '{label}は16進数で入力してください',
  en: '{label} must be a hexadecimal value',
};

const profileRangeOrderError: LocaleText = {
  ja: '{label}の最小値は最大値以下である必要があります',
  en: '{label} min must be less than or equal to max',
};

const profileMacSegmentInvalid: LocaleText = {
  ja: 'MACセグメント{index}が不正です',
  en: 'MAC segment {index} is invalid',
};

const profileMacSegmentsCountError: LocaleText = {
  ja: 'MACアドレスは6つのセグメントが必要です',
  en: 'MAC address must contain six segments',
};

const profileIntegerRequired: LocaleText = {
  ja: '{label}は必須です',
  en: '{label} is required',
};

const profileIntegerRange: LocaleText = {
  ja: '{label}は0から65535の範囲で入力してください',
  en: '{label} must be between 0 and 65535',
};

export function resolveProfileNameRequired(locale: SupportedLocale): string {
  return resolveLocaleValue(profileNameRequired, locale);
}

export function formatProfileHexInvalid(label: string, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileHexInvalid, locale);
  return template.replace('{label}', label);
}

export function formatProfileRangeOrderError(label: string, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileRangeOrderError, locale);
  return template.replace('{label}', label);
}

export function formatProfileMacSegmentInvalid(index: number, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileMacSegmentInvalid, locale);
  return template.replace('{index}', String(index));
}

export function resolveProfileMacSegmentsCountError(locale: SupportedLocale): string {
  return resolveLocaleValue(profileMacSegmentsCountError, locale);
}

export function formatProfileIntegerRequired(label: string, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileIntegerRequired, locale);
  return template.replace('{label}', label);
}

export function formatProfileIntegerRange(label: string, locale: SupportedLocale): string {
  const template = resolveLocaleValue(profileIntegerRange, locale);
  return template.replace('{label}', label);
}
