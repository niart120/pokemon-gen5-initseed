import type { SupportedLocale } from '@/types/i18n';
import dictionary from './display-names.json';

type DisplayNameEntry = Partial<Record<SupportedLocale, string>>;

interface EncounterDisplayNameDictionary {
  locations?: Record<string, DisplayNameEntry>;
  static?: Record<string, DisplayNameEntry>;
  categories?: Record<string, DisplayNameEntry>;
  types?: Record<string, DisplayNameEntry>;
}

type ResolvedEncounterDisplayNameDictionary = Required<EncounterDisplayNameDictionary>;

type DictionaryCategory = keyof ResolvedEncounterDisplayNameDictionary;

type DictionaryAccessor = ResolvedEncounterDisplayNameDictionary[DictionaryCategory];

const raw = (dictionary as EncounterDisplayNameDictionary) ?? {};

const data: ResolvedEncounterDisplayNameDictionary = {
  locations: raw.locations ?? {},
  static: raw.static ?? {},
  categories: raw.categories ?? {},
  types: raw.types ?? {},
};

function resolveFrom(bucket: DictionaryAccessor | undefined, key: string, locale: SupportedLocale, fallback?: string): string {
  if (!bucket) return fallback ?? key;
  const entry = bucket[key];
  if (!entry) return fallback ?? key;
  return entry[locale] ?? entry.ja ?? entry.en ?? fallback ?? key;
}

export function resolveEncounterLocationName(key: string, locale: SupportedLocale, fallback?: string): string {
  return resolveFrom(data.locations, key, locale, fallback);
}

export function resolveStaticEncounterName(key: string, locale: SupportedLocale, fallback?: string): string {
  return resolveFrom(data.static, key, locale, fallback);
}

export function resolveEncounterCategoryName(key: string, locale: SupportedLocale, fallback?: string): string {
  return resolveFrom(data.categories, key, locale, fallback);
}

export function resolveEncounterTypeName(key: string, locale: SupportedLocale, fallback?: string): string {
  return resolveFrom(data.types, key, locale, fallback);
}

export function resolveEncounterDisplayName(category: DictionaryCategory, key: string, locale: SupportedLocale, fallback?: string): string {
  const bucket = data[category];
  return resolveFrom(bucket, key, locale, fallback);
}

export function listEncounterDisplayNameKeys(category: DictionaryCategory): string[] {
  const bucket = data[category];
  if (!bucket) return [];
  return Object.keys(bucket);
}
