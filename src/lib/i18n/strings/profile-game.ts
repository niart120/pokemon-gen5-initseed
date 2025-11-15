import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

type ProfileGameFieldKey = 'tid' | 'sid' | 'newGame' | 'withSave' | 'shinyCharm' | 'memoryLink';

export const profileGameFieldLabels: Record<ProfileGameFieldKey, LocaleText> = {
  tid: {
    ja: 'TID',
    en: 'TID',
  },
  sid: {
    ja: 'SID',
    en: 'SID',
  },
  newGame: {
    ja: '最初から始める',
    en: 'New Game',
  },
  withSave: {
    ja: 'セーブデータあり',
    en: 'With Save',
  },
  shinyCharm: {
    ja: 'ひかるおまもり',
    en: 'Shiny Charm',
  },
  memoryLink: {
    ja: 'おもいでリンク',
    en: 'Memory Link',
  },
};

export function resolveProfileGameFieldLabel(key: ProfileGameFieldKey, locale: SupportedLocale): string {
  return resolveLocaleValue(profileGameFieldLabels[key], locale);
}
