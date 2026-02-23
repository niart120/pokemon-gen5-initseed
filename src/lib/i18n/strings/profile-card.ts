import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

type ProfileSectionKey = 'profileManagement' | 'romHardware' | 'timer0Vcount' | 'gameState';

export const profileCardTitle: LocaleText = {
  ja: 'Profile',
  en: 'Profile',
};

const profileSectionTitles: Record<ProfileSectionKey, LocaleText> = {
  profileManagement: {
    ja: 'Profile Management',
    en: 'Profile Management',
  },
  romHardware: {
    ja: 'ROM & Hardware',
    en: 'ROM & Hardware',
  },
  timer0Vcount: {
    ja: 'Timer0 / VCount',
    en: 'Timer0 / VCount',
  },
  gameState: {
    ja: 'Game State',
    en: 'Game State',
  },
};

export function resolveProfileSectionTitle(key: ProfileSectionKey, locale: SupportedLocale): string {
  return resolveLocaleValue(profileSectionTitles[key], locale);
}
