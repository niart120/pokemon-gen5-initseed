import { resolveLocaleValue, type LocaleText } from './types';
import type { SupportedLocale } from '@/types/i18n';

type ProfileSectionKey = 'profileManagement' | 'romHardware' | 'timer0Vcount' | 'gameState';

export const profileCardTitle: LocaleText = {
  ja: 'デバイスプロファイル',
  en: 'Device Profile',
};

const profileSectionTitles: Record<ProfileSectionKey, LocaleText> = {
  profileManagement: {
    ja: 'プロファイル管理',
    en: 'Profile Management',
  },
  romHardware: {
    ja: 'ROM / ハードウェア',
    en: 'ROM & Hardware',
  },
  timer0Vcount: {
    ja: 'Timer0 / VCount',
    en: 'Timer0 / VCount',
  },
  gameState: {
    ja: 'ゲーム状態',
    en: 'Game State',
  },
};

export function resolveProfileSectionTitle(key: ProfileSectionKey, locale: SupportedLocale): string {
  return resolveLocaleValue(profileSectionTitles[key], locale);
}
