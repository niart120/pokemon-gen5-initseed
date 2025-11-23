import { useCallback } from 'react';
import { toast } from 'sonner';
import { useAppStore } from '@/store/app-store';
import { lcgSeedToHex } from '@/lib/utils/lcg-seed';
import { KEY_INPUT_DEFAULT, keyCodeToMask } from '@/lib/utils/key-input';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { bootTimingCopySuccess, lcgSeedCopySuccess } from '@/lib/i18n/strings/search-results';
import type { InitialSeedResult } from '@/types/search';

interface ResultDetailsClipboard {
  copySeedToGeneration: (result: InitialSeedResult) => void;
  copyBootTimingToGeneration: (result: InitialSeedResult) => void;
}

export function useResultDetailsClipboard(locale: 'ja' | 'en'): ResultDetailsClipboard {
  const setDraftParams = useAppStore(state => state.setDraftParams);

  const copySeedToGeneration = useCallback((result: InitialSeedResult) => {
    const lcgSeedHex = lcgSeedToHex(result.lcgSeed);
    setDraftParams({
      baseSeedHex: lcgSeedHex,
    });
    toast.success(resolveLocaleValue(lcgSeedCopySuccess, locale));
  }, [locale, setDraftParams]);

  const copyBootTimingToGeneration = useCallback((result: InitialSeedResult) => {
    const timestampIso = result.datetime.toISOString();
    const timer0 = Number(result.timer0 ?? 0);
    const vcount = Number(result.vcount ?? 0);
    const keyMask = result.keyCode != null ? keyCodeToMask(result.keyCode) : KEY_INPUT_DEFAULT;

    setDraftParams({
      seedSourceMode: 'boot-timing',
      bootTiming: {
        timestampIso,
        keyMask,
        timer0Range: { min: timer0, max: timer0 },
        vcountRange: { min: vcount, max: vcount },
      },
    });
    toast.success(resolveLocaleValue(bootTimingCopySuccess, locale));
  }, [locale, setDraftParams]);

  return {
    copySeedToGeneration,
    copyBootTimingToGeneration,
  };
}
