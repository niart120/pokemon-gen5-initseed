import { useCallback } from 'react';
import { toast } from 'sonner';
import { useEggStore } from '@/store/egg-store';
import { KEY_INPUT_DEFAULT, keyCodeToMask } from '@/lib/utils/key-input';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { bootTimingCopySuccess, lcgSeedCopySuccess } from '@/lib/i18n/strings/search-results';
import type { EggBootTimingSearchResult } from '@/types/egg-boot-timing-search';

interface EggResultDetailsClipboard {
  copySeedToGeneration: (result: EggBootTimingSearchResult) => void;
  copyBootTimingToGeneration: (result: EggBootTimingSearchResult) => void;
}

export function useEggResultDetailsClipboard(locale: 'ja' | 'en'): EggResultDetailsClipboard {
  const updateDraftParams = useEggStore(state => state.updateDraftParams);
  const updateDraftBootTiming = useEggStore(state => state.updateDraftBootTiming);

  const copySeedToGeneration = useCallback((result: EggBootTimingSearchResult) => {
    // LCG Seed を Generation(Egg) Panel にコピー
    // lcgSeedHex は "0x..." 形式なので先頭の0xを除去
    const seedHex = result.lcgSeedHex.replace(/^0x/i, '');
    updateDraftParams({
      baseSeedHex: seedHex,
      seedSourceMode: 'lcg',
    });
    toast.success(resolveLocaleValue(lcgSeedCopySuccess, locale));
  }, [locale, updateDraftParams]);

  const copyBootTimingToGeneration = useCallback((result: EggBootTimingSearchResult) => {
    const timestampIso = result.boot.datetime.toISOString();
    const keyMask = result.boot.keyCode != null ? keyCodeToMask(result.boot.keyCode) : KEY_INPUT_DEFAULT;

    // seedSourceMode を 'boot-timing' に変更
    updateDraftParams({
      seedSourceMode: 'boot-timing',
    });
    // Boot Timing パラメータを更新
    updateDraftBootTiming({
      timestampIso,
      keyMask,
    });
    toast.success(resolveLocaleValue(bootTimingCopySuccess, locale));
  }, [locale, updateDraftParams, updateDraftBootTiming]);

  return {
    copySeedToGeneration,
    copyBootTimingToGeneration,
  };
}
