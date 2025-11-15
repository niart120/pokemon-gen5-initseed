import { useAppStore } from '@/store/app-store';
import type { DeviceProfile, DeviceProfileDraft } from '@/types/profile';
import { enforceMemoryLink, enforceShinyCharm } from './profileFormAdapter';

export function buildDraftFromCurrentState(label: string, base: DeviceProfile | undefined): DeviceProfileDraft {
  const state = useAppStore.getState();
  const { searchConditions } = state;
  const generationDraft = state.draftParams ?? {};
  const macAddress = Array.isArray(searchConditions.macAddress)
    ? searchConditions.macAddress.slice(0, 6)
    : [];
  while (macAddress.length < 6) macAddress.push(0);

  const romVersion = searchConditions.romVersion;
  const newGame = Boolean(generationDraft.newGame);
  const withSave = newGame ? Boolean(generationDraft.withSave) : true;
  const memoryLink = enforceMemoryLink(Boolean(generationDraft.memoryLink), romVersion, withSave);
  const shinyCharm = enforceShinyCharm(Boolean(generationDraft.shinyCharm), romVersion, withSave);

  return {
    name: base?.name ?? label,
    description: base?.description,
    romVersion,
    romRegion: searchConditions.romRegion,
    hardware: searchConditions.hardware,
    timer0Auto: searchConditions.timer0VCountConfig.useAutoConfiguration,
    timer0Range: {
      min: searchConditions.timer0VCountConfig.timer0Range.min,
      max: searchConditions.timer0VCountConfig.timer0Range.max,
    },
    vcountRange: {
      min: searchConditions.timer0VCountConfig.vcountRange.min,
      max: searchConditions.timer0VCountConfig.vcountRange.max,
    },
    macAddress,
    tid: typeof generationDraft.tid === 'number' ? generationDraft.tid : 0,
    sid: typeof generationDraft.sid === 'number' ? generationDraft.sid : 0,
    shinyCharm,
    newGame,
    withSave,
    memoryLink,
  };
}
