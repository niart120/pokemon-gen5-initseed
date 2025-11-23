import { SeedCalculator } from '@/lib/core/seed-calculator';
import type { GenerationParams, GenerationParamsHex, SeedSourceMode } from '@/types/generation';
import { hexParamsToGenerationParams } from '@/types/generation';
import type { SearchConditions } from '@/types/search';
import { KEY_INPUT_DEFAULT, keyMaskToKeyCode } from '@/lib/utils/key-input';

const seedCalculator = new SeedCalculator();
export const BOOT_TIMING_PAIR_LIMIT = 512;

export interface DerivedSeedMetadata {
  readonly seedSourceMode: SeedSourceMode;
  readonly derivedSeedIndex: number;
  readonly timer0: number;
  readonly vcount: number;
  readonly keyMask: number;
  readonly keyCode: number;
  readonly bootTimestampIso: string;
  readonly macAddress: readonly [number, number, number, number, number, number];
  readonly seedSourceSeedHex: string;
}

export interface DerivedSeedJob {
  params: GenerationParams;
  metadata: DerivedSeedMetadata;
}

export type BootTimingDerivationResult =
  | { ok: true; jobs: DerivedSeedJob[] }
  | { ok: false; error: string };

export function deriveBootTimingSeedJobs(
  draft: GenerationParamsHex,
  options?: { maxPairs?: number }
): BootTimingDerivationResult {
  const bootTiming = draft.bootTiming;
  if (!bootTiming) {
    return { ok: false, error: 'boot-timing data unavailable' };
  }

  const timestampIso = bootTiming.timestampIso;
  if (!timestampIso) {
    return { ok: false, error: 'boot-timing timestamp missing' };
  }
  const datetime = new Date(timestampIso);
  if (Number.isNaN(datetime.getTime())) {
    return { ok: false, error: 'boot-timing timestamp invalid' };
  }
  const year = datetime.getFullYear();
  const month = datetime.getMonth() + 1;
  const day = datetime.getDate();
  const hour = datetime.getHours();
  const minute = datetime.getMinutes();
  const second = datetime.getSeconds();

  const timer0Range = bootTiming.timer0Range;
  const vcountRange = bootTiming.vcountRange;
  if (!timer0Range || !vcountRange) {
    return { ok: false, error: 'timer0/vcount range missing' };
  }
  const timer0Span = timer0Range.max - timer0Range.min + 1;
  const vcountSpan = vcountRange.max - vcountRange.min + 1;
  if (timer0Span <= 0 || vcountSpan <= 0) {
    return { ok: false, error: 'timer0/vcount range invalid' };
  }
  const pairCount = timer0Span * vcountSpan;
  const maxPairs = options?.maxPairs ?? BOOT_TIMING_PAIR_LIMIT;
  if (pairCount > maxPairs) {
    return { ok: false, error: `timer0/vcount combinations exceed limit (${pairCount} > ${maxPairs})` };
  }

  const macAddress = bootTiming.macAddress;
  const keyMask = bootTiming.keyMask ?? KEY_INPUT_DEFAULT;
  const keyCode = keyMaskToKeyCode(keyMask);

  const baseConditions: SearchConditions = {
    romVersion: draft.version,
    romRegion: bootTiming.romRegion,
    hardware: bootTiming.hardware,
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: { min: timer0Range.min, max: timer0Range.max },
      vcountRange: { min: vcountRange.min, max: vcountRange.max },
    },
    timeRange: {
      hour: { start: hour, end: hour },
      minute: { start: minute, end: minute },
      second: { start: second, end: second },
    },
    dateRange: {
      startYear: year,
      endYear: year,
      startMonth: month,
      endMonth: month,
      startDay: day,
      endDay: day,
      startHour: hour,
      endHour: hour,
      startMinute: minute,
      endMinute: minute,
      startSecond: second,
      endSecond: second,
    },
    keyInput: keyMask,
    macAddress: Array.from(macAddress),
  };

  const jobs: DerivedSeedJob[] = [];
  let derivedSeedIndex = 0;
  for (let timer0 = timer0Range.min; timer0 <= timer0Range.max; timer0 += 1) {
    for (let vcount = vcountRange.min; vcount <= vcountRange.max; vcount += 1) {
      const message = seedCalculator.generateMessage(
        baseConditions,
        timer0,
        vcount,
        datetime,
        keyCode,
      );
      const { lcgSeed } = seedCalculator.calculateSeed(message);
      const seedSourceSeedHex = `0x${lcgSeed.toString(16).toUpperCase().padStart(16, '0')}`;
      const nextHex: GenerationParamsHex = {
        ...draft,
        baseSeedHex: lcgSeed.toString(16),
      };
      const params = hexParamsToGenerationParams(nextHex);
      jobs.push({
        params,
        metadata: {
          seedSourceMode: 'boot-timing',
          derivedSeedIndex,
          timer0,
          vcount,
          keyMask,
          keyCode,
          bootTimestampIso: timestampIso,
          macAddress: [...macAddress] as DerivedSeedMetadata['macAddress'],
          seedSourceSeedHex,
        },
      });
      derivedSeedIndex += 1;
    }
  }

  return { ok: true, jobs };
}
