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
  const planResult = buildBootTimingDerivationPlan(draft, options);
  if (!planResult.ok) {
    return planResult;
  }
  const entries = buildBootTimingMessageEntries(planResult.plan);
  const jobs = entries.map(entry => buildDerivedSeedJob(draft, entry));
  return { ok: true, jobs };
}

interface BootTimingDerivationPlan {
  timestampIso: string;
  datetime: Date;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  baseConditions: SearchConditions;
  keyMask: number;
  keyCode: number;
  macAddress: readonly [number, number, number, number, number, number];
}

interface BootTimingMessageEntry {
  seed: bigint;
  metadata: DerivedSeedMetadata;
}

type BootTimingPlanResult =
  | { ok: true; plan: BootTimingDerivationPlan }
  | { ok: false; error: string };

export function buildBootTimingDerivationPlan(
  draft: GenerationParamsHex,
  options?: { maxPairs?: number },
): BootTimingPlanResult {
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
  const year = datetime.getFullYear();
  const month = datetime.getMonth() + 1;
  const day = datetime.getDate();
  const hour = datetime.getHours();
  const minute = datetime.getMinutes();
  const second = datetime.getSeconds();

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

  return {
    ok: true,
    plan: {
      timestampIso,
      datetime,
      timer0Range,
      vcountRange,
      baseConditions,
      keyMask,
      keyCode,
      macAddress,
    },
  };
}

export function buildBootTimingMessageEntries(
  plan: BootTimingDerivationPlan,
  calculator: SeedCalculator = seedCalculator,
): BootTimingMessageEntry[] {
  const entries: BootTimingMessageEntry[] = [];
  let derivedSeedIndex = 0;
  for (let timer0 = plan.timer0Range.min; timer0 <= plan.timer0Range.max; timer0 += 1) {
    for (let vcount = plan.vcountRange.min; vcount <= plan.vcountRange.max; vcount += 1) {
      const message = calculator.generateMessage(
        plan.baseConditions,
        timer0,
        vcount,
        plan.datetime,
        plan.keyCode,
      );
      const { lcgSeed } = calculator.calculateSeed(message);
      const seedSourceSeedHex = `0x${lcgSeed.toString(16).toUpperCase().padStart(16, '0')}`;
      entries.push({
        seed: lcgSeed,
        metadata: {
          seedSourceMode: 'boot-timing',
          derivedSeedIndex,
          timer0,
          vcount,
          keyMask: plan.keyMask,
          keyCode: plan.keyCode,
          bootTimestampIso: plan.timestampIso,
          macAddress: [...plan.macAddress] as DerivedSeedMetadata['macAddress'],
          seedSourceSeedHex,
        },
      });
      derivedSeedIndex += 1;
    }
  }
  return entries;
}

export function buildDerivedSeedJob(
  draft: GenerationParamsHex,
  entry: BootTimingMessageEntry,
): DerivedSeedJob {
  const nextHex: GenerationParamsHex = {
    ...draft,
    baseSeedHex: entry.seed.toString(16),
  };
  const params = hexParamsToGenerationParams(nextHex);
  return {
    params,
    metadata: entry.metadata,
  };
}
