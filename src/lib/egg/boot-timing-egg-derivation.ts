/**
 * Boot-Timing Seed Derivation for Egg Generation
 * 
 * Derives multiple seed jobs from boot-timing parameters (Timer0/VCount range).
 * Based on: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md §10.4
 */

import { SeedCalculator } from '@/lib/core/seed-calculator';
import type {
  EggGenerationParamsHex,
  DerivedEggSeedMetadata,
  DerivedEggSeedJob,
  EggBootTimingDraft,
} from '@/types/egg';
import { hexParamsToEggParams } from '@/types/egg';
import type { SearchConditions } from '@/types/search';
import { KEY_INPUT_DEFAULT, keyMaskToKeyCode } from '@/lib/utils/key-input';

const seedCalculator = new SeedCalculator();

/** Maximum allowed Timer0 × VCount combinations */
export const EGG_BOOT_TIMING_PAIR_LIMIT = 512;

// === Result Types ===

export type EggBootTimingDerivationResult =
  | { ok: true; jobs: DerivedEggSeedJob[] }
  | { ok: false; error: string };

// === Internal Types ===

interface EggBootTimingDerivationPlan {
  timestampIso: string;
  datetime: Date;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  baseConditions: Partial<SearchConditions>;
  keyMask: number;
  keyCode: number;
  macAddress: readonly [number, number, number, number, number, number];
}

interface EggBootTimingMessageEntry {
  seed: bigint;
  metadata: DerivedEggSeedMetadata;
}

type EggBootTimingPlanResult =
  | { ok: true; plan: EggBootTimingDerivationPlan }
  | { ok: false; error: string };

// === Public API ===

/**
 * Boot-Timing パラメータから複数のSeedジョブを導出
 */
export function deriveBootTimingEggSeedJobs(
  draft: EggGenerationParamsHex,
  options?: { maxPairs?: number },
): EggBootTimingDerivationResult {
  const planResult = buildEggBootTimingDerivationPlan(draft.bootTiming, options);
  if (!planResult.ok) {
    return planResult;
  }
  const entries = buildEggBootTimingMessageEntries(planResult.plan);
  const jobs = entries.map(entry => buildDerivedEggSeedJob(draft, entry));
  return { ok: true, jobs };
}

/**
 * Boot-Timing バリデーション
 */
export function validateEggBootTimingInputs(draft: EggBootTimingDraft): string[] {
  const errors: string[] = [];

  if (!draft.timestampIso) {
    errors.push('boot-timing timestamp required');
  } else {
    const time = Date.parse(draft.timestampIso);
    if (Number.isNaN(time)) {
      errors.push('boot-timing timestamp invalid');
    }
  }

  const timer0Min = draft.timer0Range.min;
  const timer0Max = draft.timer0Range.max;
  if (timer0Min < 0 || timer0Min > 0xFFFF || timer0Max < 0 || timer0Max > 0xFFFF) {
    errors.push('timer0 range out of bounds');
  } else if (timer0Min > timer0Max) {
    errors.push('timer0 range invalid');
  }

  const vcountMin = draft.vcountRange.min;
  const vcountMax = draft.vcountRange.max;
  if (vcountMin < 0 || vcountMin > 0xFF || vcountMax < 0 || vcountMax > 0xFF) {
    errors.push('vcount range out of bounds');
  } else if (vcountMin > vcountMax) {
    errors.push('vcount range invalid');
  }

  const timer0Span = timer0Max - timer0Min + 1;
  const vcountSpan = vcountMax - vcountMin + 1;
  const pairCount = timer0Span > 0 && vcountSpan > 0 ? timer0Span * vcountSpan : 0;
  if (pairCount <= 0) {
    errors.push('timer0/vcount range produces no combinations');
  } else if (pairCount > EGG_BOOT_TIMING_PAIR_LIMIT) {
    errors.push(`timer0/vcount combinations exceed limit (${pairCount} > ${EGG_BOOT_TIMING_PAIR_LIMIT})`);
  }

  return errors;
}

// === Internal Functions ===

function buildEggBootTimingDerivationPlan(
  bootTiming: EggBootTimingDraft,
  options?: { maxPairs?: number },
): EggBootTimingPlanResult {
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
  const timer0Span = timer0Range.max - timer0Range.min + 1;
  const vcountSpan = vcountRange.max - vcountRange.min + 1;
  if (timer0Span <= 0 || vcountSpan <= 0) {
    return { ok: false, error: 'timer0/vcount range invalid' };
  }
  const pairCount = timer0Span * vcountSpan;
  const maxPairs = options?.maxPairs ?? EGG_BOOT_TIMING_PAIR_LIMIT;
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

  // SearchConditions 互換形式で構築（SeedCalculator.generateMessage に必要な部分のみ）
  const baseConditions: Partial<SearchConditions> = {
    romVersion: bootTiming.romVersion,
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

function buildEggBootTimingMessageEntries(
  plan: EggBootTimingDerivationPlan,
  calculator: SeedCalculator = seedCalculator,
): EggBootTimingMessageEntry[] {
  const entries: EggBootTimingMessageEntry[] = [];
  let derivedSeedIndex = 0;

  for (let timer0 = plan.timer0Range.min; timer0 <= plan.timer0Range.max; timer0 += 1) {
    for (let vcount = plan.vcountRange.min; vcount <= plan.vcountRange.max; vcount += 1) {
      const message = calculator.generateMessage(
        plan.baseConditions as SearchConditions,
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
          macAddress: [...plan.macAddress] as DerivedEggSeedMetadata['macAddress'],
          seedSourceSeedHex,
        },
      });
      derivedSeedIndex += 1;
    }
  }
  return entries;
}

function buildDerivedEggSeedJob(
  draft: EggGenerationParamsHex,
  entry: EggBootTimingMessageEntry,
): DerivedEggSeedJob {
  const nextHex: EggGenerationParamsHex = {
    ...draft,
    baseSeedHex: entry.seed.toString(16),
  };
  const params = hexParamsToEggParams(nextHex);
  return {
    params,
    metadata: entry.metadata,
  };
}
