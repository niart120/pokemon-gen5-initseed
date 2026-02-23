import { describe, it, expect, vi } from 'vitest';
import {
  buildBootTimingDerivationPlan,
  buildBootTimingMessageEntries,
  buildDerivedSeedJob,
} from '@/lib/generation/boot-timing-derivation';
import type { GenerationParamsHex } from '@/types/generation';
import { BOOT_TIMING_PAIR_LIMIT } from '@/lib/generation/boot-timing-derivation';
import type { SeedCalculator } from '@/lib/core/seed-calculator';

function makeDraft(overrides: Partial<GenerationParamsHex['bootTiming']> = {}): GenerationParamsHex {
  return {
    baseSeedHex: '0',
    offsetHex: '0',
    maxAdvances: 100,
    maxResults: 50,
    version: 'B',
    encounterType: 0,
    tid: 0,
    sid: 0,
    syncEnabled: false,
    syncNatureId: 0,
    shinyCharm: false,
    isShinyLocked: false,
    stopAtFirstShiny: false,
    stopOnCap: true,
    abilityMode: 'none',
    memoryLink: false,
    newGame: false,
    withSave: false,
    seedSourceMode: 'boot-timing',
    bootTiming: {
      timestampIso: '2024-05-01T10:20:30.000Z',
      keyMask: 0,
      timer0Range: { min: 10, max: 11 },
      vcountRange: { min: 20, max: 20 },
      romRegion: 'JPN',
      hardware: 'DS',
      macAddress: [1, 2, 3, 4, 5, 6],
      ...overrides,
    },
  };
}

describe('boot-timing derivation helpers', () => {
  it('rejects invalid range spans in plan builder', () => {
    const draft = makeDraft({ timer0Range: { min: 5, max: 4 } });
    const result = buildBootTimingDerivationPlan(draft);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toMatch(/range invalid/);
    }
  });

  it('rejects pairs beyond limit', () => {
    const draft = makeDraft({ timer0Range: { min: 0, max: BOOT_TIMING_PAIR_LIMIT }, vcountRange: { min: 0, max: 0 } });
    const result = buildBootTimingDerivationPlan(draft, { maxPairs: 10 });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toMatch(/exceed limit/);
    }
  });

  it('builds message entries with injected calculator', () => {
    const draft = makeDraft();
    const planResult = buildBootTimingDerivationPlan(draft);
    expect(planResult.ok).toBe(true);
    if (!planResult.ok) {
      return;
    }
    const calculator: Pick<SeedCalculator, 'generateMessage' | 'calculateSeed'> = {
      generateMessage: vi.fn((_, timer0, vcount) => ({ timer0, vcount })) as any,
      calculateSeed: vi.fn(({ timer0, vcount }) => ({
        lcgSeed: BigInt(timer0 * 1000 + vcount),
        seed: Number(timer0 * 1000 + vcount),
        hash: 'stub',
      })),
    };
    const entries = buildBootTimingMessageEntries(planResult.plan, calculator as SeedCalculator);
    expect(entries).toHaveLength(2); // timer0 10,11 × vcount single value
    expect(calculator.generateMessage).toHaveBeenCalledTimes(2);
    expect(entries[0].metadata.timer0).toBe(10);
    expect(entries[1].seed).toBe(BigInt(11 * 1000 + 20));
  });

  it('builds derived seed job from entry', () => {
    const planResult = buildBootTimingDerivationPlan(makeDraft());
    if (!planResult.ok) {
      throw new Error('plan build failed');
    }
    const entry = {
      seed: BigInt(0x1234),
      metadata: planResult.plan
        ? {
            seedSourceMode: 'boot-timing' as const,
            derivedSeedIndex: 0,
            timer0: 10,
            vcount: 20,
            keyMask: planResult.plan.keyMask,
            keyCode: planResult.plan.keyCode,
            bootTimestampIso: planResult.plan.timestampIso,
            macAddress: planResult.plan.macAddress,
            seedSourceSeedHex: '0x1234',
          }
        : (undefined as any),
    };
    const job = buildDerivedSeedJob(makeDraft(), entry);
    expect(job.metadata.seedSourceSeedHex).toBe('0x1234');
    expect(job.params.baseSeed.toString(16)).toBe(entry.seed.toString(16));
  });
});
