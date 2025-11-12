import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { SeedCalculator } from '@/lib/core/seed-calculator';
import { buildSearchContext } from '@/lib/webgpu/seed-search/message-encoder';
import {
  createWebGpuSeedSearchRunner,
  isWebGpuSeedSearchSupported,
  type WebGpuSeedSearchRunner,
} from '@/lib/webgpu/seed-search/runner';
import type { WebGpuRunnerCallbacks, WebGpuRunnerProgress, WebGpuSearchContext } from '@/lib/webgpu/seed-search/types';
import type { InitialSeedResult, SearchConditions } from '@/types/search';

const hasWebGpu = isWebGpuSeedSearchSupported();
const describeWebGpu = hasWebGpu ? describe : describe.skip;

interface CpuBaselineEntry {
  seed: number;
  timer0: number;
  vcount: number;
  datetime: Date;
}

const calculator = new SeedCalculator();

function enumerateCpuBaseline(conditions: SearchConditions, context: WebGpuSearchContext): CpuBaselineEntry[] {
  const entries: CpuBaselineEntry[] = [];

  for (const segment of context.segments) {
    const rangeSeconds = Math.max(1, segment.rangeSeconds);
    for (let timer0 = segment.timer0Min; timer0 <= segment.timer0Max; timer0 += 1) {
      for (let secondOffset = 0; secondOffset < rangeSeconds; secondOffset += 1) {
        const datetime = new Date(context.startTimestampMs + secondOffset * 1000);
  const message = calculator.generateMessage(conditions, timer0, segment.vcount, datetime, segment.keyCode);
        const { seed } = calculator.calculateSeed(message);
        entries.push({ seed, timer0, vcount: segment.vcount, datetime });
      }
    }
  }

  return entries;
}

describeWebGpu('webgpu seed search runner integration', () => {
  let runner: WebGpuSeedSearchRunner;

  beforeAll(async () => {
    runner = createWebGpuSeedSearchRunner();
    await runner.init();
  });

  afterAll(() => {
    runner?.dispose();
  });

  it(
    'discovers target seeds and reports progress using production search context',
    async () => {
      const conditions: SearchConditions = {
        romVersion: 'W2',
        romRegion: 'JPN',
        hardware: 'DS',
        timer0VCountConfig: {
          useAutoConfiguration: false,
          timer0Range: { min: 0x10f4, max: 0x10f7 },
          vcountRange: { min: 0x82, max: 0x82 },
        },
        dateRange: {
          startYear: 2012,
          endYear: 2012,
          startMonth: 6,
          endMonth: 6,
          startDay: 12,
          endDay: 12,
          startHour: 10,
          endHour: 10,
          startMinute: 15,
          endMinute: 15,
          startSecond: 0,
          endSecond: 5,
        },
        keyInput: 0x0000,
        macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
      };

      const context = buildSearchContext(conditions);
      expect(context.totalMessages).toBeGreaterThan(0);

      const baseline = enumerateCpuBaseline(conditions, context);
      expect(baseline.length).toBe(context.totalMessages);

      const uniqueBySeed = new Map<number, CpuBaselineEntry>();
      for (const entry of baseline) {
        if (!uniqueBySeed.has(entry.seed)) {
          uniqueBySeed.set(entry.seed, entry);
        }
      }

      const uniqueEntries = Array.from(uniqueBySeed.values());
      expect(uniqueEntries.length).toBeGreaterThan(0);

      let targetEntries = uniqueEntries.filter((_, index) => index % 2 === 0);
      if (targetEntries.length === uniqueEntries.length && uniqueEntries.length > 1) {
        targetEntries = targetEntries.slice(0, uniqueEntries.length - 1);
      }
      if (targetEntries.length === 0) {
        targetEntries = [uniqueEntries[0]!];
      }

      const targetSeeds = targetEntries.map((entry) => entry.seed);
      const expectedBySeed = new Map<number, CpuBaselineEntry>(targetEntries.map((entry) => [entry.seed, entry]));
      expect(targetSeeds.length).toBeGreaterThan(0);
      expect(targetSeeds.length).toBeLessThanOrEqual(uniqueEntries.length);

      const results: InitialSeedResult[] = [];
      const progressEvents: WebGpuRunnerProgress[] = [];
      const completionMessages: string[] = [];
      const errors: Array<{ message: string; code?: string }> = [];
      const stopped: Array<{ message: string; progress: WebGpuRunnerProgress }> = [];

      const callbacks: WebGpuRunnerCallbacks = {
        onProgress: (progress) => {
          progressEvents.push(progress);
        },
        onResult: (result) => {
          results.push(result);
        },
        onComplete: (message) => {
          completionMessages.push(message);
        },
        onError: (message, errorCode) => {
          errors.push({ message, code: errorCode });
        },
        onPaused: () => {
          // intentionally empty
        },
        onResumed: () => {
          // intentionally empty
        },
        onStopped: (message, progress) => {
          stopped.push({ message, progress });
        },
      };

      await runner.run({
        context,
        targetSeeds,
        callbacks,
      });

      expect(errors).toHaveLength(0);
      expect(stopped).toHaveLength(0);
      expect(completionMessages).toHaveLength(1);
      expect(results.length).toBe(targetSeeds.length);

      const actualSeedsSorted = [...results].map((result) => result.seed).sort((a, b) => a - b);
      const expectedSeedsSorted = [...targetSeeds].sort((a, b) => a - b);
      expect(actualSeedsSorted).toEqual(expectedSeedsSorted);

      for (const result of results) {
        const expected = expectedBySeed.get(result.seed);
        expect(expected).toBeDefined();
        if (!expected) {
          continue;
        }
        expect(result.timer0).toBe(expected.timer0);
        expect(result.vcount).toBe(expected.vcount);
        expect(result.datetime.getTime()).toBe(expected.datetime.getTime());
        expect(result.isMatch).toBe(true);
      }

      expect(progressEvents.length).toBeGreaterThan(0);
      const finalProgress = progressEvents[progressEvents.length - 1]!;
      expect(finalProgress.currentStep).toBe(context.totalMessages);
      expect(finalProgress.totalSteps).toBe(context.totalMessages);
      expect(finalProgress.matchesFound).toBe(results.length);
    },
    60_000
  );
});
