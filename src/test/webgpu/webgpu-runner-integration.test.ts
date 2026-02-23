import { beforeAll, describe, expect, it } from 'vitest';
import { prepareSearchJob } from '@/lib/webgpu/seed-search/prepare-search-job';
import { createSeedSearchController } from '@/lib/webgpu/seed-search/seed-search-controller';
import type { SeedSearchController } from '@/lib/webgpu/seed-search/seed-search-controller';
import { isWebGpuSeedSearchSupported } from '@/lib/webgpu/utils';
import type { WebGpuRunnerCallbacks, WebGpuRunnerProgress } from '@/lib/webgpu/seed-search/types';
import { enumerateJobCpuBaseline, pickUniqueEntries } from './job-baseline-helpers';
import type { CpuBaselineEntry } from './job-baseline-helpers';
import type { InitialSeedResult, SearchConditions } from '@/types/search';
import { createTestSeedSearchJobLimits } from './seed-search-job-limit-helpers';

const hasWebGpu = isWebGpuSeedSearchSupported();
const describeWebGpu = hasWebGpu ? describe : describe.skip;
const TEST_LIMITS = createTestSeedSearchJobLimits({
  workgroupSize: 64,
  maxWorkgroupsPerDispatch: 2048,
  candidateCapacityPerDispatch: 4096,
});

describeWebGpu('webgpu seed search runner integration', () => {
  let controller: SeedSearchController;

  beforeAll(() => {
    controller = createSeedSearchController();
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
        },
        timeRange: {
          hour: { start: 10, end: 10 },
          minute: { start: 15, end: 15 },
          second: { start: 0, end: 5 },
        },
        keyInput: 0x0000,
        macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
      };

      const baselineJob = prepareSearchJob(conditions, undefined, { limits: TEST_LIMITS });
      expect(baselineJob.summary.totalMessages).toBeGreaterThan(0);

      const baseline = enumerateJobCpuBaseline(conditions, baselineJob);
      expect(baseline.length).toBe(baselineJob.summary.totalMessages);

      const uniqueEntries = pickUniqueEntries(baseline);
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

      const job = prepareSearchJob(conditions, targetSeeds, { limits: TEST_LIMITS });
      expect(job.summary.totalMessages).toBe(baselineJob.summary.totalMessages);

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

      await controller.run(job, callbacks);

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
      expect(finalProgress.currentStep).toBe(job.summary.totalMessages);
      expect(finalProgress.totalSteps).toBe(job.summary.totalMessages);
      expect(finalProgress.matchesFound).toBe(results.length);
    },
    60_000
  );
});
