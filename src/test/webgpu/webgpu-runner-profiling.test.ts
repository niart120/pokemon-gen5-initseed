/// <reference types="@webgpu/types" />

import { beforeAll, describe, expect, it } from 'vitest';
import { prepareSearchJob } from '@/lib/webgpu/seed-search/prepare-search-job';
import { createSeedSearchController } from '@/lib/webgpu/seed-search/seed-search-controller';
import {
  createWebGpuDeviceContext,
  isWebGpuSeedSearchSupported,
  type SeedSearchLimitPreferences,
  type WebGpuDeviceContext,
  type SeedSearchJobLimits,
} from '@/lib/webgpu/utils';
import { createSeedSearchEngine, type SeedSearchEngineObserver } from '@/lib/webgpu/seed-search/seed-search-engine';
import type {
  SeedSearchJobSegment,
  WebGpuRunnerCallbacks,
} from '@/lib/webgpu/seed-search/types';
import { enumerateJobCpuBaseline, pickUniqueEntries } from './job-baseline-helpers';
import type { InitialSeedResult, SearchConditions } from '@/types/search';

const hasWebGpu = isWebGpuSeedSearchSupported();
const describeWebGpu = hasWebGpu ? describe : describe.skip;

function selectTargetSeeds(
  conditions: SearchConditions,
  limits: SeedSearchJobLimits,
  sampleCount: number
): number[] {
  if (sampleCount <= 0) {
    return [];
  }
  const baselineJob = prepareSearchJob(conditions, undefined, { limits });
  const baseline = enumerateJobCpuBaseline(conditions, baselineJob, sampleCount * 4);
  const unique = pickUniqueEntries(baseline);
  return unique.slice(0, sampleCount).map((entry) => entry.seed);
}

function buildSearchConditions(rangeSeconds: number, timer0Range: { min: number; max: number }): SearchConditions {
  const baseStart = new Date(2012, 5, 12, 10, 15, 0);
  const secondsPerDayBlock = Math.max(1, Math.min(rangeSeconds, 24 * 60 * 60));
  const dayCount = Math.max(1, Math.ceil(rangeSeconds / secondsPerDayBlock));

  const perDayStart = secondsPerDayBlock >= 24 * 60 * 60
    ? new Date(baseStart.getFullYear(), baseStart.getMonth(), baseStart.getDate(), 0, 0, 0)
    : baseStart;
  const perDayEnd = new Date(perDayStart.getTime() + Math.max(0, secondsPerDayBlock - 1) * 1000);
  const endDateForRange = new Date(perDayStart.getTime() + (dayCount - 1) * 24 * 60 * 60 * 1000);

  return {
    romVersion: 'W2',
    romRegion: 'JPN',
    hardware: 'DS',
    keyInput: 0x0000,
    macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: { min: timer0Range.min, max: timer0Range.max },
      vcountRange: { min: 0x60, max: 0x60 },
    },
    dateRange: {
      startYear: perDayStart.getFullYear(),
      endYear: endDateForRange.getFullYear(),
      startMonth: perDayStart.getMonth() + 1,
      endMonth: endDateForRange.getMonth() + 1,
      startDay: perDayStart.getDate(),
      endDay: endDateForRange.getDate(),
      startHour: perDayStart.getHours(),
      endHour: perDayStart.getHours(),
      startMinute: perDayStart.getMinutes(),
      endMinute: perDayStart.getMinutes(),
      startSecond: perDayStart.getSeconds(),
      endSecond: perDayStart.getSeconds(),
    },
    timeRange: {
      hour: { start: perDayStart.getHours(), end: perDayEnd.getHours() },
      minute: { start: perDayStart.getMinutes(), end: perDayEnd.getMinutes() },
      second: { start: perDayStart.getSeconds(), end: perDayEnd.getSeconds() },
    },
  };
}


interface ProgressSnapshot {
  eventIndex: number;
  timestampMs: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  currentStep: number;
  matchesFound: number;
}

interface ProfilingCollector {
  results: InitialSeedResult[];
  progressSteps: number[];
  timeline: ProgressSnapshot[];
}

interface EngineEnsureEvent {
  timestampMs: number;
  pipelineRecreated: boolean;
  workgroupSize: number;
  candidateCapacity: number;
}

type EngineBufferKind = 'config' | 'match-output' | 'match-readback' | 'target' | 'uniform';

interface EngineBufferEvent {
  timestampMs: number;
  kind: EngineBufferKind;
  sizeBytes: number;
}

interface EngineDispatchEvent {
  timestampMs: number;
  segmentId: string;
  messageCount: number;
  workgroupCount: number;
  matchCount: number;
  candidateCapacity: number;
  timings: {
    totalMs: number;
    setupMs: number;
    gpuMs: number;
    readbackMs: number;
  };
}

interface EngineProfilingStats {
  ensures: EngineEnsureEvent[];
  buffers: EngineBufferEvent[];
  dispatches: EngineDispatchEvent[];
}

interface ProfilingSummaryRow {
  scenario: string;
  workgroupSize: number;
  dispatchConcurrency: number;
  effectiveDispatchLimit: number;
  totalMessages: number;
  totalSegments: number;
  totalWorkgroups: number;
  targetSeeds: number;
  runtimeMs: number;
  throughputPerSecond: number;
  matchesFound: number;
  progressEvents: number;
  segmentMinMessages: number;
  segmentMaxMessages: number;
  segmentAvgMessages: number;
  pipelineReconfigurations: number;
  configBufferRebuilds: number;
  matchBufferRebuilds: number;
  readbackBufferRebuilds: number;
  targetBufferRebuilds: number;
  uniformBufferRebuilds: number;
  dispatchTotalMs: number;
  dispatchSetupMs: number;
  dispatchGpuMs: number;
  dispatchReadbackMs: number;
  dispatchCount: number;
}

const nowMs = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function createProfilingCollector(): ProfilingCollector {
  return {
    results: [],
    progressSteps: [],
    timeline: [],
  };
}

function createEngineProfilingStats(): EngineProfilingStats {
  return {
    ensures: [],
    buffers: [],
    dispatches: [],
  };
}

function createEngineObserver(stats: EngineProfilingStats): SeedSearchEngineObserver {
  return {
    onEnsureConfigured: (event) => {
      stats.ensures.push({
        timestampMs: event.timestampMs,
        pipelineRecreated: event.pipelineRecreated,
        workgroupSize: event.workgroupSize,
        candidateCapacity: event.candidateCapacity,
      });
    },
    onBufferRecreated: (event) => {
      stats.buffers.push({
        timestampMs: event.timestampMs,
        kind: event.kind,
        sizeBytes: event.sizeBytes,
      });
    },
    onDispatchComplete: (event) => {
      stats.dispatches.push({
        timestampMs: event.timestampMs,
        segmentId: event.segmentId,
        messageCount: event.messageCount,
        workgroupCount: event.workgroupCount,
        matchCount: event.matchCount,
        candidateCapacity: event.candidateCapacity,
        timings: event.timings,
      });
    },
  };
}

interface SegmentStats {
  minMessages: number;
  maxMessages: number;
  avgMessages: number;
}

function summarizeSegmentStats(segments: readonly SeedSearchJobSegment[]): SegmentStats {
  if (segments.length === 0) {
    return { minMessages: 0, maxMessages: 0, avgMessages: 0 };
  }
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  let total = 0;
  for (const segment of segments) {
    const count = segment.messageCount;
    if (count < min) {
      min = count;
    }
    if (count > max) {
      max = count;
    }
    total += count;
  }
  return {
    minMessages: min,
    maxMessages: max,
    avgMessages: total / segments.length,
  };
}

interface EngineStatsSummary {
  pipelineReconfigurations: number;
  configBufferRebuilds: number;
  matchBufferRebuilds: number;
  readbackBufferRebuilds: number;
  targetBufferRebuilds: number;
  uniformBufferRebuilds: number;
  dispatchTotalMs: number;
  dispatchSetupMs: number;
  dispatchGpuMs: number;
  dispatchReadbackMs: number;
  dispatchCount: number;
}

function summarizeEngineStats(stats: EngineProfilingStats): EngineStatsSummary {
  const pipelineReconfigurations = stats.ensures.filter((event) => event.pipelineRecreated).length;
  const configBufferRebuilds = stats.buffers.filter((event) => event.kind === 'config').length;
  const matchBufferRebuilds = stats.buffers.filter((event) => event.kind === 'match-output').length;
  const readbackBufferRebuilds = stats.buffers.filter((event) => event.kind === 'match-readback').length;
  const targetBufferRebuilds = stats.buffers.filter((event) => event.kind === 'target').length;
  const uniformBufferRebuilds = stats.buffers.filter((event) => event.kind === 'uniform').length;
  const dispatchTotals = stats.dispatches.reduce(
    (acc, dispatch) => {
      acc.dispatchTotalMs += dispatch.timings.totalMs;
      acc.dispatchSetupMs += dispatch.timings.setupMs;
      acc.dispatchGpuMs += dispatch.timings.gpuMs;
      acc.dispatchReadbackMs += dispatch.timings.readbackMs;
      return acc;
    },
    {
      dispatchTotalMs: 0,
      dispatchSetupMs: 0,
      dispatchGpuMs: 0,
      dispatchReadbackMs: 0,
    }
  );

  return {
    pipelineReconfigurations,
    configBufferRebuilds,
    matchBufferRebuilds,
    readbackBufferRebuilds,
    targetBufferRebuilds,
    uniformBufferRebuilds,
    dispatchTotalMs: dispatchTotals.dispatchTotalMs,
    dispatchSetupMs: dispatchTotals.dispatchSetupMs,
    dispatchGpuMs: dispatchTotals.dispatchGpuMs,
    dispatchReadbackMs: dispatchTotals.dispatchReadbackMs,
    dispatchCount: stats.dispatches.length,
  };
}

function buildCallbacks(collector: ProfilingCollector): WebGpuRunnerCallbacks {
  return {
    onProgress: (progress) => {
      collector.progressSteps.push(progress.currentStep);
      collector.timeline.push({
        eventIndex: collector.timeline.length,
        timestampMs: nowMs(),
        elapsedTime: progress.elapsedTime,
        estimatedTimeRemaining: progress.estimatedTimeRemaining,
        currentStep: progress.currentStep,
        matchesFound: progress.matchesFound,
      });
    },
    onResult: (result) => {
      collector.results.push(result);
    },
    onComplete: () => {
      // noop
    },
    onError: (message: string) => {
      throw new Error(message);
    },
    onPaused: () => {
      // noop
    },
    onResumed: () => {
      // noop
    },
    onStopped: (message: string) => {
      throw new Error(message);
    },
  };
}

function expectMonotonicSeries(series: number[]): void {
  for (let i = 1; i < series.length; i += 1) {
    expect(series[i]).toBeGreaterThanOrEqual(series[i - 1]);
  }
}

function recordProfilingSummary(
  summaries: ProfilingSummaryRow[],
  params: {
    scenario: string;
    workgroupSize: number;
    dispatchConcurrency: number;
    effectiveDispatchLimit: number;
    jobMessages: number;
    segments: number;
    workgroups: number;
    targetSeeds: number;
    runtimeMs: number;
    matchesFound: number;
    progressEvents: number;
    segmentStats: SegmentStats;
    engineStats: EngineStatsSummary;
  }
): void {
  const throughput = params.runtimeMs > 0 ? params.jobMessages / (params.runtimeMs / 1000) : 0;
  summaries.push({
    scenario: params.scenario,
    workgroupSize: params.workgroupSize,
    dispatchConcurrency: params.dispatchConcurrency,
    effectiveDispatchLimit: params.effectiveDispatchLimit,
    totalMessages: params.jobMessages,
    totalSegments: params.segments,
    totalWorkgroups: params.workgroups,
    targetSeeds: params.targetSeeds,
    runtimeMs: params.runtimeMs,
    throughputPerSecond: throughput,
    matchesFound: params.matchesFound,
    progressEvents: params.progressEvents,
    segmentMinMessages: params.segmentStats.minMessages,
    segmentMaxMessages: params.segmentStats.maxMessages,
    segmentAvgMessages: params.segmentStats.avgMessages,
    pipelineReconfigurations: params.engineStats.pipelineReconfigurations,
    configBufferRebuilds: params.engineStats.configBufferRebuilds,
    matchBufferRebuilds: params.engineStats.matchBufferRebuilds,
    readbackBufferRebuilds: params.engineStats.readbackBufferRebuilds,
    targetBufferRebuilds: params.engineStats.targetBufferRebuilds,
    uniformBufferRebuilds: params.engineStats.uniformBufferRebuilds,
    dispatchTotalMs: params.engineStats.dispatchTotalMs,
    dispatchSetupMs: params.engineStats.dispatchSetupMs,
    dispatchGpuMs: params.engineStats.dispatchGpuMs,
    dispatchReadbackMs: params.engineStats.dispatchReadbackMs,
    dispatchCount: params.engineStats.dispatchCount,
  });
}

function outputProfilingSummaries(rows: ProfilingSummaryRow[]): void {
  if (rows.length === 0 || typeof console === 'undefined') {
    return;
  }

  const formattedRows = rows.map((row) => ({
    scenario: row.scenario,
    workgroup: row.workgroupSize,
    dispatchPref: row.dispatchConcurrency,
    dispatchLimit: row.effectiveDispatchLimit,
    runtimeMs: row.runtimeMs.toFixed(1),
    throughput: row.throughputPerSecond.toFixed(1),
    messages: row.totalMessages,
    segments: row.totalSegments,
    workgroups: row.totalWorkgroups,
    matches: row.matchesFound,
    progressEvents: row.progressEvents,
    targets: row.targetSeeds,
    segmentMin: row.segmentMinMessages,
    segmentMax: row.segmentMaxMessages,
    segmentAvg: row.segmentAvgMessages.toFixed(1),
    pipelineResets: row.pipelineReconfigurations,
    configBuf: row.configBufferRebuilds,
    matchBuf: row.matchBufferRebuilds,
    readbackBuf: row.readbackBufferRebuilds,
    targetBuf: row.targetBufferRebuilds,
    uniformBuf: row.uniformBufferRebuilds,
    dispatchCount: row.dispatchCount,
    dispatchTotalMs: row.dispatchTotalMs.toFixed(1),
    dispatchGpuMs: row.dispatchGpuMs.toFixed(1),
    dispatchReadbackMs: row.dispatchReadbackMs.toFixed(1),
  }));

  if (typeof console.table === 'function') {
    console.table(formattedRows);
  }

  for (const row of formattedRows) {
    console.log(
      `[profiling] scenario=${row.scenario} workgroup=${row.workgroup} ` +
        `dispatchPref=${row.dispatchPref} dispatchLimit=${row.dispatchLimit} ` +
        `runtimeMs=${row.runtimeMs} throughput=${row.throughput} ` +
        `messages=${row.messages} segments=${row.segments} workgroups=${row.workgroups} ` +
        `segment[min/max/avg]=${row.segmentMin}/${row.segmentMax}/${row.segmentAvg} ` +
        `pipelineResets=${row.pipelineResets} buf(config/match/readback/target/uniform)=${row.configBuf}/${row.matchBuf}/${row.readbackBuf}/${row.targetBuf}/${row.uniformBuf} ` +
        `dispatches=${row.dispatchCount} dispatchMs(total/gpu/readback)=${row.dispatchTotalMs}/${row.dispatchGpuMs}/${row.dispatchReadbackMs} ` +
        `matches=${row.matches} progressEvents=${row.progressEvents} targets=${row.targets}`
    );
  }
}

const WORKGROUP_CONFIGS = [256] as const;
const DISPATCH_CONCURRENCY_CONFIGS = [1, 4] as const;

interface ScenarioConfig {
  label: string;
  rangeSeconds: number;
  timer0Range: { min: number; max: number };
  sampleSeeds: number;
  limitOverrides?: SeedSearchLimitPreferences;
}

function buildScenarioLimitPreferences(
  scenario: ScenarioConfig,
  workgroupSize: number,
  dispatchConcurrency: number
): SeedSearchLimitPreferences {
  const overrides = scenario.limitOverrides;
  const base: SeedSearchLimitPreferences = {
    workgroupSize,
    maxDispatchesInFlight: dispatchConcurrency,
  };
  if (!overrides) {
    return base;
  }
  return {
    ...base,
    ...overrides,
    workgroupSize: overrides.workgroupSize ?? workgroupSize,
    maxDispatchesInFlight: overrides.maxDispatchesInFlight ?? dispatchConcurrency,
  };
}

const SECONDS_IN_YEAR = 365 * 24 * 60 * 60;

const SCENARIOS: readonly ScenarioConfig[] = [
  {
    label: 'baseline-10year',
    rangeSeconds: SECONDS_IN_YEAR * 10,
    timer0Range: { min: 0x10a0, max: 0x10a0 },
    sampleSeeds: 8,
  },
  {
    label: 'high-50year',
    rangeSeconds: SECONDS_IN_YEAR * 50,
    timer0Range: { min: 0x10a0, max: 0x10a0 },
    sampleSeeds: 8,
  },
  {
    label: 'ultra-100year',
    rangeSeconds: SECONDS_IN_YEAR * 100,
    timer0Range: { min: 0x10a0, max: 0x10a0 },
    sampleSeeds: 8,
  },
];

describeWebGpu('webgpu seed search profiling instrumentation', () => {
  let deviceContext: WebGpuDeviceContext;

  beforeAll(async () => {
    deviceContext = await createWebGpuDeviceContext();
  });

  it(
    'collects controller-level progress across scenarios',
    async () => {
      const profilingSummaries: ProfilingSummaryRow[] = [];
      for (const scenario of SCENARIOS) {
        for (const dispatchConcurrency of DISPATCH_CONCURRENCY_CONFIGS) {
          for (const workgroupSize of WORKGROUP_CONFIGS) {
            const engineStats = createEngineProfilingStats();
            const engine = createSeedSearchEngine(createEngineObserver(engineStats), deviceContext);
            const controller = createSeedSearchController(engine);
            const conditions = buildSearchConditions(scenario.rangeSeconds, scenario.timer0Range);
            const limitPreferences = buildScenarioLimitPreferences(
              scenario,
              workgroupSize,
              dispatchConcurrency
            );
            const jobLimits = deviceContext.deriveSearchJobLimits(limitPreferences);
            const targetSeeds = selectTargetSeeds(conditions, jobLimits, scenario.sampleSeeds);
            expect(targetSeeds.length).toBeGreaterThan(0);

            const job = prepareSearchJob(conditions, targetSeeds, { limits: jobLimits });

            const collector = createProfilingCollector();
            const callbacks = buildCallbacks(collector);

            const startTimestamp = nowMs();
            try {
              await controller.run(job, callbacks);
            } finally {
              engine.dispose();
            }
            const runtimeMs = nowMs() - startTimestamp;

            const { results, progressSteps, timeline } = collector;

            expect(results.length).toBeGreaterThanOrEqual(targetSeeds.length);
            const seedSet = new Set(results.map((result) => result.seed));
            for (const seed of targetSeeds) {
              expect(seedSet.has(seed)).toBe(true);
            }

            expect(progressSteps.length).toBeGreaterThan(0);
            expect(progressSteps[progressSteps.length - 1]).toBe(job.summary.totalMessages);

            expect(timeline.length).toBe(progressSteps.length);
            expect(timeline.length).toBeGreaterThan(0);
            expect(timeline[0]?.currentStep).toBe(0);
            const lastSnapshot = timeline[timeline.length - 1]!;
            expect(lastSnapshot.currentStep).toBe(job.summary.totalMessages);
            expect(lastSnapshot.estimatedTimeRemaining).toBe(0);
            expect(lastSnapshot.matchesFound).toBe(results.length);
            expect(lastSnapshot.elapsedTime).toBeGreaterThan(0);

            expectMonotonicSeries(progressSteps);
            expectMonotonicSeries(timeline.map((entry) => entry.timestampMs));
            expectMonotonicSeries(timeline.map((entry) => entry.elapsedTime));
            expectMonotonicSeries(timeline.map((entry) => entry.matchesFound));

            const elapsedDelta = Math.abs(lastSnapshot.elapsedTime - runtimeMs);
            const allowedDelta = Math.max(1500, runtimeMs * 0.25);
            expect(elapsedDelta).toBeLessThanOrEqual(allowedDelta);

            const totalWorkgroups = job.segments.reduce((sum, segment) => sum + segment.workgroupCount, 0);
            const segmentStats = summarizeSegmentStats(job.segments);
            const engineSummary = summarizeEngineStats(engineStats);
            recordProfilingSummary(profilingSummaries, {
              scenario: scenario.label,
              workgroupSize,
              dispatchConcurrency,
              effectiveDispatchLimit: jobLimits.maxDispatchesInFlight,
              jobMessages: job.summary.totalMessages,
              segments: job.summary.totalSegments,
              workgroups: totalWorkgroups,
              targetSeeds: targetSeeds.length,
              runtimeMs,
              matchesFound: results.length,
              progressEvents: progressSteps.length,
              segmentStats,
              engineStats: engineSummary,
            });
          }
        }
      }

      outputProfilingSummaries(profilingSummaries);
      expect(profilingSummaries).toHaveLength(
        SCENARIOS.length * WORKGROUP_CONFIGS.length * DISPATCH_CONCURRENCY_CONFIGS.length
      );
    },
    120_000
  );
});
