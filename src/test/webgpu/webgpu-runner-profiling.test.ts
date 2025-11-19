/// <reference types="@webgpu/types" />

import { describe, expect, it } from 'vitest';
import { SeedCalculator } from '@/lib/core/seed-calculator';
import { buildSearchContext } from '@/lib/webgpu/seed-search/message-encoder';
import { getDateFromTimePlan } from '@/lib/webgpu/seed-search/time-plan';
import { DEFAULT_HOST_MEMORY_LIMIT_BYTES, DOUBLE_BUFFER_SET_COUNT } from '@/lib/webgpu/seed-search/constants';
import {
  createWebGpuSeedSearchRunner,
  isWebGpuSeedSearchSupported,
  type WebGpuSeedSearchRunnerOptions,
} from '@/lib/webgpu/seed-search/runner';
import type {
  WebGpuRunnerInstrumentation,
  WebGpuRunnerSpanContext,
  WebGpuRunnerSpanKind,
  WebGpuSearchContext,
} from '@/lib/webgpu/seed-search/types';
import type { InitialSeedResult, SearchConditions } from '@/types/search';

const hasWebGpu = isWebGpuSeedSearchSupported();
const describeWebGpu = hasWebGpu ? describe : describe.skip;

interface SpanRecord {
  kind: WebGpuRunnerSpanKind;
  metadata: Record<string, unknown>;
  durationMs: number;
}

class CollectingInstrumentation implements WebGpuRunnerInstrumentation {
  public readonly spans: SpanRecord[] = [];

  async trace<T>(context: WebGpuRunnerSpanContext, operation: () => Promise<T>): Promise<T> {
    const start = performance.now();
    try {
      return await operation();
    } finally {
      const duration = performance.now() - start;
      this.spans.push({
        kind: context.kind,
        metadata: { ...context.metadata },
        durationMs: duration,
      });
    }
  }
}

const sampleSeedCalculator = new SeedCalculator();

function computeSampleSeeds(
  conditions: SearchConditions,
  context: WebGpuSearchContext,
  sampleCount: number
): number[] {
  if (sampleCount <= 0) {
    return [];
  }

  const seeds: number[] = [];

  outer: for (const segment of context.segments) {
    const rangeSeconds = Math.max(1, segment.rangeSeconds);
    for (let timer0 = segment.timer0Min; timer0 <= segment.timer0Max; timer0 += 1) {
      for (let offset = 0; offset < rangeSeconds; offset += 1) {
        const datetime = getDateFromTimePlan(context.timePlan, offset);
  const message = sampleSeedCalculator.generateMessage(conditions, timer0, segment.vcount, datetime, segment.keyCode);
        const { seed } = sampleSeedCalculator.calculateSeed(message);
        seeds.push(seed);
        if (seeds.length >= sampleCount) {
          break outer;
        }
      }
    }
  }

  return seeds;
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

function summarizeSpans(spans: SpanRecord[]) {
  const byKind = (kind: SpanRecord['kind']): SpanRecord[] => spans.filter((span) => span.kind === kind);
  const totalDuration = (kind: SpanRecord['kind']): number => byKind(kind).reduce((sum, span) => sum + span.durationMs, 0);
  const sumMetadata = (targetSpans: SpanRecord[], key: string): number =>
    targetSpans.reduce((sum, span) => sum + Number(span.metadata[key] ?? 0), 0);
  const maxMetadata = (targetSpans: SpanRecord[], key: string): number =>
    targetSpans.reduce((max, span) => Math.max(max, Number(span.metadata[key] ?? 0)), 0);
  const averageOf = (values: number[]): number => {
    if (values.length === 0) {
      return 0;
    }
    const total = values.reduce((sum, value) => sum + value, 0);
    return total / values.length;
  };

  const dispatchSpans = byKind('dispatch');
  const totalMessages = sumMetadata(dispatchSpans, 'messageCount');
  const batches = dispatchSpans.length;
  const maxBatchMessages = maxMetadata(dispatchSpans, 'messageCount');
  const avgBatchMessages = batches > 0 ? totalMessages / batches : 0;
  const maxCandidateCapacity = maxMetadata(dispatchSpans, 'candidateCapacity');
  const avgCandidateCapacity = averageOf(dispatchSpans.map((span) => Number(span.metadata.candidateCapacity ?? 0)));
  const maxWorkgroupCount = maxMetadata(dispatchSpans, 'workgroupCount');
  const avgWorkgroupCount = averageOf(dispatchSpans.map((span) => Number(span.metadata.workgroupCount ?? 0)));

  const dispatchMs = dispatchSpans.reduce((sum, span) => sum + span.durationMs, 0);
  const throughputPerSecond = dispatchMs > 0 ? (totalMessages / dispatchMs) * 1000 : 0;
  const avgPerMessageNs = totalMessages > 0 ? (dispatchMs / totalMessages) * 1_000_000 : 0;

  const copySpans = byKind('dispatch.copyResults');
  const totalCopyBytes = sumMetadata(copySpans, 'totalCopyBytes');
  const maxCopyBytes = maxMetadata(copySpans, 'totalCopyBytes');

  return {
    dispatch: {
      batches,
      totalMessages,
      maxBatchMessages,
      avgBatchMessages,
      maxCandidateCapacity,
      avgCandidateCapacity,
      maxWorkgroupCount,
      avgWorkgroupCount,
      dispatchMs,
      throughputPerSecond,
      avgPerMessageNs,
    },
    stages: {
      submitMs: totalDuration('dispatch.submit'),
      submitEncodeMs: totalDuration('dispatch.submit.encode'),
      submitWaitMs: totalDuration('dispatch.submit.wait'),
      mapCountMs: totalDuration('dispatch.mapMatchCount'),
      copyMs: totalDuration('dispatch.copyResults'),
      copyEncodeMs: totalDuration('dispatch.copyResults.encode'),
      copyWaitMs: totalDuration('dispatch.copyResults.wait'),
      mapResultsMs: totalDuration('dispatch.mapResults'),
      processMatchesMs: totalDuration('dispatch.processMatches'),
    },
    copyBytes: {
      totalBytes: totalCopyBytes,
      maxBytesPerDispatch: maxCopyBytes,
      avgBytesPerDispatch: copySpans.length > 0 ? totalCopyBytes / copySpans.length : 0,
    },
  } as const;
}

function createRunner(options: WebGpuSeedSearchRunnerOptions) {
  return createWebGpuSeedSearchRunner(options);
}

function buildCallbacks(results: InitialSeedResult[], progressSteps: number[]) {
  return {
    onProgress: (progress: { currentStep: number }) => {
      progressSteps.push(progress.currentStep);
    },
    onResult: (result: InitialSeedResult) => {
      results.push(result);
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

const WORKGROUP_CONFIGS = [
  { label: 'wg-64', workgroupSize: 64 },
  { label: 'wg-128', workgroupSize: 128 },
  { label: 'wg-256', workgroupSize: 256 },
] as const;

const DISPATCH_LIMITS = [
  { label: 'auto', maxMessagesPerDispatch: null },
  { label: 'limit-256', maxMessagesPerDispatch: 256 },
  { label: 'limit-2048', maxMessagesPerDispatch: 2048 },
] as const;

const BUFFER_SLOT_CONFIGS = [
  { label: 'slots-2', slots: 2 },
  { label: 'slots-4', slots: 4 },
] as const;

const HOST_MEMORY_CONFIGS = [
  { label: 'per-slot-default', perSlotMultiplier: 1 },
  { label: 'per-slot-x2', perSlotMultiplier: 2 },
] as const;

type DispatchLimitLabel = (typeof DISPATCH_LIMITS)[number]['label'];

interface ScenarioConfig {
  label: string;
  rangeSeconds: number;
  timer0Range: { min: number; max: number };
  targetSeeds?: readonly number[];
  targetSampleCount?: number;
  dispatchLimitLabels?: readonly DispatchLimitLabel[];
}

const SECONDS_IN_YEAR = 365 * 24 * 60 * 60;
const SECONDS_IN_50_YEARS = SECONDS_IN_YEAR * 50;

const SCENARIOS: readonly ScenarioConfig[] = [
  {
    label: 'baseline-600s',
    rangeSeconds: 600,
    timer0Range: { min: 0x10a0, max: 0x10a3 },
    targetSampleCount: 6,
  },
  {
    label: 'ultra-50years',
    rangeSeconds: SECONDS_IN_50_YEARS,
    timer0Range: { min: 0x10a0, max: 0x10a0 },
    targetSampleCount: 8,
    dispatchLimitLabels: ['auto'],
  },
];

describeWebGpu('webgpu seed search profiling instrumentation', () => {
  it(
    'collects instrumentation spans from the production runner',
    async () => {
      const defaultHostLimitPerSlot = DEFAULT_HOST_MEMORY_LIMIT_BYTES / DOUBLE_BUFFER_SET_COUNT;

      for (const scenario of SCENARIOS) {
        for (const workgroup of WORKGROUP_CONFIGS) {
          for (const bufferSlot of BUFFER_SLOT_CONFIGS) {
            const hostMemoryConfigs = bufferSlot.slots === 2 ? HOST_MEMORY_CONFIGS : [HOST_MEMORY_CONFIGS[0]];
            for (const hostMemory of hostMemoryConfigs) {
              for (const dispatch of DISPATCH_LIMITS) {
                if (scenario.dispatchLimitLabels && !scenario.dispatchLimitLabels.includes(dispatch.label)) {
                  continue;
                }

                const conditions = buildSearchConditions(scenario.rangeSeconds, scenario.timer0Range);
                const context = buildSearchContext(conditions);
                const sampleSeeds =
                  scenario.targetSeeds !== undefined
                    ? Array.from(new Set(scenario.targetSeeds))
                    : computeSampleSeeds(conditions, context, scenario.targetSampleCount ?? 0);
                const targetSeeds = sampleSeeds;
                expect(targetSeeds.length).toBeGreaterThan(0);

                const instrumentation = new CollectingInstrumentation();
                const hostMemoryLimitPerSlotBytes = defaultHostLimitPerSlot * hostMemory.perSlotMultiplier;
                const adjustedHostLimitBytes = Math.max(
                  DEFAULT_HOST_MEMORY_LIMIT_BYTES,
                  hostMemoryLimitPerSlotBytes * bufferSlot.slots
                );
                const runnerOptions: WebGpuSeedSearchRunnerOptions = {
                  workgroupSize: workgroup.workgroupSize,
                  instrumentation,
                  bufferSlots: bufferSlot.slots,
                  hostMemoryLimitBytes: adjustedHostLimitBytes,
                  hostMemoryLimitPerSlotBytes,
                };
                if (dispatch.maxMessagesPerDispatch !== null) {
                  runnerOptions.maxMessagesPerDispatch = dispatch.maxMessagesPerDispatch;
                }
                const runner = createRunner(runnerOptions);

                const results: InitialSeedResult[] = [];
                const progressSteps: number[] = [];
                const callbacks = buildCallbacks(results, progressSteps);

                await runner.run({
                  context,
                  targetSeeds,
                  callbacks,
                });
                runner.dispose();

                expect(results.length).toBeGreaterThanOrEqual(targetSeeds.length);
                const resultSeedSet = new Set(results.map((result) => result.seed));
                expect(resultSeedSet.size).toBe(targetSeeds.length);
                for (const seed of targetSeeds) {
                  expect(resultSeedSet.has(seed)).toBe(true);
                }

                const summary = summarizeSpans(instrumentation.spans);
                expect(summary.dispatch.batches).toBeGreaterThan(0);
                expect(summary.dispatch.totalMessages).toBe(context.totalMessages);
                if (dispatch.maxMessagesPerDispatch !== null) {
                  expect(summary.dispatch.maxBatchMessages).toBeLessThanOrEqual(dispatch.maxMessagesPerDispatch);
                } else {
                  expect(summary.dispatch.maxBatchMessages).toBeGreaterThan(0);
                  expect(summary.dispatch.maxBatchMessages).toBeLessThanOrEqual(context.totalMessages);
                }
                expect(summary.dispatch.dispatchMs).toBeGreaterThan(0);
                expect(summary.stages.submitMs).toBeGreaterThanOrEqual(0);
                expect(summary.stages.mapResultsMs).toBeGreaterThanOrEqual(0);

                console.info('[runner-profile] summary', {
                  scenario: scenario.label,
                  workgroup: workgroup.label,
                  dispatchLimit: dispatch.label,
                  dispatchLimitOverride: dispatch.maxMessagesPerDispatch,
                  bufferSlots: bufferSlot.slots,
                  hostMemoryConfig: hostMemory.label,
                  hostMemoryLimitPerSlotBytes,
                  targetSeedCount: targetSeeds.length,
                  batches: summary.dispatch.batches,
                  totalMessages: summary.dispatch.totalMessages,
                  maxBatchMessages: summary.dispatch.maxBatchMessages,
                  avgBatchMessages: Number(summary.dispatch.avgBatchMessages.toFixed(2)),
                  maxCandidateCapacity: summary.dispatch.maxCandidateCapacity,
                  avgCandidateCapacity: Number(summary.dispatch.avgCandidateCapacity.toFixed(2)),
                  maxWorkgroupCount: summary.dispatch.maxWorkgroupCount,
                  avgWorkgroupCount: Number(summary.dispatch.avgWorkgroupCount.toFixed(2)),
                  dispatchMs: Number(summary.dispatch.dispatchMs.toFixed(3)),
                  throughputPerSecond: Number(summary.dispatch.throughputPerSecond.toFixed(2)),
                  avgPerMessageNs: Number(summary.dispatch.avgPerMessageNs.toFixed(2)),
                  submitMs: Number(summary.stages.submitMs.toFixed(3)),
                  submitEncodeMs: Number(summary.stages.submitEncodeMs.toFixed(3)),
                  submitWaitMs: Number(summary.stages.submitWaitMs.toFixed(3)),
                  mapCountMs: Number(summary.stages.mapCountMs.toFixed(3)),
                  copyMs: Number(summary.stages.copyMs.toFixed(3)),
                  copyEncodeMs: Number(summary.stages.copyEncodeMs.toFixed(3)),
                  copyWaitMs: Number(summary.stages.copyWaitMs.toFixed(3)),
                  mapResultsMs: Number(summary.stages.mapResultsMs.toFixed(3)),
                  processMatchesMs: Number(summary.stages.processMatchesMs.toFixed(3)),
                  copyTotalBytes: summary.copyBytes.totalBytes,
                  copyMaxBytes: summary.copyBytes.maxBytesPerDispatch,
                  copyAvgBytes: Number(summary.copyBytes.avgBytesPerDispatch.toFixed(2)),
                });

                expect(progressSteps.length).toBeGreaterThan(0);
                expect(progressSteps[progressSteps.length - 1]).toBe(context.totalMessages);
              }
            }
          }
        }
      }
    },
    180_000
  );
});
