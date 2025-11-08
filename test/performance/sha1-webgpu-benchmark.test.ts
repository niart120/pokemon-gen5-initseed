/// <reference types="@webgpu/types" />

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { ChunkCalculator } from '@/lib/search/chunk-calculator';
import type { SearchConditions } from '@/types/search';
import {
  WORDS_PER_HASH,
  benchmarkWorkload,
  buildGpuWorkloadConfig,
  createWorkload,
  createWorkloadMessages,
  hashesEqual,
  runWasmHashes,
  runWasmHashesStreaming,
} from '@/test-utils/perf/sha1-webgpu-harness';
import {
  WebGpuSha1Runner,
  type WebGpuSha1BatchDetail,
  type WebGpuSha1ProfilingSample,
} from '@/test-utils/webgpu/webgpu-sha1-runner';

const hasWebGpu = typeof navigator !== 'undefined' && navigator.gpu !== undefined && navigator.gpu !== null;
const describeWebGpu = hasWebGpu ? describe : describe.skip;

const WASM_ITERATIONS = 1;
const GPU_ITERATIONS = 5;
const TEST_TIMEOUT_MS = 120_000;
type OverrideKey = 'GPU_BATCH_LIMIT' | 'GPU_WORKGROUP_SIZE' | 'STRESS_WINDOW' | 'WORKER_COUNT';

function readOverride(key: OverrideKey): string | null {
  const globalCandidates = [`__SHA1_${key}__`, `VITE_SHA1_${key}`];
  for (const candidate of globalCandidates) {
    const value = (globalThis as Record<string, unknown>)[candidate];
    if (typeof value === 'string' && value.trim().length > 0) {
      return value;
    }
  }

  if (typeof import.meta !== 'undefined') {
    const env = (import.meta as { env?: Record<string, unknown> }).env;
    if (env) {
      for (const candidate of globalCandidates) {
        const value = env[candidate];
        if (typeof value === 'string' && value.trim().length > 0) {
          return value;
        }
      }
    }
  }

  if (typeof process !== 'undefined' && typeof process.env !== 'undefined') {
    for (const candidate of globalCandidates) {
      const value = process.env[candidate];
      if (typeof value === 'string' && value.trim().length > 0) {
        return value;
      }
    }
  }

  return null;
}

function readNumericOverride(key: OverrideKey): number | null {
  const raw = readOverride(key);
  if (!raw) {
    return null;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

const GPU_BATCH_LIMIT_OVERRIDE = readNumericOverride('GPU_BATCH_LIMIT');
const GPU_WORKGROUP_SIZE_OVERRIDE = readNumericOverride('GPU_WORKGROUP_SIZE');
const STRESS_WINDOW_OVERRIDE = readNumericOverride('STRESS_WINDOW');
const WORKER_COUNT_OVERRIDE = readNumericOverride('WORKER_COUNT');

// Default host-side memory ceiling to keep result buffers within a manageable size.
const HOST_MEMORY_BUDGET_BYTES = 96 * 1024 * 1024;
const HOST_MEMORY_MESSAGE_LIMIT = Math.max(
  1,
  Math.floor(HOST_MEMORY_BUDGET_BYTES / (WORDS_PER_HASH * Uint32Array.BYTES_PER_ELEMENT))
);

const ACCURACY_CONFIG = {
  rangeSeconds: 256,
  timer0Range: { min: 0x10a0, max: 0x10a0 },
  vcountRange: { min: 0x60, max: 0x60 },
} as const;

const STRESS_TIMER0_RANGE = { min: 0xc67, max: 0xc69 } as const;
const STRESS_VCOUNT_RANGE = { min: 0x5f, max: 0x5f } as const;

function resolveStressWindowSeconds(): number {
  if (Number.isFinite(STRESS_WINDOW_OVERRIDE) && (STRESS_WINDOW_OVERRIDE as number) > 0) {
    return STRESS_WINDOW_OVERRIDE as number;
  }
  return 24 * 60 * 60;
}

function resolveStreamBatchLimit(): number | null {
  if (Number.isFinite(GPU_BATCH_LIMIT_OVERRIDE) && (GPU_BATCH_LIMIT_OVERRIDE as number) > 0) {
    return GPU_BATCH_LIMIT_OVERRIDE as number;
  }
  return null;
}

function determineWorkerCount(): number {
  const override = Number.isFinite(WORKER_COUNT_OVERRIDE) && (WORKER_COUNT_OVERRIDE as number) > 0
    ? (WORKER_COUNT_OVERRIDE as number)
    : null;
  const hardwareConcurrency =
    typeof navigator !== 'undefined' && Number.isFinite(navigator.hardwareConcurrency)
      ? Math.max(1, Math.floor(navigator.hardwareConcurrency))
      : null;

  if (override !== null) {
    if (hardwareConcurrency !== null && override > hardwareConcurrency) {
      console.info(
        `[sha1] worker override ${override} exceeds hardware concurrency ${hardwareConcurrency}, clamping to hardware limit`
      );
      return hardwareConcurrency;
    }
    return override;
  }

  if (hardwareConcurrency !== null) {
    return hardwareConcurrency;
  }

  return 4;
}

function buildProductionSearchConditions(totalWindowSeconds: number): SearchConditions {
  const searchStart = new Date(Date.UTC(2000, 0, 1, 0, 0, 0));
  const searchEnd = new Date(searchStart.getTime() + (totalWindowSeconds - 1) * 1000);

  return {
    romVersion: 'W',
    romRegion: 'JPN',
    hardware: 'DS',
    keyInput: 0x0000,
    macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: { ...STRESS_TIMER0_RANGE },
      vcountRange: { ...STRESS_VCOUNT_RANGE },
    },
    dateRange: {
      startYear: searchStart.getUTCFullYear(),
      endYear: searchEnd.getUTCFullYear(),
      startMonth: searchStart.getUTCMonth() + 1,
      endMonth: searchEnd.getUTCMonth() + 1,
      startDay: searchStart.getUTCDate(),
      endDay: searchEnd.getUTCDate(),
      startHour: searchStart.getUTCHours(),
      endHour: searchEnd.getUTCHours(),
      startMinute: searchStart.getUTCMinutes(),
      endMinute: searchEnd.getUTCMinutes(),
      startSecond: searchStart.getUTCSeconds(),
      endSecond: searchEnd.getUTCSeconds(),
    },
  };
}

function computeChunkRangeSeconds(workerCount: number, totalWindowSeconds: number): number {
  const conditions = buildProductionSearchConditions(totalWindowSeconds);
  const chunks = ChunkCalculator.calculateOptimalChunks(conditions, Math.max(1, workerCount));
  if (chunks.length === 0) {
    return totalWindowSeconds;
  }
  const firstChunk = chunks[0];
  const seconds = Math.floor((firstChunk.endDateTime.getTime() - firstChunk.startDateTime.getTime()) / 1000) + 1;
  return Math.max(1, Math.min(totalWindowSeconds, seconds));
}

type ProfilingAccumulator = WebGpuSha1ProfilingSample & { details: WebGpuSha1BatchDetail[] };

async function runGpuWorkloadStreaming(
  runner: WebGpuSha1Runner,
  context: ReturnType<typeof createWorkload>,
  batchSize: number
): Promise<WebGpuSha1ProfilingSample | null> {
  const workload = buildGpuWorkloadConfig(context);
  const hostSafeLimit = Math.max(1, Math.min(context.totalMessages, HOST_MEMORY_MESSAGE_LIMIT));
  const enforcedBatchSize = Math.max(1, Math.min(batchSize, hostSafeLimit));
  let accumulator: ProfilingAccumulator | null = null;

  for (let offset = 0; offset < context.totalMessages; ) {
    const remaining = context.totalMessages - offset;
    const messageCount = Math.min(enforcedBatchSize, remaining);
    await runner.computeGenerated(workload, offset, messageCount);
    const sample = runner.getLastProfiling();

    if (sample) {
      if (!accumulator) {
        accumulator = {
          ...sample,
          details: sample.details ? sample.details.map((detail) => ({ ...detail })) : [],
        };
      } else {
        accumulator.totalMs += sample.totalMs;
        accumulator.uploadMs += sample.uploadMs;
        accumulator.dispatchMs += sample.dispatchMs;
        accumulator.readbackMs += sample.readbackMs;
        accumulator.batches += sample.batches;
        accumulator.totalMessages += sample.totalMessages;
        accumulator.maxBatchMessages = Math.max(accumulator.maxBatchMessages, sample.maxBatchMessages);
        if (sample.details) {
          accumulator.details.push(...sample.details.map((detail) => ({ ...detail })));
        }
      }
    }

    offset += messageCount;
  }

  return accumulator;
}

describeWebGpu('WebGPU SHA-1 benchmark', () => {
  let runner: WebGpuSha1Runner;

  beforeAll(async () => {
    runner = new WebGpuSha1Runner();
    if (Number.isFinite(GPU_WORKGROUP_SIZE_OVERRIDE) && (GPU_WORKGROUP_SIZE_OVERRIDE as number) > 0) {
      await runner.setWorkgroupSize(GPU_WORKGROUP_SIZE_OVERRIDE as number);
      console.info(`[sha1] forcing GPU workgroup size to ${GPU_WORKGROUP_SIZE_OVERRIDE}`);
    }
    await runner.init();
    if (Number.isFinite(GPU_BATCH_LIMIT_OVERRIDE) && (GPU_BATCH_LIMIT_OVERRIDE as number) > 0) {
      runner.setMaxMessagesPerDispatch(GPU_BATCH_LIMIT_OVERRIDE as number);
      console.info(`[sha1] forcing GPU batch limit to ${GPU_BATCH_LIMIT_OVERRIDE}`);
    }
  });

  afterAll(() => {
    runner?.dispose();
  });

  it('matches TypeScript and WebAssembly baselines for an integrated workload', async () => {
    const context = createWorkload(ACCURACY_CONFIG);
    const messages = createWorkloadMessages(context);

    const [wasmHashes, gpuHashes] = await Promise.all([runWasmHashes(messages), runner.compute(messages)]);

    expect(hashesEqual(wasmHashes, gpuHashes)).toBe(true);
  });

  it(
    'collects timing metrics for a stress workload',
    async () => {
      const stressWindowSeconds = resolveStressWindowSeconds();
      const workerCount = determineWorkerCount();
      const chunkSeconds = computeChunkRangeSeconds(workerCount, stressWindowSeconds);
      const context = createWorkload({
        rangeSeconds: chunkSeconds,
        timer0Range: STRESS_TIMER0_RANGE,
        vcountRange: STRESS_VCOUNT_RANGE,
      });

      const batchLimitOverride = resolveStreamBatchLimit();
      const deviceCapacity = runner.getDispatchMessageCapacity();
      const hostLimitedBatch = Math.max(1, Math.min(context.totalMessages, HOST_MEMORY_MESSAGE_LIMIT));
      const defaultBatchLimit = hostLimitedBatch;
      const effectiveBatchLimit = batchLimitOverride ?? defaultBatchLimit;
      let streamBatchSize = Math.max(1, Math.min(effectiveBatchLimit, hostLimitedBatch));
      const shouldForceDoubleBuffer =
        batchLimitOverride == null && streamBatchSize <= deviceCapacity && hostLimitedBatch > deviceCapacity;
      if (shouldForceDoubleBuffer) {
        const doubleBufferedTarget = Math.max(deviceCapacity + 1, deviceCapacity * 2);
        streamBatchSize = Math.min(hostLimitedBatch, doubleBufferedTarget);
      }
      const totalBatches = Math.ceil(context.totalMessages / streamBatchSize);

      console.info(
        `[sha1] streaming pipeline configured (batch ${streamBatchSize} messages, window ${chunkSeconds}s, total batches ${totalBatches})`
      );
      console.info(
        `[sha1] stress workload derived from production chunk: workers=${workerCount}, window=${stressWindowSeconds}s, chunk=${chunkSeconds}s, messages=${context.totalMessages}`
      );

      const wasmStats = await benchmarkWorkload({
        runner: async () => {
          await runWasmHashesStreaming(context, streamBatchSize);
        },
        context,
        iterations: WASM_ITERATIONS,
        warmupIterations: 1,
      });

      let gpuStreamingProfiling: WebGpuSha1ProfilingSample | null = null;
      const gpuStats = await benchmarkWorkload({
        runner: async () => {
          gpuStreamingProfiling = await runGpuWorkloadStreaming(runner, context, streamBatchSize);
        },
        context,
        iterations: GPU_ITERATIONS,
        warmupIterations: 1,
      });

      const speedupVsWasm = wasmStats.averageMs === 0 ? 0 : wasmStats.averageMs / gpuStats.averageMs;
      console.info(
        `[sha1] WASM avg: ${wasmStats.averageMs.toFixed(3)}ms, WebGPU avg: ${gpuStats.averageMs.toFixed(3)}ms, speedup wasm x${speedupVsWasm.toFixed(2)}`
      );

      const gpuProfile = gpuStreamingProfiling ?? runner.getLastProfiling();
      if (gpuProfile) {
        const perMessage = gpuProfile.totalMessages === 0 ? 0 : gpuProfile.totalMs / gpuProfile.totalMessages;
        console.info(
          `[sha1-gpu-profile] total ${gpuProfile.totalMs.toFixed(3)}ms (upload ${gpuProfile.uploadMs.toFixed(3)}ms, dispatch ${
            gpuProfile.dispatchMs.toFixed(3)
          }ms, readback ${gpuProfile.readbackMs.toFixed(3)}ms) for ${gpuProfile.totalMessages} messages across ${
            gpuProfile.batches
          } batches (max batch ${gpuProfile.maxBatchMessages}, avg per message ${(perMessage * 1_000_000).toFixed(2)}ns)`
        );
        const batchDetails = gpuProfile.details ?? [];
        batchDetails.forEach((detail, index) => {
          const submitToGpuDone = detail.submitToGpuDoneMs === null ? 'n/a' : `${detail.submitToGpuDoneMs.toFixed(3)}ms`;
          const gpuDoneToMap = detail.gpuDoneToMapMs === null ? 'n/a' : `${detail.gpuDoneToMapMs.toFixed(3)}ms`;
          const gpuIdleBefore = detail.gpuIdleBeforeMs === null ? 'n/a' : `${detail.gpuIdleBeforeMs.toFixed(3)}ms`;
          console.info(
            `[sha1-gpu-batch] #${index} slot ${detail.slot} messages ${detail.messageCount} total ${detail.totalMs.toFixed(
              3
            )}ms (upload ${detail.uploadMs.toFixed(3)}ms, dispatch ${detail.dispatchMs.toFixed(3)}ms, readback ${detail.readbackMs.toFixed(
              3
            )}ms, submit ${detail.queueSubmitMs.toFixed(3)}ms, submit→GPUdone ${submitToGpuDone}, GPUdone→map ${gpuDoneToMap}, gpuIdleBefore ${gpuIdleBefore})`
          );
        });
      }

      expect(wasmStats.samples).toBeGreaterThan(0);
      expect(gpuStats.samples).toBeGreaterThan(0);
    },
    TEST_TIMEOUT_MS
  );
});

if (!hasWebGpu) {
  describe.skip('WebGPU SHA-1 benchmark', () => {
    it('skipped because WebGPU is not available in this environment', () => {
      expect(true).toBe(true);
    });
  });
}
