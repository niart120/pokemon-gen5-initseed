/**
 * MT Seed Search パフォーマンス比較テスト
 *
 * CPU Worker と GPU Worker のスループットを計測・比較する。
 *
 * 期待スループット:
 * - GPU: ~100M/s
 * - CPU (32並列): ~40M/s
 * - CPU (単一Worker): ~1-5M/s
 */
import { describe, it, expect, afterEach } from 'vitest';
import type { MtSeedSearchCompletion } from '@/types/mt-seed-search';
import {
  createCpuWorker,
  createGpuWorker,
  waitForMessage,
  createTestJob,
  IV_CODE_6V,
  getIvCodes4V,
  calculateThroughput,
  formatNumber,
  isWebGpuAvailable,
  type PerformanceResult,
} from './helpers/test-helpers';

// テスト設定
const SEARCH_CONFIGS = [
  { range: 1_000_000, label: '1M', cpuTimeoutMs: 30_000, gpuTimeoutMs: 30_000 },
  { range: 10_000_000, label: '10M', cpuTimeoutMs: 60_000, gpuTimeoutMs: 30_000 },
  { range: 100_000_000, label: '100M', cpuTimeoutMs: 180_000, gpuTimeoutMs: 60_000 },
];

// IV コード設定
const IV_CODE_CONFIGS = [
  { codes: [IV_CODE_6V] as number[], label: '1 code (6V)' },
  { codes: getIvCodes4V(), label: '1024 codes (4V)' },
];

// パフォーマンス閾値
const THRESHOLDS = {
  cpu: {
    minimum: 1_000_000,    // 1M/s - 単一Workerの最低基準
    expected: 10_000_000,  // 10M/s - 期待値（単一Worker）
  },
  gpu: {
    minimum: 50_000_000,   // 50M/s - 最低基準
    expected: 100_000_000, // 100M/s - 期待値
  },
};

describe('MT Seed Search Performance', () => {
  const results: PerformanceResult[] = [];
  let currentWorker: Worker | null = null;

  afterEach(() => {
    if (currentWorker) {
      currentWorker.terminate();
      currentWorker = null;
    }
  });

  describe('CPU Worker スループット', () => {
    for (const { range, label, cpuTimeoutMs } of SEARCH_CONFIGS) {
      it(
        `${label} レンジのスループット計測`,
        async () => {
          const worker = createCpuWorker();
          currentWorker = worker;

          await waitForMessage(worker, (msg) => msg.type === 'READY', 10000);

          const job = createTestJob({
            start: 0,
            end: range - 1,
            ivCodes: [IV_CODE_6V],
            mtAdvances: 0,
          });

          const startTime = performance.now();
          worker.postMessage({ type: 'START', job });

          const complete = await waitForMessage(
            worker,
            (msg) => msg.type === 'COMPLETE',
            cpuTimeoutMs
          );

          const elapsedMs = performance.now() - startTime;

          expect(complete.type).toBe('COMPLETE');
          if (complete.type === 'COMPLETE') {
            const payload = complete.payload as MtSeedSearchCompletion;
            const throughput = calculateThroughput(range, elapsedMs);

            results.push({
              mode: 'cpu',
              searchRange: range,
              elapsedMs,
              seedsPerSecond: throughput,
              matchesFound: payload.totalMatches,
            });

            console.log(
              `[CPU] ${formatNumber(range)}: ` +
                `${(elapsedMs / 1000).toFixed(2)}s, ` +
                `${formatNumber(Math.round(throughput))} seeds/sec`
            );

            expect(throughput).toBeGreaterThan(THRESHOLDS.cpu.minimum);
          }
        },
        240000
      );
    }
  });

  const hasWebGpu = isWebGpuAvailable();
  const describeGpu = hasWebGpu ? describe : describe.skip;

  describeGpu('GPU Worker スループット', () => {
    for (const { range, label, gpuTimeoutMs } of SEARCH_CONFIGS) {
      it(
        `${label} レンジのスループット計測`,
        async () => {
          const worker = createGpuWorker();
          currentWorker = worker;

          await waitForMessage(worker, (msg) => msg.type === 'READY', 10000);

          const job = createTestJob({
            start: 0,
            end: range - 1,
            ivCodes: [IV_CODE_6V],
            mtAdvances: 0,
          });

          const startTime = performance.now();
          worker.postMessage({ type: 'START', job });

          const complete = await waitForMessage(
            worker,
            (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
            gpuTimeoutMs
          );

          const elapsedMs = performance.now() - startTime;

          if (complete.type === 'ERROR') {
            console.warn(`GPU Worker error: ${complete.message}`);
            return;
          }

          expect(complete.type).toBe('COMPLETE');
          if (complete.type === 'COMPLETE') {
            const payload = complete.payload as MtSeedSearchCompletion;
            const throughput = calculateThroughput(range, elapsedMs);

            results.push({
              mode: 'gpu',
              searchRange: range,
              elapsedMs,
              seedsPerSecond: throughput,
              matchesFound: payload.totalMatches,
            });

            console.log(
              `[GPU] ${formatNumber(range)}: ` +
                `${(elapsedMs / 1000).toFixed(2)}s, ` +
                `${formatNumber(Math.round(throughput))} seeds/sec`
            );

            expect(throughput).toBeGreaterThan(THRESHOLDS.gpu.minimum);
          }
        },
        120000
      );
    }
  });

  describe('IVコード数によるパフォーマンス変化', () => {
    const testRange = 10_000_000;

    for (const { codes, label } of IV_CODE_CONFIGS) {
      it(
        `CPU: ${label} でのスループット`,
        async () => {
          const worker = createCpuWorker();
          currentWorker = worker;

          await waitForMessage(worker, (msg) => msg.type === 'READY', 10000);

          const job = createTestJob({
            start: 0,
            end: testRange - 1,
            ivCodes: codes,
            mtAdvances: 0,
          });

          const startTime = performance.now();
          worker.postMessage({ type: 'START', job });

          const complete = await waitForMessage(
            worker,
            (msg) => msg.type === 'COMPLETE',
            120000
          );

          const elapsedMs = performance.now() - startTime;

          if (complete.type === 'COMPLETE') {
            const payload = complete.payload as MtSeedSearchCompletion;
            const throughput = calculateThroughput(testRange, elapsedMs);

            console.log(
              `[CPU] ${label}: ` +
                `${(elapsedMs / 1000).toFixed(2)}s, ` +
                `${formatNumber(Math.round(throughput))} seeds/sec, ` +
                `${payload.totalMatches} matches`
            );

            expect(throughput).toBeGreaterThan(THRESHOLDS.cpu.minimum);
          }
        },
        180000
      );
    }

    if (hasWebGpu) {
      for (const { codes, label } of IV_CODE_CONFIGS) {
        it(
          `GPU: ${label} でのスループット`,
          async () => {
            const worker = createGpuWorker();
            currentWorker = worker;

            await waitForMessage(worker, (msg) => msg.type === 'READY', 10000);

            const job = createTestJob({
              start: 0,
              end: testRange - 1,
              ivCodes: codes,
              mtAdvances: 0,
            });

            const startTime = performance.now();
            worker.postMessage({ type: 'START', job });

            const complete = await waitForMessage(
              worker,
              (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
              60000
            );

            const elapsedMs = performance.now() - startTime;

            if (complete.type === 'ERROR') {
              console.warn(`GPU Worker error: ${complete.message}`);
              return;
            }

            if (complete.type === 'COMPLETE') {
              const payload = complete.payload as MtSeedSearchCompletion;
              const throughput = calculateThroughput(testRange, elapsedMs);

              console.log(
                `[GPU] ${label}: ` +
                  `${(elapsedMs / 1000).toFixed(2)}s, ` +
                  `${formatNumber(Math.round(throughput))} seeds/sec, ` +
                  `${payload.totalMatches} matches`
              );

              expect(throughput).toBeGreaterThan(THRESHOLDS.gpu.minimum);
            }
          },
          120000
        );
      }
    }
  });

  describe('MT消費数によるパフォーマンス変化', () => {
    const testRange = 10_000_000;
    const mtAdvancesConfigs = [0, 10, 100];

    for (const mtAdvances of mtAdvancesConfigs) {
      it(
        `CPU: mtAdvances=${mtAdvances} でのスループット`,
        async () => {
          const worker = createCpuWorker();
          currentWorker = worker;

          await waitForMessage(worker, (msg) => msg.type === 'READY', 10000);

          const job = createTestJob({
            start: 0,
            end: testRange - 1,
            ivCodes: [IV_CODE_6V],
            mtAdvances,
          });

          const startTime = performance.now();
          worker.postMessage({ type: 'START', job });

          const complete = await waitForMessage(
            worker,
            (msg) => msg.type === 'COMPLETE',
            120000
          );

          const elapsedMs = performance.now() - startTime;

          if (complete.type === 'COMPLETE') {
            const throughput = calculateThroughput(testRange, elapsedMs);

            console.log(
              `[CPU] mtAdvances=${mtAdvances}: ` +
                `${(elapsedMs / 1000).toFixed(2)}s, ` +
                `${formatNumber(Math.round(throughput))} seeds/sec`
            );

            expect(throughput).toBeGreaterThan(THRESHOLDS.cpu.minimum);
          }
        },
        180000
      );
    }
  });
});
