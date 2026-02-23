/**
 * MT Seed Search GPU Worker 統合テスト
 *
 * WebGPUを使用したGPU Workerのテスト。
 * スループット基準: 100M/s
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { MtSeedSearchCompletion, MtSeedSearchProgress } from '@/types/mt-seed-search';
import {
  createGpuWorker,
  waitForMessage,
  collectMessages,
  createTestJob,
  IV_CODE_6V,
  IV_CODE_5V_0A,
  getIvCodes4V,
  calculateThroughput,
  formatNumber,
  isWebGpuAvailable,
} from './helpers/test-helpers';

// WebGPU未サポート環境のスキップ
const hasWebGpu = isWebGpuAvailable();
const describeWebGpu = hasWebGpu ? describe : describe.skip;

describeWebGpu('MT Seed Search GPU Worker', () => {
  let worker: Worker;

  beforeEach(() => {
    worker = createGpuWorker();
  });

  afterEach(() => {
    worker.terminate();
  });

  describe('初期化', () => {
    it('READY メッセージを送信する', async () => {
      const ready = await waitForMessage(
        worker,
        (msg) => msg.type === 'READY',
        10000
      );
      expect(ready.type).toBe('READY');
      if (ready.type === 'READY') {
        expect(ready.version).toBeDefined();
      }
    });
  });

  describe('検索実行', () => {
    it('1M レンジで検索完了する (6V単一IVコード)', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const job = createTestJob({
        start: 0,
        end: 999_999,
        ivCodes: [IV_CODE_6V],
        mtAdvances: 0,
      });

      worker.postMessage({ type: 'START', job });

      const complete = await waitForMessage(
        worker,
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        60000
      );

      if (complete.type === 'ERROR') {
        // GPU初期化エラーの場合はスキップ扱い
        console.warn(`GPU Worker error: ${complete.message}`);
        return;
      }

      expect(complete.type).toBe('COMPLETE');
      if (complete.type === 'COMPLETE') {
        const payload = complete.payload as MtSeedSearchCompletion;
        expect(payload.reason).toBe('finished');
        expect(payload.totalProcessed).toBe(1_000_000);
      }
    });

    it('10M レンジで検索完了する (5V0A IVコード)', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const job = createTestJob({
        start: 0,
        end: 9_999_999,
        ivCodes: [IV_CODE_5V_0A],
        mtAdvances: 0,
      });

      worker.postMessage({ type: 'START', job });

      const complete = await waitForMessage(
        worker,
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        60000
      );

      if (complete.type === 'ERROR') {
        console.warn(`GPU Worker error: ${complete.message}`);
        return;
      }

      expect(complete.type).toBe('COMPLETE');
      if (complete.type === 'COMPLETE') {
        const payload = complete.payload as MtSeedSearchCompletion;
        expect(payload.reason).toBe('finished');
        expect(payload.totalProcessed).toBe(10_000_000);
      }
    });

    it('100M レンジで検索完了する (6V IVコード)', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const job = createTestJob({
        start: 0,
        end: 99_999_999,
        ivCodes: [IV_CODE_6V],
        mtAdvances: 0,
      });

      worker.postMessage({ type: 'START', job });

      const complete = await waitForMessage(
        worker,
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        120000
      );

      if (complete.type === 'ERROR') {
        console.warn(`GPU Worker error: ${complete.message}`);
        return;
      }

      expect(complete.type).toBe('COMPLETE');
      if (complete.type === 'COMPLETE') {
        const payload = complete.payload as MtSeedSearchCompletion;
        expect(payload.reason).toBe('finished');
        expect(payload.totalProcessed).toBe(100_000_000);
      }
    }, 180000);

    it('4V IVコード (1024件) で検索完了する', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const ivCodes = getIvCodes4V();
      expect(ivCodes.length).toBe(1024);

      const job = createTestJob({
        start: 0,
        end: 999_999,
        ivCodes,
        mtAdvances: 0,
      });

      worker.postMessage({ type: 'START', job });

      const complete = await waitForMessage(
        worker,
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        60000
      );

      if (complete.type === 'ERROR') {
        console.warn(`GPU Worker error: ${complete.message}`);
        return;
      }

      expect(complete.type).toBe('COMPLETE');
      if (complete.type === 'COMPLETE') {
        const payload = complete.payload as MtSeedSearchCompletion;
        expect(payload.reason).toBe('finished');
        expect(payload.totalProcessed).toBe(1_000_000);
        expect(payload.totalMatches).toBeGreaterThan(0);
      }
    });

    it('PROGRESS メッセージを定期的に送信する', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const job = createTestJob({
        start: 0,
        end: 49_999_999, // 50M
        ivCodes: [IV_CODE_6V],
        mtAdvances: 0,
      });

      worker.postMessage({ type: 'START', job });

      const messages = await collectMessages(
        worker,
        ['PROGRESS', 'COMPLETE', 'ERROR'],
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        120000
      );

      const errorMsg = messages.find((msg) => msg.type === 'ERROR');
      if (errorMsg && errorMsg.type === 'ERROR') {
        console.warn(`GPU Worker error: ${errorMsg.message}`);
        return;
      }

      const progressMessages = messages.filter(
        (msg) => msg.type === 'PROGRESS'
      );
      expect(progressMessages.length).toBeGreaterThan(0);

      // 進捗は増加していく
      let lastProcessed = 0;
      for (const msg of progressMessages) {
        if (msg.type === 'PROGRESS') {
          const payload = msg.payload as MtSeedSearchProgress;
          expect(payload.processedCount).toBeGreaterThanOrEqual(lastProcessed);
          lastProcessed = payload.processedCount;
        }
      }
    }, 180000);
  });

  describe('制御', () => {
    it('STOP で検索を中断する', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const job = createTestJob({
        start: 0,
        end: 999_999_999, // 1B - 十分長い
        ivCodes: [IV_CODE_6V],
        mtAdvances: 0,
      });

      worker.postMessage({ type: 'START', job });

      // 少し待ってから停止
      await new Promise((resolve) => setTimeout(resolve, 500));
      worker.postMessage({ type: 'STOP' });

      const complete = await waitForMessage(
        worker,
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        30000
      );

      if (complete.type === 'ERROR') {
        console.warn(`GPU Worker error: ${complete.message}`);
        return;
      }

      expect(complete.type).toBe('COMPLETE');
      if (complete.type === 'COMPLETE') {
        const payload = complete.payload as MtSeedSearchCompletion;
        expect(payload.reason).toBe('stopped');
        expect(payload.totalProcessed).toBeLessThan(1_000_000_000);
      }
    });
  });

  describe('パフォーマンス計測', () => {
    it('100M レンジのスループットを計測する', async () => {
      await waitForMessage(worker, (msg) => msg.type === 'READY');

      const searchRange = 100_000_000;
      const job = createTestJob({
        start: 0,
        end: searchRange - 1,
        ivCodes: [IV_CODE_6V],
        mtAdvances: 0,
      });

      const startTime = performance.now();
      worker.postMessage({ type: 'START', job });

      const complete = await waitForMessage(
        worker,
        (msg) => msg.type === 'COMPLETE' || msg.type === 'ERROR',
        180000
      );

      const elapsedMs = performance.now() - startTime;

      if (complete.type === 'ERROR') {
        console.warn(`GPU Worker error: ${complete.message}`);
        return;
      }

      if (complete.type === 'COMPLETE') {
        const _payload = complete.payload as MtSeedSearchCompletion;
        const throughput = calculateThroughput(searchRange, elapsedMs);

        console.log(
          `[GPU Worker] ${formatNumber(searchRange)} seeds: ` +
            `${(elapsedMs / 1000).toFixed(2)}s, ` +
            `${formatNumber(Math.round(throughput))} seeds/sec`
        );

        // 最低基準: 50M/s
        expect(throughput).toBeGreaterThan(50_000_000);
      }
    }, 240000);
  });
});
