/**
 * MT Seed Search CPU Worker 統合テスト
 *
 * WASM SIMD最適化版MT19937を使用したCPU Workerのテスト。
 * スループット基準: 40M/s (32並列時)
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { MtSeedSearchCompletion, MtSeedSearchProgress } from '@/types/mt-seed-search';
import {
  createCpuWorker,
  waitForMessage,
  collectMessages,
  createTestJob,
  IV_CODE_6V,
  IV_CODE_5V_0A,
  getIvCodes4V,
  calculateThroughput,
  formatNumber,
} from './helpers/test-helpers';

// Worker未サポート環境のスキップ
if (typeof Worker === 'undefined') {
  describe.skip('MT Seed Search CPU Worker (no Worker support)', () => {
    it('skipped', () => {
      expect(true).toBe(true);
    });
  });
} else {
  describe('MT Seed Search CPU Worker', () => {
    let worker: Worker;

    beforeEach(() => {
      worker = createCpuWorker();
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
          (msg) => msg.type === 'COMPLETE',
          30000
        );

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
          (msg) => msg.type === 'COMPLETE',
          60000
        );

        expect(complete.type).toBe('COMPLETE');
        if (complete.type === 'COMPLETE') {
          const payload = complete.payload as MtSeedSearchCompletion;
          expect(payload.reason).toBe('finished');
          expect(payload.totalProcessed).toBe(10_000_000);
        }
      });

      it('4V IVコード (1024件) で検索完了する', async () => {
        await waitForMessage(worker, (msg) => msg.type === 'READY');

        const ivCodes = getIvCodes4V();
        expect(ivCodes.length).toBe(1024); // 32 * 32 = 1024

        const job = createTestJob({
          start: 0,
          end: 999_999,
          ivCodes,
          mtAdvances: 0,
        });

        worker.postMessage({ type: 'START', job });

        const complete = await waitForMessage(
          worker,
          (msg) => msg.type === 'COMPLETE',
          60000
        );

        expect(complete.type).toBe('COMPLETE');
        if (complete.type === 'COMPLETE') {
          const payload = complete.payload as MtSeedSearchCompletion;
          expect(payload.reason).toBe('finished');
          expect(payload.totalProcessed).toBe(1_000_000);
          // 4Vは条件が緩いためマッチ数が多いはず
          expect(payload.totalMatches).toBeGreaterThan(0);
        }
      });

      it('PROGRESS メッセージを定期的に送信する', async () => {
        await waitForMessage(worker, (msg) => msg.type === 'READY');

        const job = createTestJob({
          start: 0,
          end: 4_999_999, // 5M
          ivCodes: [IV_CODE_6V],
          mtAdvances: 0,
        });

        worker.postMessage({ type: 'START', job });

        const messages = await collectMessages(
          worker,
          ['PROGRESS', 'COMPLETE'],
          (msg) => msg.type === 'COMPLETE',
          60000
        );

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
      });
    });

    describe('制御', () => {
      it('STOP で検索を中断する', async () => {
        await waitForMessage(worker, (msg) => msg.type === 'READY');

        const job = createTestJob({
          start: 0,
          end: 99_999_999, // 100M - 十分長い
          ivCodes: [IV_CODE_6V],
          mtAdvances: 0,
        });

        worker.postMessage({ type: 'START', job });

        // 少し待ってから停止
        await new Promise((resolve) => setTimeout(resolve, 100));
        worker.postMessage({ type: 'STOP' });

        const complete = await waitForMessage(
          worker,
          (msg) => msg.type === 'COMPLETE',
          30000
        );

        expect(complete.type).toBe('COMPLETE');
        if (complete.type === 'COMPLETE') {
          const payload = complete.payload as MtSeedSearchCompletion;
          expect(payload.reason).toBe('stopped');
          // 全範囲を処理していないはず
          expect(payload.totalProcessed).toBeLessThan(100_000_000);
        }
      });
    });

    describe('パフォーマンス計測', () => {
      it('10M レンジのスループットを計測する', async () => {
        await waitForMessage(worker, (msg) => msg.type === 'READY');

        const searchRange = 10_000_000;
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
          (msg) => msg.type === 'COMPLETE',
          120000
        );

        const elapsedMs = performance.now() - startTime;

        if (complete.type === 'COMPLETE') {
          const _payload = complete.payload as MtSeedSearchCompletion;
          const throughput = calculateThroughput(searchRange, elapsedMs);

          console.log(
            `[CPU Worker] ${formatNumber(searchRange)} seeds: ` +
              `${(elapsedMs / 1000).toFixed(2)}s, ` +
              `${formatNumber(Math.round(throughput))} seeds/sec`
          );

          // 最低基準: 1M/s (単一Worker)
          expect(throughput).toBeGreaterThan(1_000_000);
        }
      });
    });
  });
}
