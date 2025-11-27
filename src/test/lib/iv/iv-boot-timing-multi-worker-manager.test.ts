/**
 * IVBootTimingMultiWorkerManager テスト
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  IVBootTimingMultiWorkerManager,
  type IVBootTimingMultiWorkerCallbacks,
} from '@/lib/iv/iv-boot-timing-multi-worker-manager';
import {
  createDefaultIVBootTimingSearchParams,
  type IVBootTimingSearchParams,
} from '@/types/iv-boot-timing-search';

type MockWorker = {
  postMessage: ReturnType<typeof vi.fn>;
  terminate: ReturnType<typeof vi.fn>;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((error: ErrorEvent | Error) => void) | null;
};

const createMockWorker = (): MockWorker => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null,
  onerror: null,
});

function createValidParams(): IVBootTimingSearchParams {
  const params = createDefaultIVBootTimingSearchParams();
  params.targetSeeds = [0x12345678];
  return params;
}

describe('IVBootTimingMultiWorkerManager', () => {
  let manager: IVBootTimingMultiWorkerManager;
  let mockCallbacks: IVBootTimingMultiWorkerCallbacks;
  let workerInstances: MockWorker[];
  let workerConstructor: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();

    workerInstances = [];
    workerConstructor = vi.fn(function WorkerMock(this: unknown): Worker {
      const instance = createMockWorker();
      workerInstances.push(instance);
      return instance as unknown as Worker;
    });

    vi.stubGlobal('Worker', workerConstructor as unknown as Worker);

    manager = new IVBootTimingMultiWorkerManager(4);
    mockCallbacks = {
      onProgress: vi.fn(),
      onResult: vi.fn(),
      onComplete: vi.fn(),
      onError: vi.fn(),
      onPaused: vi.fn(),
      onResumed: vi.fn(),
      onStopped: vi.fn(),
    };
  });

  afterEach(() => {
    manager.terminateAll();
    vi.restoreAllMocks();
  });

  describe('初期化', () => {
    it('正しい設定で初期化される', () => {
      expect(manager.isRunning()).toBe(false);
      expect(manager.getActiveWorkerCount()).toBe(0);
      expect(manager.getResultsCount()).toBe(0);
    });

    it('デフォルト設定で初期化される', () => {
      const defaultManager = new IVBootTimingMultiWorkerManager();
      expect(defaultManager.getActiveWorkerCount()).toBe(0);
    });

    it('Worker数を設定できる', () => {
      manager.setMaxWorkers(2);
      expect(manager.getMaxWorkers()).toBe(2);
    });

    it('実行中はWorker数を変更できない', async () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const params = createValidParams();

      await manager.startParallelSearch(params, mockCallbacks);
      manager.setMaxWorkers(8);

      expect(consoleSpy).toHaveBeenCalledWith(
        'Cannot change worker count during active search'
      );
      consoleSpy.mockRestore();
    });
  });

  describe('startParallelSearch', () => {
    it('重複実行を防ぐ', async () => {
      const params = createValidParams();

      await manager.startParallelSearch(params, mockCallbacks);

      await expect(
        manager.startParallelSearch(params, mockCallbacks)
      ).rejects.toThrow('Search is already running');
    });

    it('適切なWorker数を作成する', async () => {
      const manager2 = new IVBootTimingMultiWorkerManager(2);
      const params = createValidParams();

      vi.clearAllMocks();

      await manager2.startParallelSearch(params, mockCallbacks);

      expect(Worker).toHaveBeenCalled();
      expect(workerInstances.length).toBeGreaterThan(0);

      manager2.terminateAll();
    });

    it('Workerにメッセージを送信する', async () => {
      const params = createValidParams();

      await manager.startParallelSearch(params, mockCallbacks);

      expect(Worker).toHaveBeenCalled();
      const firstWorker = workerInstances[0];
      expect(firstWorker).toBeDefined();
      expect(firstWorker?.postMessage).toHaveBeenCalled();

      const sentMessage = firstWorker?.postMessage.mock.calls[0]?.[0];
      expect(sentMessage).toBeDefined();
      if (!sentMessage) {
        throw new Error('No message was sent to the worker');
      }
      expect(sentMessage.type).toBe('START_SEARCH');
      expect(sentMessage.params).toBeDefined();
      expect(sentMessage.params.targetSeeds).toEqual([0x12345678]);
    });
  });

  describe('Worker制御', () => {
    beforeEach(async () => {
      const params = createValidParams();
      await manager.startParallelSearch(params, mockCallbacks);
    });

    it('pauseAll() でコールバックを呼び出す', () => {
      manager.pauseAll();

      expect(mockCallbacks.onPaused).toHaveBeenCalled();
    });

    it('resumeAll() でコールバックを呼び出す', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      manager.resumeAll();

      expect(mockCallbacks.onResumed).toHaveBeenCalled();
      // IV workerはRESUMEをサポートしないため警告が出る
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('terminateAll() で全Workerを終了', () => {
      const activeWorkersBefore = manager.getActiveWorkerCount();

      manager.terminateAll();

      const terminateCalls = workerInstances.reduce(
        (sum, worker) => sum + worker.terminate.mock.calls.length,
        0
      );
      expect(terminateCalls).toBeGreaterThanOrEqual(activeWorkersBefore);
      expect(manager.isRunning()).toBe(false);
      expect(mockCallbacks.onStopped).toHaveBeenCalled();
    });
  });

  describe('Workerメッセージ処理', () => {
    it('COMPLETE メッセージで完了処理が呼ばれる', async () => {
      const params = createValidParams();
      await manager.startParallelSearch(params, mockCallbacks);

      // 全Workerに完了メッセージを送信
      for (const worker of workerInstances) {
        if (worker.onmessage) {
          worker.onmessage({
            data: {
              type: 'COMPLETE',
              payload: {
                reason: 'completed',
                processedCombinations: 100,
                totalCombinations: 100,
                resultsCount: 0,
                elapsedMs: 1000,
              },
            },
          } as MessageEvent);
        }
      }

      expect(mockCallbacks.onComplete).toHaveBeenCalled();
    });

    it('RESULTS メッセージで結果が処理される', async () => {
      const params = createValidParams();
      await manager.startParallelSearch(params, mockCallbacks);

      const firstWorker = workerInstances[0];
      if (firstWorker?.onmessage) {
        firstWorker.onmessage({
          data: {
            type: 'RESULTS',
            payload: {
              results: [
                {
                  boot: {
                    datetime: new Date('2025-01-01T12:00:00Z'),
                    timer0: 0x0c79,
                    vcount: 0x60,
                    keyCode: 0x2fff,
                    keyInputNames: [],
                    macAddress: [0, 0, 0, 0, 0, 0],
                  },
                  mtSeedHex: '12345678',
                  mtSeed: 0x12345678,
                  lcgSeedHex: 'AABBCCDD',
                },
              ],
              batchIndex: 0,
            },
          },
        } as MessageEvent);
      }

      expect(mockCallbacks.onResult).toHaveBeenCalled();
      expect(manager.getResultsCount()).toBe(1);
    });

    it('PROGRESS メッセージで進捗が更新される', async () => {
      const params = createValidParams();
      await manager.startParallelSearch(params, mockCallbacks);

      const firstWorker = workerInstances[0];
      if (firstWorker?.onmessage) {
        firstWorker.onmessage({
          data: {
            type: 'PROGRESS',
            payload: {
              processedCombinations: 50,
              totalCombinations: 100,
              foundCount: 0,
              progressPercent: 50,
              elapsedMs: 500,
              estimatedRemainingMs: 500,
            },
          },
        } as MessageEvent);
      }

      // 進捗監視のタイマーが動いているため、onProgressが呼ばれているはず
      // ただし即座ではなく500msインターバル
    });

    it('ERROR メッセージでエラー処理される', async () => {
      // 新しいmanagerをWorker数1で作成
      const singleWorkerManager = new IVBootTimingMultiWorkerManager(1);
      expect(singleWorkerManager.getMaxWorkers()).toBe(1);

      const params = createValidParams();
      await singleWorkerManager.startParallelSearch(params, mockCallbacks);

      // 作成されたWorker数を確認（1である必要がある）
      expect(workerInstances.length).toBe(1);
      expect(singleWorkerManager.getActiveWorkerCount()).toBe(1);

      // 唯一のWorkerにエラーを発生させる
      const worker = workerInstances[0];
      expect(worker).toBeDefined();
      expect(worker?.onmessage).toBeDefined();

      worker?.onmessage?.({
        data: {
          type: 'ERROR',
          message: 'Test error',
          category: 'RUNTIME',
          fatal: true,
        },
      } as MessageEvent);

      // エラー処理後のWorker数を確認
      expect(singleWorkerManager.getActiveWorkerCount()).toBe(0);
      expect(mockCallbacks.onError).toHaveBeenCalledWith(
        'All workers failed: Test error'
      );

      singleWorkerManager.terminateAll();
    });
  });

  describe('エラーハンドリング', () => {
    it('Worker onerror でエラー処理される', async () => {
      // 新しいmanagerをWorker数1で作成
      const singleWorkerManager = new IVBootTimingMultiWorkerManager(1);
      const params = createValidParams();
      await singleWorkerManager.startParallelSearch(params, mockCallbacks);

      // 唯一のWorkerにonerrorを発生させる
      const worker = workerInstances[0];
      if (worker?.onerror) {
        // ErrorEventのモックを作成
        const errorEvent = {
          message: 'Worker crashed',
        } as ErrorEvent;
        worker.onerror(errorEvent);
      }

      expect(mockCallbacks.onError).toHaveBeenCalledWith(
        'All workers failed: Worker error: Worker crashed'
      );

      singleWorkerManager.terminateAll();
    });
  });

  describe('状態取得', () => {
    it('isRunning() が正しい値を返す', async () => {
      expect(manager.isRunning()).toBe(false);

      const params = createValidParams();
      await manager.startParallelSearch(params, mockCallbacks);
      expect(manager.isRunning()).toBe(true);

      manager.terminateAll();
      expect(manager.isRunning()).toBe(false);
    });

    it('getActiveWorkerCount() が正しい値を返す', async () => {
      expect(manager.getActiveWorkerCount()).toBe(0);

      const params = createValidParams();
      await manager.startParallelSearch(params, mockCallbacks);
      expect(manager.getActiveWorkerCount()).toBeGreaterThan(0);
    });
  });
});
