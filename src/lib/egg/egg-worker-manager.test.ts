import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EggWorkerManager } from './egg-worker-manager';
import {
  createDefaultEggParamsHex,
  hexParamsToEggParams,
  type EggWorkerResponse,
} from '@/types/egg';

describe('EggWorkerManager', () => {
  let mockPostMessage: ReturnType<typeof vi.fn>;
  let mockTerminate: ReturnType<typeof vi.fn>;
  let mockOnMessage: ((ev: MessageEvent) => void) | null;

  function createMockWorker() {
    mockPostMessage = vi.fn();
    mockTerminate = vi.fn();
    mockOnMessage = null;

    return {
      postMessage: mockPostMessage,
      terminate: mockTerminate,
      set onmessage(handler: ((ev: MessageEvent) => void) | null) {
        mockOnMessage = handler;
      },
      get onmessage() {
        return mockOnMessage;
      },
      onerror: null as ((ev: ErrorEvent) => void) | null,
    } as unknown as Worker;
  }

  function simulateWorkerMessage(msg: EggWorkerResponse) {
    if (mockOnMessage) {
      mockOnMessage({ data: msg } as MessageEvent);
    }
  }

  beforeEach(() => {
    mockPostMessage = vi.fn();
    mockTerminate = vi.fn();
    mockOnMessage = null;
  });

  it('should start and post START_GENERATION message', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const params = hexParamsToEggParams(createDefaultEggParamsHex());

    await manager.start(params);

    expect(mockPostMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'START_GENERATION' })
    );
    expect(manager.isRunning()).toBe(true);
    expect(manager.getStatus()).toBe('running');
  });

  it('should throw when starting while already running', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const params = hexParamsToEggParams(createDefaultEggParamsHex());

    await manager.start(params);

    expect(() => manager.start(params)).toThrow('egg generation already running');
  });

  it('should reject with validation errors', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const hex = createDefaultEggParamsHex();
    hex.count = 0; // invalid
    const params = hexParamsToEggParams(hex);

    await expect(manager.start(params)).rejects.toThrow('count');
  });

  it('should handle RESULTS callback', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const resultsCb = vi.fn();
    manager.onResults(resultsCb);

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    const resultsPayload = { results: [{ advance: 0, egg: {} as any, isStable: false }] };
    simulateWorkerMessage({ type: 'RESULTS', payload: resultsPayload });

    expect(resultsCb).toHaveBeenCalledWith(resultsPayload);
  });

  it('should handle COMPLETE callback and cleanup', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const completeCb = vi.fn();
    manager.onComplete(completeCb);

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    const completion = { reason: 'max-count' as const, processedCount: 100, filteredCount: 50, elapsedMs: 100 };
    simulateWorkerMessage({ type: 'COMPLETE', payload: completion });

    expect(completeCb).toHaveBeenCalledWith(completion);
    expect(manager.isRunning()).toBe(false);
    expect(manager.getStatus()).toBe('idle');
  });

  it('should handle ERROR callback', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const errorCb = vi.fn();
    manager.onError(errorCb);

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    simulateWorkerMessage({
      type: 'ERROR',
      message: 'test error',
      category: 'RUNTIME',
      fatal: false
    });

    expect(errorCb).toHaveBeenCalledWith('test error', 'RUNTIME', false);
    expect(manager.isRunning()).toBe(true); // non-fatal
  });

  it('should terminate on fatal error', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const errorCb = vi.fn();
    manager.onError(errorCb);

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    simulateWorkerMessage({
      type: 'ERROR',
      message: 'fatal error',
      category: 'WASM_INIT',
      fatal: true
    });

    expect(errorCb).toHaveBeenCalledWith('fatal error', 'WASM_INIT', true);
    expect(manager.isRunning()).toBe(false);
  });

  it('should send STOP message', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    manager.stop();

    expect(mockPostMessage).toHaveBeenLastCalledWith(
      expect.objectContaining({ type: 'STOP' })
    );
    expect(manager.getStatus()).toBe('stopping');
  });

  it('should terminate worker', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    manager.terminate();

    expect(mockTerminate).toHaveBeenCalled();
    expect(manager.isRunning()).toBe(false);
    expect(manager.getStatus()).toBe('idle');
  });

  it('should clear callbacks', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const resultsCb = vi.fn();
    manager.onResults(resultsCb);

    manager.clearCallbacks();

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    simulateWorkerMessage({ type: 'RESULTS', payload: { results: [] } });

    expect(resultsCb).not.toHaveBeenCalled();
  });

  it('should support chained callback registration', () => {
    const manager = new EggWorkerManager(createMockWorker);

    const result = manager
      .onResults(() => {})
      .onComplete(() => {})
      .onError(() => {});

    expect(result).toBe(manager);
  });

  it('should ignore non-EggWorkerResponse messages', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const resultsCb = vi.fn();
    manager.onResults(resultsCb);

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    // Send invalid message
    if (mockOnMessage) {
      mockOnMessage({ data: { type: 'INVALID' } } as MessageEvent);
    }

    expect(resultsCb).not.toHaveBeenCalled();
  });

  it('should ignore READY message', async () => {
    const manager = new EggWorkerManager(createMockWorker);
    const resultsCb = vi.fn();
    manager.onResults(resultsCb);

    const params = hexParamsToEggParams(createDefaultEggParamsHex());
    await manager.start(params);

    simulateWorkerMessage({ type: 'READY', version: '1' });

    // READY should be ignored, no callbacks should be called
    expect(resultsCb).not.toHaveBeenCalled();
  });
});
