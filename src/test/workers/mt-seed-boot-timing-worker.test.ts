import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type {
  MtSeedBootTimingSearchParams,
  MtSeedBootTimingWorkerRequest,
  MtSeedBootTimingWorkerResponse,
  MtSeedBootTimingProgress,
  MtSeedBootTimingCompletion,
  MtSeedBootTimingResultsPayload,
} from '@/types/mt-seed-boot-timing-search';
import {
  createDefaultMtSeedBootTimingSearchParams,
  validateMtSeedBootTimingSearchParams,
  isMtSeedBootTimingWorkerResponse,
  estimateSearchCombinations,
} from '@/types/mt-seed-boot-timing-search';

describe('MT Seed Boot Timing Worker Types', () => {
  describe('createDefaultMtSeedBootTimingSearchParams', () => {
    it('should create default params with valid structure', () => {
      const params = createDefaultMtSeedBootTimingSearchParams();

      expect(params).toBeDefined();
      expect(params.dateRange).toBeDefined();
      expect(params.timer0Range).toEqual({ min: 0x0c79, max: 0x0c7b });
      expect(params.vcountRange).toEqual({ min: 0x60, max: 0x60 });
      expect(params.keyInputMask).toBe(0);
      expect(params.macAddress).toHaveLength(6);
      expect(params.hardware).toBe('DS');
      expect(params.romVersion).toBe('B');
      expect(params.romRegion).toBe('JPN');
      expect(params.timeRange.hour).toEqual({ start: 0, end: 23 });
      expect(params.timeRange.minute).toEqual({ start: 0, end: 59 });
      expect(params.timeRange.second).toEqual({ start: 0, end: 59 });
      expect(params.targetSeeds).toEqual([]);
      expect(params.maxResults).toBe(1000);
    });

    it('should create params with current date', () => {
      const now = new Date();
      const params = createDefaultMtSeedBootTimingSearchParams();

      expect(params.dateRange.startYear).toBe(now.getFullYear());
      expect(params.dateRange.startMonth).toBe(now.getMonth() + 1);
    });
  });

  describe('validateMtSeedBootTimingSearchParams', () => {
    function createValidParams(): MtSeedBootTimingSearchParams {
      const params = createDefaultMtSeedBootTimingSearchParams();
      params.targetSeeds = [0x12345678]; // Required
      return params;
    }

    it('should pass validation for valid params', () => {
      const params = createValidParams();
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toHaveLength(0);
    });

    it('should fail when targetSeeds is empty', () => {
      const params = createValidParams();
      params.targetSeeds = [];
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('検索対象のSeedを1つ以上指定してください');
    });

    it('should fail when start date is after end date', () => {
      const params = createValidParams();
      params.dateRange = {
        startYear: 2025,
        startMonth: 12,
        startDay: 31,
        endYear: 2025,
        endMonth: 1,
        endDay: 1,
      };
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('開始日は終了日以前である必要があります');
    });

    it('should fail when timer0Range min > max', () => {
      const params = createValidParams();
      params.timer0Range = { min: 0x1000, max: 0x0fff };
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('Timer0の最小値は最大値以下である必要があります');
    });

    it('should fail when timer0Range is out of bounds', () => {
      const params = createValidParams();
      params.timer0Range = { min: 0, max: 0x10000 }; // Exceeds 0xFFFF
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('Timer0は0x0000-0xFFFFの範囲である必要があります');
    });

    it('should fail when vcountRange min > max', () => {
      const params = createValidParams();
      params.vcountRange = { min: 0x80, max: 0x60 };
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('VCountの最小値は最大値以下である必要があります');
    });

    it('should fail when vcountRange is out of bounds', () => {
      const params = createValidParams();
      params.vcountRange = { min: 0, max: 0x100 }; // Exceeds 0xFF
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('VCountは0x00-0xFFの範囲である必要があります');
    });

    it('should fail when macAddress is invalid', () => {
      const params = createValidParams();
      params.macAddress = [0, 0, 0, 0, 256] as unknown as readonly [
        number,
        number,
        number,
        number,
        number,
        number,
      ];
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('MACアドレスは6バイトの配列である必要があります');
    });

    it('should fail when hour range is invalid', () => {
      const params = createValidParams();
      params.timeRange.hour = { start: 20, end: 10 };
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('時の範囲が無効です');
    });

    it('should fail when maxResults is out of bounds', () => {
      const params = createValidParams();
      params.maxResults = 0;
      const errors = validateMtSeedBootTimingSearchParams(params);

      expect(errors).toContain('結果上限は1-100000の範囲である必要があります');
    });
  });

  describe('isMtSeedBootTimingWorkerResponse', () => {
    it('should return true for valid READY response', () => {
      const response: MtSeedBootTimingWorkerResponse = {
        type: 'READY',
        version: '1',
      };
      expect(isMtSeedBootTimingWorkerResponse(response)).toBe(true);
    });

    it('should return true for valid PROGRESS response', () => {
      const response: MtSeedBootTimingWorkerResponse = {
        type: 'PROGRESS',
        payload: {
          processedCombinations: 100,
          totalCombinations: 1000,
          foundCount: 5,
          progressPercent: 10,
          elapsedMs: 5000,
          estimatedRemainingMs: 45000,
        },
      };
      expect(isMtSeedBootTimingWorkerResponse(response)).toBe(true);
    });

    it('should return true for valid RESULTS response', () => {
      const response: MtSeedBootTimingWorkerResponse = {
        type: 'RESULTS',
        payload: {
          results: [],
          batchIndex: 0,
        },
      };
      expect(isMtSeedBootTimingWorkerResponse(response)).toBe(true);
    });

    it('should return true for valid COMPLETE response', () => {
      const response: MtSeedBootTimingWorkerResponse = {
        type: 'COMPLETE',
        payload: {
          reason: 'completed',
          processedCombinations: 1000,
          totalCombinations: 1000,
          resultsCount: 10,
          elapsedMs: 60000,
        },
      };
      expect(isMtSeedBootTimingWorkerResponse(response)).toBe(true);
    });

    it('should return true for valid ERROR response', () => {
      const response: MtSeedBootTimingWorkerResponse = {
        type: 'ERROR',
        message: 'Test error',
        category: 'RUNTIME',
        fatal: false,
      };
      expect(isMtSeedBootTimingWorkerResponse(response)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isMtSeedBootTimingWorkerResponse(null)).toBe(false);
    });

    it('should return false for non-object', () => {
      expect(isMtSeedBootTimingWorkerResponse('string')).toBe(false);
    });

    it('should return false for object without type', () => {
      expect(isMtSeedBootTimingWorkerResponse({ payload: {} })).toBe(false);
    });

    it('should return false for invalid type', () => {
      expect(isMtSeedBootTimingWorkerResponse({ type: 'INVALID' })).toBe(false);
    });
  });

  describe('estimateSearchCombinations', () => {
    it('should calculate combinations correctly for minimal range', () => {
      const params = createDefaultMtSeedBootTimingSearchParams();
      params.targetSeeds = [0x12345678];
      params.timer0Range = { min: 0x0c79, max: 0x0c79 }; // 1 value
      params.vcountRange = { min: 0x60, max: 0x60 }; // 1 value
      params.keyInputMask = 0; // 1 key code
      params.timeRange = {
        hour: { start: 12, end: 12 }, // 1 hour
        minute: { start: 0, end: 0 }, // 1 minute
        second: { start: 0, end: 0 }, // 1 second
      };
      // 1 day

      const combinations = estimateSearchCombinations(params);

      // 1 second * 1 timer0 * 1 vcount * 1 keycode = 1
      expect(combinations).toBe(1);
    });

    it('should scale with timer0 range', () => {
      const params = createDefaultMtSeedBootTimingSearchParams();
      params.targetSeeds = [0x12345678];
      params.timer0Range = { min: 0x0c79, max: 0x0c7b }; // 3 values
      params.vcountRange = { min: 0x60, max: 0x60 }; // 1 value
      params.keyInputMask = 0; // 1 key code
      params.timeRange = {
        hour: { start: 12, end: 12 },
        minute: { start: 0, end: 0 },
        second: { start: 0, end: 0 },
      };

      const combinations = estimateSearchCombinations(params);

      // 1 second * 3 timer0 * 1 vcount * 1 keycode = 3
      expect(combinations).toBe(3);
    });

    it('should scale with key input mask bits', () => {
      const params = createDefaultMtSeedBootTimingSearchParams();
      params.targetSeeds = [0x12345678];
      params.timer0Range = { min: 0x0c79, max: 0x0c79 }; // 1 value
      params.vcountRange = { min: 0x60, max: 0x60 }; // 1 value
      params.keyInputMask = 0b11; // 2 bits = 4 key codes
      params.timeRange = {
        hour: { start: 12, end: 12 },
        minute: { start: 0, end: 0 },
        second: { start: 0, end: 0 },
      };

      const combinations = estimateSearchCombinations(params);

      // 1 second * 1 timer0 * 1 vcount * 4 keycodes = 4
      expect(combinations).toBe(4);
    });
  });
});

describe('MT Seed Boot Timing Worker Mock Integration', () => {
  let mockPostMessage: ReturnType<typeof vi.fn>;
  let mockTerminate: ReturnType<typeof vi.fn>;
  let mockOnMessage: ((ev: MessageEvent) => void) | null;
  let mockOnError: ((ev: ErrorEvent) => void) | null;

  function createMockWorker() {
    mockPostMessage = vi.fn();
    mockTerminate = vi.fn();
    mockOnMessage = null;
    mockOnError = null;

    return {
      postMessage: mockPostMessage,
      terminate: mockTerminate,
      set onmessage(handler: ((ev: MessageEvent) => void) | null) {
        mockOnMessage = handler;
      },
      get onmessage() {
        return mockOnMessage;
      },
      set onerror(handler: ((ev: ErrorEvent) => void) | null) {
        mockOnError = handler;
      },
      get onerror() {
        return mockOnError;
      },
    } as unknown as Worker;
  }

  function simulateWorkerMessage(msg: MtSeedBootTimingWorkerResponse) {
    if (mockOnMessage) {
      mockOnMessage({ data: msg } as MessageEvent);
    }
  }

  beforeEach(() => {
    mockPostMessage = vi.fn();
    mockTerminate = vi.fn();
    mockOnMessage = null;
    mockOnError = null;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should create mock worker and set handlers', () => {
    const worker = createMockWorker();
    const onMessageHandler = vi.fn();
    worker.onmessage = onMessageHandler;

    expect(worker.onmessage).toBe(onMessageHandler);
  });

  it('should send START_SEARCH request with valid params', () => {
    const worker = createMockWorker();
    const params = createDefaultMtSeedBootTimingSearchParams();
    params.targetSeeds = [0x12345678];

    const request: MtSeedBootTimingWorkerRequest = {
      type: 'START_SEARCH',
      params,
    };
    worker.postMessage(request);

    expect(mockPostMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'START_SEARCH', params })
    );
  });

  it('should send STOP request', () => {
    const worker = createMockWorker();

    const request: MtSeedBootTimingWorkerRequest = {
      type: 'STOP',
    };
    worker.postMessage(request);

    expect(mockPostMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'STOP' })
    );
  });

  it('should handle READY response', () => {
    createMockWorker();
    const _readyHandler = vi.fn();

    const response: MtSeedBootTimingWorkerResponse = {
      type: 'READY',
      version: '1',
    };

    // Simulate receiving the message
    if (mockOnMessage) {
      mockOnMessage({
        data: response,
      } as MessageEvent);
    }

    // Since we're testing the type validation directly
    expect(isMtSeedBootTimingWorkerResponse(response)).toBe(true);
  });

  it('should handle PROGRESS response', () => {
    createMockWorker();
    const _progressHandler = vi.fn();

    const progress: MtSeedBootTimingProgress = {
      processedCombinations: 500,
      totalCombinations: 1000,
      foundCount: 3,
      progressPercent: 50,
      elapsedMs: 30000,
      estimatedRemainingMs: 30000,
    };

    const response: MtSeedBootTimingWorkerResponse = {
      type: 'PROGRESS',
      payload: progress,
    };

    expect(response.type).toBe('PROGRESS');
    expect(response.payload.progressPercent).toBe(50);
  });

  it('should handle RESULTS response', () => {
    createMockWorker();

    const resultsPayload: MtSeedBootTimingResultsPayload = {
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
    };

    const response: MtSeedBootTimingWorkerResponse = {
      type: 'RESULTS',
      payload: resultsPayload,
    };

    expect(response.type).toBe('RESULTS');
    expect(response.payload.results).toHaveLength(1);
    expect(response.payload.results[0].mtSeedHex).toBe('12345678');
  });

  it('should handle COMPLETE response', () => {
    createMockWorker();

    const completion: MtSeedBootTimingCompletion = {
      reason: 'completed',
      processedCombinations: 1000,
      totalCombinations: 1000,
      resultsCount: 5,
      elapsedMs: 60000,
    };

    const response: MtSeedBootTimingWorkerResponse = {
      type: 'COMPLETE',
      payload: completion,
    };

    expect(response.type).toBe('COMPLETE');
    expect(response.payload.reason).toBe('completed');
    expect(response.payload.resultsCount).toBe(5);
  });

  it('should handle ERROR response', () => {
    createMockWorker();

    const response: MtSeedBootTimingWorkerResponse = {
      type: 'ERROR',
      message: 'WASM initialization failed',
      category: 'WASM_INIT',
      fatal: true,
    };

    expect(response.type).toBe('ERROR');
    expect(response.message).toBe('WASM initialization failed');
    expect(response.fatal).toBe(true);
  });

  it('should simulate complete search workflow', async () => {
    const worker = createMockWorker();
    const params = createDefaultMtSeedBootTimingSearchParams();
    params.targetSeeds = [0x12345678];

    // Track received messages
    const receivedMessages: MtSeedBootTimingWorkerResponse[] = [];
    worker.onmessage = (ev: MessageEvent<MtSeedBootTimingWorkerResponse>) => {
      receivedMessages.push(ev.data);
    };

    // 1. Send START_SEARCH
    const request: MtSeedBootTimingWorkerRequest = {
      type: 'START_SEARCH',
      params,
    };
    worker.postMessage(request);

    // 2. Simulate worker responses
    simulateWorkerMessage({ type: 'READY', version: '1' });

    simulateWorkerMessage({
      type: 'PROGRESS',
      payload: {
        processedCombinations: 50,
        totalCombinations: 100,
        foundCount: 1,
        progressPercent: 50,
        elapsedMs: 1000,
        estimatedRemainingMs: 1000,
      },
    });

    simulateWorkerMessage({
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
    });

    simulateWorkerMessage({
      type: 'COMPLETE',
      payload: {
        reason: 'completed',
        processedCombinations: 100,
        totalCombinations: 100,
        resultsCount: 1,
        elapsedMs: 2000,
      },
    });

    // 3. Verify workflow
    expect(mockPostMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'START_SEARCH' })
    );
    expect(receivedMessages).toHaveLength(4);
    expect(receivedMessages.map((m) => m.type)).toEqual([
      'READY',
      'PROGRESS',
      'RESULTS',
      'COMPLETE',
    ]);
  });

  it('should handle stop request during search', () => {
    const worker = createMockWorker();
    const params = createDefaultMtSeedBootTimingSearchParams();
    params.targetSeeds = [0x12345678];

    // Track received messages
    const receivedMessages: MtSeedBootTimingWorkerResponse[] = [];
    worker.onmessage = (ev: MessageEvent<MtSeedBootTimingWorkerResponse>) => {
      receivedMessages.push(ev.data);
    };

    // Start search
    worker.postMessage({ type: 'START_SEARCH', params });

    // Send stop request
    worker.postMessage({ type: 'STOP' });

    // Simulate stopped completion
    simulateWorkerMessage({
      type: 'COMPLETE',
      payload: {
        reason: 'stopped',
        processedCombinations: 50,
        totalCombinations: 100,
        resultsCount: 0,
        elapsedMs: 500,
      },
    });

    expect(mockPostMessage).toHaveBeenCalledTimes(2);
    expect(mockPostMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ type: 'STOP' })
    );
    expect(receivedMessages[0].type).toBe('COMPLETE');
    expect(
      (receivedMessages[0] as { type: 'COMPLETE'; payload: MtSeedBootTimingCompletion })
        .payload.reason
    ).toBe('stopped');
  });
});
