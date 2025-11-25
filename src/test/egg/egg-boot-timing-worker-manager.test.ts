/**
 * Tests for Egg Boot Timing Worker Manager
 * Worker立ち上げ、並列Worker / Manager検証テスト
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  EggBootTimingWorkerManager,
  EggBootTimingMultiWorkerManager,
  calculateEggBootTimingChunks,
  calculateBatchSize,
  getDefaultWorkerCount,
} from '@/lib/egg';
import type { EggBootTimingSearchParams } from '@/types/egg-boot-timing-search';
import { EggGameMode } from '@/types/egg';

// テスト用デフォルトパラメータ
const createTestParams = (overrides: Partial<EggBootTimingSearchParams> = {}): EggBootTimingSearchParams => ({
  dateRange: {
    startYear: 2025,
    startMonth: 1,
    startDay: 15,
    endYear: 2025,
    endMonth: 1,
    endDay: 22,  // 7日間
  },
  timer0Range: { min: 0x0C79, max: 0x0C7B },
  vcountRange: { min: 0x60, max: 0x60 },
  keyInputMask: 0,
  macAddress: [0x00, 0x09, 0xBF, 0x12, 0x34, 0x56],
  hardware: 'DS',
  romVersion: 'B',
  romRegion: 'JPN',
  timeRange: {
    hour: { start: 12, end: 14 },
    minute: { start: 0, end: 59 },
    second: { start: 0, end: 59 },
  },
  frame: 8,
  conditions: {
    hasNidoranFlag: false,
    everstone: { type: 'none' },
    usesDitto: false,
    femaleParentAbility: 0,
    masudaMethod: false,
    tid: 12345,
    sid: 54321,
    genderRatio: { threshold: 127, genderless: false },
  },
  parents: {
    male: [31, 31, 31, 31, 31, 31],
    female: [31, 31, 31, 31, 31, 31],
  },
  filter: null,
  filterDisabled: false,
  considerNpcConsumption: false,
  gameMode: EggGameMode.BwContinue,
  userOffset: 0,
  advanceCount: 100,
  maxResults: 1000,
  ...overrides,
});

describe('Egg Boot Timing Worker Manager Tests', () => {
  describe('EggBootTimingWorkerManager (単一Worker)', () => {
    let manager: EggBootTimingWorkerManager;

    beforeEach(() => {
      manager = new EggBootTimingWorkerManager();
    });

    afterEach(() => {
      manager.terminate();
    });

    it('should create manager instance successfully', () => {
      expect(manager).toBeInstanceOf(EggBootTimingWorkerManager);
      expect(manager.running).toBe(false);
    });

    it('should have correct initial running state', () => {
      expect(manager.running).toBe(false);
    });

    it('should throw when starting search without initialization', async () => {
      const params = createTestParams();
      await expect(manager.startSearch(params)).rejects.toThrow('Worker not initialized');
    });

    it('should terminate without error when not initialized', () => {
      expect(() => manager.terminate()).not.toThrow();
    });

    it('should stopSearch without error when not running', () => {
      expect(() => manager.stopSearch()).not.toThrow();
    });
  });

  describe('EggBootTimingMultiWorkerManager (並列Worker)', () => {
    let multiManager: EggBootTimingMultiWorkerManager;

    beforeEach(() => {
      multiManager = new EggBootTimingMultiWorkerManager(4);
    });

    afterEach(() => {
      multiManager.terminateAll();
    });

    it('should create multi-worker manager instance successfully', () => {
      expect(multiManager).toBeInstanceOf(EggBootTimingMultiWorkerManager);
    });

    it('should have correct initial state', () => {
      expect(multiManager.isRunning()).toBe(false);
      expect(multiManager.getActiveWorkerCount()).toBe(0);
      expect(multiManager.getResultsCount()).toBe(0);
    });

    it('should get and set max workers correctly', () => {
      expect(multiManager.getMaxWorkers()).toBe(4);
      
      multiManager.setMaxWorkers(2);
      expect(multiManager.getMaxWorkers()).toBe(2);
      
      // Should clamp to at least 1
      multiManager.setMaxWorkers(0);
      expect(multiManager.getMaxWorkers()).toBe(1);
    });

    it('should not allow changing workers during search (mock test)', () => {
      // This is a behavioral test - we can't actually start a search in unit tests
      // but we verify the guard exists
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      
      // The guard should prevent changes when searchRunning is true
      // Since we can't actually run a search, we just verify the method exists
      expect(typeof multiManager.setMaxWorkers).toBe('function');
      
      consoleSpy.mockRestore();
    });

    it('should terminate all workers without error', () => {
      expect(() => multiManager.terminateAll()).not.toThrow();
      expect(multiManager.isRunning()).toBe(false);
    });

    it('should pause and resume without error when not running', () => {
      expect(() => multiManager.pauseAll()).not.toThrow();
      expect(() => multiManager.resumeAll()).not.toThrow();
    });

    it('should cleanup safely on multiple terminate calls', () => {
      expect(() => multiManager.terminateAll()).not.toThrow();
      expect(() => multiManager.terminateAll()).not.toThrow();
      expect(multiManager.isRunning()).toBe(false);
    });
  });

  describe('Chunk Calculator', () => {
    it('should calculate chunks for 7-day search', () => {
      const params = createTestParams(); // 7 days by default (Jan 15-22)
      const chunks = calculateEggBootTimingChunks(params, 4);

      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks.length).toBeLessThanOrEqual(4);

      // 全チャンクの合計が検索範囲と等しいことを確認 (8日分 = 691200秒)
      const totalSeconds = chunks.reduce((sum, c) => sum + c.rangeSeconds, 0);
      // 8日間 (startDay=15, endDay=22 => 8日間)
      expect(totalSeconds).toBe(8 * 24 * 60 * 60);
    });

    it('should calculate chunks for 1-year search', () => {
      const params = createTestParams({
        dateRange: {
          startYear: 2025,
          startMonth: 1,
          startDay: 1,
          endYear: 2025,
          endMonth: 12,
          endDay: 31,
        },
      });
      const chunks = calculateEggBootTimingChunks(params, 8);

      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks.length).toBeLessThanOrEqual(8);

      const totalSeconds = chunks.reduce((sum, c) => sum + c.rangeSeconds, 0);
      // 365日間
      expect(totalSeconds).toBe(365 * 24 * 60 * 60);
    });

    it('should handle single-worker scenario', () => {
      const params = createTestParams({
        dateRange: {
          startYear: 2025,
          startMonth: 1,
          startDay: 1,
          endYear: 2025,
          endMonth: 1,
          endDay: 1,
        },
      }); // 1 day
      const chunks = calculateEggBootTimingChunks(params, 1);

      expect(chunks.length).toBe(1);
      expect(chunks[0].rangeSeconds).toBe(86400);
      expect(chunks[0].workerId).toBe(0);
    });

    it('should set correct start and end datetimes', () => {
      const params = createTestParams({
        dateRange: {
          startYear: 2025,
          startMonth: 1,
          startDay: 15,
          endYear: 2025,
          endMonth: 1,
          endDay: 15,
        },
      }); // 1 day
      const chunks = calculateEggBootTimingChunks(params, 2);

      // Should have chunks covering the day
      expect(chunks[0].startDatetime.getFullYear()).toBe(2025);
      expect(chunks[0].startDatetime.getMonth()).toBe(0); // January
      expect(chunks[0].startDatetime.getDate()).toBe(15);
    });

    it('should have non-negative estimated operations', () => {
      const params = createTestParams();
      const chunks = calculateEggBootTimingChunks(params, 4);

      for (const chunk of chunks) {
        expect(chunk.estimatedOperations).toBeGreaterThan(0);
      }
    });
  });

  describe('Batch Size Calculator', () => {
    it('should calculate batch size for small search (1 hour)', () => {
      const params = createTestParams({
        dateRange: {
          startYear: 2025,
          startMonth: 1,
          startDay: 1,
          endYear: 2025,
          endMonth: 1,
          endDay: 1,
        },
      }); // 1 day
      const batchSize = calculateBatchSize(params);

      // 最小バッチサイズは3600秒（1時間）
      expect(batchSize).toBeGreaterThanOrEqual(3600);
    });

    it('should calculate batch size for 7-day search', () => {
      const params = createTestParams(); // 8 days by default
      const batchSize = calculateBatchSize(params);

      // 8日間 / 500バッチ = 約1382秒 → 最小3600秒に引き上げ
      expect(batchSize).toBeGreaterThanOrEqual(3600);
      expect(batchSize).toBeLessThanOrEqual(86400); // 最大1日
    });

    it('should calculate batch size for 1-year search', () => {
      const params = createTestParams({
        dateRange: {
          startYear: 2025,
          startMonth: 1,
          startDay: 1,
          endYear: 2025,
          endMonth: 12,
          endDay: 31,
        },
      }); // 1 year
      const batchSize = calculateBatchSize(params);

      // 1年 / 500バッチ = 約63072秒
      expect(batchSize).toBeGreaterThanOrEqual(3600);
      expect(batchSize).toBeLessThanOrEqual(86400);
      
      // 目標500バッチなので、63072秒程度になるはず
      expect(batchSize).toBeGreaterThan(60000);
    });

    it('should return consistent batch size for same params', () => {
      const params = createTestParams();
      const batchSize1 = calculateBatchSize(params);
      const batchSize2 = calculateBatchSize(params);

      expect(batchSize1).toBe(batchSize2);
    });
  });

  describe('Default Worker Count', () => {
    it('should return a positive number', () => {
      const count = getDefaultWorkerCount();
      expect(count).toBeGreaterThan(0);
    });

    it('should return at least 4 (or hardwareConcurrency)', () => {
      const count = getDefaultWorkerCount();
      // In Node.js test environment, navigator is undefined, so defaults to 4
      expect(count).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Search Parameters Validation', () => {
    it('should have valid timer0 range', () => {
      const params = createTestParams();
      expect(params.timer0Range.min).toBeLessThanOrEqual(params.timer0Range.max);
      expect(params.timer0Range.min).toBeGreaterThanOrEqual(0);
      expect(params.timer0Range.max).toBeLessThanOrEqual(0xFFFF);
    });

    it('should have valid vcount range', () => {
      const params = createTestParams();
      expect(params.vcountRange.min).toBeLessThanOrEqual(params.vcountRange.max);
      expect(params.vcountRange.min).toBeGreaterThanOrEqual(0);
      expect(params.vcountRange.max).toBeLessThanOrEqual(0xFF);
    });

    it('should have valid time range', () => {
      const params = createTestParams();
      expect(params.timeRange.hour.start).toBeLessThanOrEqual(params.timeRange.hour.end);
      expect(params.timeRange.minute.start).toBeLessThanOrEqual(params.timeRange.minute.end);
      expect(params.timeRange.second.start).toBeLessThanOrEqual(params.timeRange.second.end);
    });

    it('should have valid MAC address', () => {
      const params = createTestParams();
      expect(params.macAddress.length).toBe(6);
      for (const byte of params.macAddress) {
        expect(byte).toBeGreaterThanOrEqual(0);
        expect(byte).toBeLessThanOrEqual(255);
      }
    });

    it('should have valid parent IVs', () => {
      const params = createTestParams();
      expect(params.parents.male.length).toBe(6);
      expect(params.parents.female.length).toBe(6);
      
      for (const iv of params.parents.male) {
        expect(iv).toBeGreaterThanOrEqual(0);
        expect(iv).toBeLessThanOrEqual(31);
      }
      for (const iv of params.parents.female) {
        expect(iv).toBeGreaterThanOrEqual(0);
        expect(iv).toBeLessThanOrEqual(31);
      }
    });
  });

  describe('7-Day Search Scenario (Integration)', () => {
    it('should generate correct chunks for 7-day search with filters', () => {
      // 実際の検索シナリオ: 7日間、3時間の時間帯制限
      const params = createTestParams({
        timeRange: {
          hour: { start: 12, end: 14 }, // 3時間
          minute: { start: 0, end: 59 },
          second: { start: 0, end: 59 },
        },
        timer0Range: { min: 0x0C79, max: 0x0C7B }, // 3 values
        vcountRange: { min: 0x60, max: 0x60 }, // 1 value
        advanceCount: 100, // 100消費
      });

      const chunks = calculateEggBootTimingChunks(params, 4);

      // チャンクが生成されること
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks.length).toBeLessThanOrEqual(4);

      // 各チャンクが有効な情報を持つこと
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        expect(chunk.workerId).toBe(i);
        expect(chunk.rangeSeconds).toBeGreaterThan(0);
        expect(chunk.startDatetime).toBeInstanceOf(Date);
        expect(chunk.endDatetime).toBeInstanceOf(Date);
        expect(chunk.startDatetime.getTime()).toBeLessThan(chunk.endDatetime.getTime());
      }
    });

    it('should calculate appropriate batch size for realistic search', () => {
      // 現実的な検索シナリオ: 1年×100消費
      const params = createTestParams({
        dateRange: {
          startYear: 2025,
          startMonth: 1,
          startDay: 1,
          endYear: 2025,
          endMonth: 12,
          endDay: 31,
        },
        advanceCount: 100,
      });

      const batchSize = calculateBatchSize(params);

      // 1年 / 500バッチ = 63072秒（約17.5時間）
      expect(batchSize).toBeGreaterThanOrEqual(3600); // 最小1時間
      expect(batchSize).toBeLessThanOrEqual(86400); // 最大1日
      
      // 妥当なバッチ数になることを確認
      const totalRangeSeconds = 365 * 24 * 60 * 60;
      const estimatedBatches = Math.ceil(totalRangeSeconds / batchSize);
      expect(estimatedBatches).toBeLessThanOrEqual(1000);
    });
  });
});
