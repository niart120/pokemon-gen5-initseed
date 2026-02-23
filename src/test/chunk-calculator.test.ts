/**
 * calculateTimeChunks テスト
 */

import { describe, it, expect } from 'vitest';
import {
  calculateTimeChunks,
  calculateOperationsPerSecond,
  getDefaultWorkerCount,
} from '../lib/search/chunk-calculator';
import { countValidKeyCombinations } from '../lib/utils/key-input';
import type { SearchConditions } from '../types/search';

const mockConditions: SearchConditions = {
  romVersion: 'B2',
  romRegion: 'JPN',
  hardware: 'DS',
  timer0VCountConfig: {
    useAutoConfiguration: false,
    timer0Range: { min: 3193, max: 3194 },
    vcountRange: { min: 160, max: 167 }
  },
  dateRange: {
    startYear: 2013,
    endYear: 2013,
    startMonth: 1,
    endMonth: 1,
    startDay: 1,
    endDay: 2,
  },
  timeRange: {
    hour: { start: 0, end: 23 },
    minute: { start: 0, end: 59 },
    second: { start: 0, end: 59 },
  },
  keyInput: 0,
  macAddress: [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]
};

/**
 * SearchConditions から TimeChunkCalculationInput を構築するヘルパー
 */
function buildTimeChunkInput(conditions: SearchConditions) {
  const timer0Count =
    conditions.timer0VCountConfig.timer0Range.max - conditions.timer0VCountConfig.timer0Range.min + 1;
  const vcountCount =
    conditions.timer0VCountConfig.vcountRange.max - conditions.timer0VCountConfig.vcountRange.min + 1;
  const keyCombinationCount = countValidKeyCombinations(conditions.keyInput);

  const operationsPerSecond = calculateOperationsPerSecond({
    timer0Count,
    vcountCount,
    keyCombinationCount,
  });

  const startDateTime = new Date(
    conditions.dateRange.startYear,
    conditions.dateRange.startMonth - 1,
    conditions.dateRange.startDay,
    conditions.timeRange.hour.start,
    conditions.timeRange.minute.start,
    conditions.timeRange.second.start
  );

  const endDateTime = new Date(
    conditions.dateRange.endYear,
    conditions.dateRange.endMonth - 1,
    conditions.dateRange.endDay,
    conditions.timeRange.hour.end,
    conditions.timeRange.minute.end,
    conditions.timeRange.second.end
  );

  return { startDateTime, endDateTime, operationsPerSecond };
}

describe('calculateTimeChunks', () => {
  describe('基本的な分割', () => {
    it('4コア環境で適切な分割を行う', () => {
      const input = buildTimeChunkInput(mockConditions);
      const chunks = calculateTimeChunks(input, 4);

      expect(chunks).toHaveLength(4);
      expect(chunks[0].workerId).toBe(0);
      expect(chunks[3].workerId).toBe(3);

      chunks.forEach(chunk => {
        expect(chunk.startDateTime).toBeInstanceOf(Date);
        expect(chunk.endDateTime).toBeInstanceOf(Date);
        expect(chunk.startDateTime.getTime()).toBeLessThanOrEqual(chunk.endDateTime.getTime());
        expect(chunk.estimatedOperations).toBeGreaterThan(0);
        expect(chunk.rangeSeconds).toBeGreaterThan(0);
      });
    });

    it('時刻範囲の境界が正確', () => {
      const input = buildTimeChunkInput(mockConditions);
      const chunks = calculateTimeChunks(input, 2);

      expect(chunks).toHaveLength(2);

      const firstChunk = chunks[0];
      expect(firstChunk.startDateTime.getFullYear()).toBe(2013);
      expect(firstChunk.startDateTime.getMonth()).toBe(0);
      expect(firstChunk.startDateTime.getDate()).toBe(1);

      // 最後のチャンクの終了日時は入力の endDateTime に基づく
      const lastChunk = chunks[chunks.length - 1];
      expect(lastChunk.endDateTime.getFullYear()).toBe(2013);
      expect(lastChunk.endDateTime.getMonth()).toBe(0);
      // endDateTime は startDateTime + totalSeconds なので、日付が範囲内であること
      expect(lastChunk.endDateTime.getTime()).toBeLessThanOrEqual(input.endDateTime.getTime() + 1000);
    });

    it('チャンク間にオーバーラップがない', () => {
      const input = buildTimeChunkInput(mockConditions);
      const chunks = calculateTimeChunks(input, 3);

      for (let i = 0; i < chunks.length - 1; i++) {
        const currentEnd = chunks[i].endDateTime.getTime();
        const nextStart = chunks[i + 1].startDateTime.getTime();
        expect(nextStart).toBeGreaterThanOrEqual(currentEnd);
      }
    });
  });

  describe('境界条件テスト', () => {
    it('チャンクが半開区間として連続している（endDateTime === 次のstartDateTime）', () => {
      // 10秒の範囲を3 Workerで分割
      const startDateTime = new Date('2024-01-01T00:00:00');
      const endDateTime = new Date('2024-01-01T00:00:09'); // 10秒間 (0-9秒)
      const chunks = calculateTimeChunks(
        { startDateTime, endDateTime, operationsPerSecond: 1 },
        3
      );

      // 各チャンクの endDateTime が次の startDateTime と一致（半開区間）
      for (let i = 0; i < chunks.length - 1; i++) {
        expect(chunks[i].endDateTime.getTime()).toBe(chunks[i + 1].startDateTime.getTime());
      }
    });

    it('rangeSecondsの合計が全体の秒数と一致する', () => {
      const startDateTime = new Date('2024-01-01T00:00:00');
      const endDateTime = new Date('2024-01-01T00:00:09'); // 10秒間
      const chunks = calculateTimeChunks(
        { startDateTime, endDateTime, operationsPerSecond: 1 },
        3
      );

      const totalRangeSeconds = chunks.reduce((sum, c) => sum + c.rangeSeconds, 0);
      const expectedTotalSeconds = Math.floor((endDateTime.getTime() - startDateTime.getTime()) / 1000) + 1;
      expect(totalRangeSeconds).toBe(expectedTotalSeconds);
    });

    it('最初のチャンクの開始と最後のチャンクの終了が入力範囲をカバーする', () => {
      const startDateTime = new Date('2024-01-01T00:00:00');
      const endDateTime = new Date('2024-01-01T00:00:59'); // 60秒間
      const chunks = calculateTimeChunks(
        { startDateTime, endDateTime, operationsPerSecond: 1 },
        4
      );

      // 最初のチャンクは入力の開始時刻から始まる
      expect(chunks[0].startDateTime.getTime()).toBe(startDateTime.getTime());

      // 最後のチャンクの終了は totalSeconds 後（半開区間の終端）
      const expectedEndMs = startDateTime.getTime() + 60 * 1000; // 60秒後
      expect(chunks[chunks.length - 1].endDateTime.getTime()).toBe(expectedEndMs);
    });

    it('Worker数が秒数より多い場合、実際の秒数分のチャンクのみ生成される', () => {
      const startDateTime = new Date('2024-01-01T00:00:00');
      const endDateTime = new Date('2024-01-01T00:00:02'); // 3秒間
      const chunks = calculateTimeChunks(
        { startDateTime, endDateTime, operationsPerSecond: 1 },
        10 // 秒数より多いWorker数
      );

      // 3秒間なので最大3チャンク
      expect(chunks.length).toBeLessThanOrEqual(3);
      expect(chunks.length).toBeGreaterThan(0);

      // rangeSecondsの合計は3秒
      const totalRangeSeconds = chunks.reduce((sum, c) => sum + c.rangeSeconds, 0);
      expect(totalRangeSeconds).toBe(3);
    });

    it('1秒の範囲でも正しく処理される', () => {
      const startDateTime = new Date('2024-01-01T00:00:00');
      const endDateTime = new Date('2024-01-01T00:00:00'); // 同じ時刻 = 1秒
      const chunks = calculateTimeChunks(
        { startDateTime, endDateTime, operationsPerSecond: 1 },
        4
      );

      expect(chunks).toHaveLength(1);
      expect(chunks[0].rangeSeconds).toBe(1);
      expect(chunks[0].startDateTime.getTime()).toBe(startDateTime.getTime());
    });

    it('rangeSecondsは半開区間として扱われる（各秒は1回だけ処理される）', () => {
      // 具体例: 0秒, 1秒, 2秒 の3秒間を2 Workerで処理
      // Worker 0: 0秒, 1秒 (rangeSeconds=2, offset [0,2))
      // Worker 1: 2秒 (rangeSeconds=1, offset [2,3))
      const startDateTime = new Date('2024-01-01T00:00:00');
      const endDateTime = new Date('2024-01-01T00:00:02'); // 3秒間
      const chunks = calculateTimeChunks(
        { startDateTime, endDateTime, operationsPerSecond: 1 },
        2
      );

      expect(chunks).toHaveLength(2);

      // 各チャンクの rangeSeconds の合計 = 3（各秒が1回だけカウント）
      expect(chunks[0].rangeSeconds + chunks[1].rangeSeconds).toBe(3);

      // estimatedOperations も rangeSeconds に基づく
      expect(chunks[0].estimatedOperations).toBe(chunks[0].rangeSeconds);
      expect(chunks[1].estimatedOperations).toBe(chunks[1].rangeSeconds);
    });
  });

  describe('メモリ制約テスト', () => {
    it('最大Worker数まで処理できる', () => {
      const input = buildTimeChunkInput(mockConditions);
      const chunks = calculateTimeChunks(input, 8);
      expect(chunks.length).toBeLessThanOrEqual(8);
    });
  });
});

describe('calculateOperationsPerSecond', () => {
  it('基本的な計算が正しい', () => {
    const result = calculateOperationsPerSecond({
      timer0Count: 2,
      vcountCount: 8,
      keyCombinationCount: 1,
    });
    expect(result).toBe(16);
  });

  it('advanceCount が考慮される', () => {
    const result = calculateOperationsPerSecond({
      timer0Count: 2,
      vcountCount: 8,
      keyCombinationCount: 1,
      advanceCount: 100,
    });
    expect(result).toBe(1600);
  });

  it('最小値は1を返す', () => {
    const result = calculateOperationsPerSecond({
      timer0Count: 0,
      vcountCount: 0,
      keyCombinationCount: 0,
    });
    expect(result).toBe(1);
  });
});

describe('getDefaultWorkerCount', () => {
  it('正の整数を返す', () => {
    const count = getDefaultWorkerCount();
    expect(count).toBeGreaterThan(0);
    expect(Number.isInteger(count)).toBe(true);
  });
});
