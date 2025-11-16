/**
 * calculateOptimalChunks テスト
 */

import { describe, it, expect } from 'vitest';
import { calculateOptimalChunks } from '../lib/search/chunk-calculator';
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
    startHour: 0,
    endHour: 23,
    startMinute: 0,
    endMinute: 59,
    startSecond: 0,
    endSecond: 59
  },
  keyInput: 0,
  macAddress: [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]
};

describe('calculateOptimalChunks', () => {
  describe('基本的な分割', () => {
    it('4コア環境で適切な分割を行う', () => {
      const chunks = calculateOptimalChunks(mockConditions, 4);

      expect(chunks).toHaveLength(4);
      expect(chunks[0].workerId).toBe(0);
      expect(chunks[3].workerId).toBe(3);

      chunks.forEach(chunk => {
        expect(chunk.startDateTime).toBeInstanceOf(Date);
        expect(chunk.endDateTime).toBeInstanceOf(Date);
        expect(chunk.startDateTime.getTime()).toBeLessThanOrEqual(chunk.endDateTime.getTime());
        expect(chunk.estimatedOperations).toBeGreaterThan(0);
      });
    });

    it('時刻範囲の境界が正確', () => {
      const chunks = calculateOptimalChunks(mockConditions, 2);

      expect(chunks).toHaveLength(2);

      const firstChunk = chunks[0];
      expect(firstChunk.startDateTime.getFullYear()).toBe(2013);
      expect(firstChunk.startDateTime.getMonth()).toBe(0);
      expect(firstChunk.startDateTime.getDate()).toBe(1);

      const lastChunk = chunks[chunks.length - 1];
      expect(lastChunk.endDateTime.getFullYear()).toBe(2013);
      expect(lastChunk.endDateTime.getMonth()).toBe(0);
      expect(lastChunk.endDateTime.getDate()).toBeLessThanOrEqual(2);
    });

    it('チャンク間にオーバーラップがない', () => {
      const chunks = calculateOptimalChunks(mockConditions, 3);

      for (let i = 0; i < chunks.length - 1; i++) {
        const currentEnd = chunks[i].endDateTime.getTime();
        const nextStart = chunks[i + 1].startDateTime.getTime();
        expect(nextStart).toBeGreaterThan(currentEnd);
      }
    });
  });

  describe('メモリ制約テスト', () => {
    it('最大Worker数まで処理できる', () => {
      const chunks = calculateOptimalChunks(mockConditions, 8);
      expect(chunks.length).toBeLessThanOrEqual(8);
    });
  });
});
