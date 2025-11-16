import { describe, expect, it, vi } from 'vitest';
import { hasImpossibleKeyCombination, keyCodeToMask, keyNamesToMask } from '@/lib/utils/key-input';
import { buildSearchContext } from '@/lib/webgpu/seed-search/message-encoder';
import type { SearchConditions } from '@/types/search';

const REAL_DATE = Date;

function createSearchConditions(
  dateRange: SearchConditions['dateRange'],
  options?: { timeRange?: SearchConditions['timeRange'] }
): SearchConditions {
  const timeRange = options?.timeRange ?? {
    hour: { start: dateRange.startHour, end: dateRange.endHour },
    minute: { start: dateRange.startMinute, end: dateRange.endMinute },
    second: { start: dateRange.startSecond, end: dateRange.endSecond },
  };
  return {
    romVersion: 'B',
    romRegion: 'JPN',
    hardware: 'DS',
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: { min: 0xC79, max: 0xC7A },
      vcountRange: { min: 0x60, max: 0x60 },
    },
    dateRange,
    timeRange,
    keyInput: 0,
    macAddress: [0x00, 0x1B, 0x2C, 0x3D, 0x4E, 0x5F],
  };
}

function withMockedLocalTimezone<T>(offsetMinutes: number, callback: () => T): T {
  class MockDate extends REAL_DATE {
    constructor(...args: unknown[]) {
      if (args.length === 0) {
        super();
        return;
      }

      if (args.length === 1 && typeof args[0] === 'number') {
        super(args[0]);
        return;
      }

      if (args.length === 1 && typeof args[0] === 'string') {
        super(args[0]);
        return;
      }

      const [year, month = 0, day = 1, hour = 0, minute = 0, second = 0, millisecond = 0] =
        args as number[];

      const zoneAdjusted =
        REAL_DATE.UTC(year, month, day, hour, minute, second, millisecond) - offsetMinutes * 60 * 1000;

      super(zoneAdjusted);
    }
  }

  Object.setPrototypeOf(MockDate, REAL_DATE);
  MockDate.UTC = REAL_DATE.UTC;
  MockDate.parse = REAL_DATE.parse;
  MockDate.now = REAL_DATE.now;

  vi.stubGlobal('Date', MockDate as unknown as DateConstructor);

  try {
    return callback();
  } finally {
    vi.unstubAllGlobals();
  }
}

describe('buildSearchContext', () => {
  const singleDayRange: SearchConditions['dateRange'] = {
    startYear: 2000,
    startMonth: 1,
    startDay: 1,
    startHour: 0,
    startMinute: 0,
    startSecond: 0,
    endYear: 2000,
    endMonth: 1,
    endDay: 1,
    endHour: 0,
    endMinute: 0,
    endSecond: 0,
  };

  it('2000-01-01 はいかなるローカルタイムゾーンでもエポックチェックを通過するべき', () => {
    const result = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      const context = buildSearchContext(conditions);

      const expectedLocalTimestamp = new Date(
        conditions.dateRange.startYear,
        conditions.dateRange.startMonth - 1,
        conditions.dateRange.startDay,
        conditions.dateRange.startHour,
        conditions.dateRange.startMinute,
        conditions.dateRange.startSecond
      ).getTime();

      expect(context.startTimestampMs).toBe(expectedLocalTimestamp);
      return context;
    });

    const expectedEpoch = REAL_DATE.UTC(2000, 0, 1, 0, 0, 0);

    expect(result.rangeSeconds).toBe(1);
    expect(result.segments[0]?.config.startSecondsSince2000).toBe(0);
    expect(result.startTimestampMs).toBeGreaterThanOrEqual(expectedEpoch);
  });

  it('2100-01-01 は 2000 エポック以降の正しい秒数を返すべき', () => {
    const futureRange: SearchConditions['dateRange'] = {
      ...singleDayRange,
      startYear: 2100,
      endYear: 2100,
    };

    const result = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(futureRange);
      const context = buildSearchContext(conditions);

      const expectedLocalTimestamp = new Date(
        conditions.dateRange.startYear,
        conditions.dateRange.startMonth - 1,
        conditions.dateRange.startDay,
        conditions.dateRange.startHour,
        conditions.dateRange.startMinute,
        conditions.dateRange.startSecond
      ).getTime();

      expect(context.startTimestampMs).toBe(expectedLocalTimestamp);
      return context;
    });

    const expectedEpoch = REAL_DATE.UTC(2000, 0, 1, 0, 0, 0);
    const expectedStart = REAL_DATE.UTC(2100, 0, 1, 0, 0, 0);
    const expectedSeconds = Math.floor((expectedStart - expectedEpoch) / 1000);

    expect(result.segments[0]?.config.startSecondsSince2000).toBe(expectedSeconds >>> 0);
    expect(result.startTimestampMs).toBeGreaterThan(expectedEpoch);
  });

  it('時刻範囲の境界が正確に rangeSeconds に反映される', () => {
    const multiSecondRange: SearchConditions['dateRange'] = {
      startYear: 2001,
      endYear: 2001,
      startMonth: 5,
      endMonth: 5,
      startDay: 10,
      endDay: 10,
      startHour: 12,
      endHour: 12,
      startMinute: 0,
      endMinute: 0,
      startSecond: 0,
      endSecond: 5,
    };

    const context = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(multiSecondRange);
      return buildSearchContext(conditions);
    });

    expect(context.rangeSeconds).toBe(6);
    expect(context.totalMessages).toBe(6 * 2);
  });

  it('生成されたセグメントにチャンクのオーバーラップがない', () => {
    const wideTimerRangeConditions = createSearchConditions({
      ...singleDayRange,
    });

    wideTimerRangeConditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC7C };

    const context = withMockedLocalTimezone(-540, () => buildSearchContext(wideTimerRangeConditions));

    const { segments } = context;
    const totalMessages = segments.reduce((sum, segment) => sum + segment.totalMessages, 0);

    expect(segments.length).toBeGreaterThan(0);

    for (let index = 1; index < segments.length; index += 1) {
      const previous = segments[index - 1];
      const current = segments[index];
      expect(current.baseOffset).toBe(previous.baseOffset + previous.totalMessages);
    }

    expect(totalMessages).toBe(context.totalMessages);
  });

  it('上下同時押しを含むキーコードを除外する', () => {
    const context = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      conditions.keyInput = keyNamesToMask(['[↑]', '[↓]']);
      conditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC79 };
      return buildSearchContext(conditions);
    });

    const uniqueKeyCodes = new Set(context.segments.map(segment => segment.keyCode));
    expect(uniqueKeyCodes.size).toBe(3);
    for (const code of uniqueKeyCodes) {
      expect(hasImpossibleKeyCombination(keyCodeToMask(code))).toBe(false);
    }
  });

  it('左右同時押しを含むキーコードを除外する', () => {
    const context = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      conditions.keyInput = keyNamesToMask(['[←]', '[→]', 'A']);
      conditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC79 };
      return buildSearchContext(conditions);
    });

    const uniqueKeyCodes = new Set(context.segments.map(segment => segment.keyCode));
    expect(uniqueKeyCodes.size).toBe(6);
    for (const code of uniqueKeyCodes) {
      expect(hasImpossibleKeyCombination(keyCodeToMask(code))).toBe(false);
    }
  });

  it('Start+Select+L+R同時押しを含むキーコードを除外する', () => {
    const context = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      conditions.keyInput = keyNamesToMask(['Start', 'Select', 'L', 'R', 'B']);
      conditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC79 };
      return buildSearchContext(conditions);
    });

    const uniqueKeyCodes = new Set(context.segments.map(segment => segment.keyCode));
    expect(uniqueKeyCodes.size).toBe(30);
    for (const code of uniqueKeyCodes) {
      expect(hasImpossibleKeyCombination(keyCodeToMask(code))).toBe(false);
    }
  });
});
