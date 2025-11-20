import { describe, expect, it, vi } from 'vitest';
import { hasImpossibleKeyCombination, keyCodeToMask, keyNamesToMask } from '@/lib/utils/key-input';
import { prepareSearchJob } from '@/lib/webgpu/seed-search/prepare-search-job';
import type { SearchConditions } from '@/types/search';
import { createTestSeedSearchJobLimits } from './seed-search-job-limit-helpers';

const REAL_DATE = Date;
const TEST_LIMITS = createTestSeedSearchJobLimits();

function prepareJob(conditions: SearchConditions) {
  return prepareSearchJob(conditions, undefined, { limits: TEST_LIMITS });
}

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

describe('prepareSearchJob', () => {
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

  it('computes start timestamp using local timezone information', () => {
    withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      const job = prepareJob(conditions);
      const expectedLocalTimestamp = new Date(
        singleDayRange.startYear,
        singleDayRange.startMonth - 1,
        singleDayRange.startDay,
        singleDayRange.startHour,
        singleDayRange.startMinute,
        singleDayRange.startSecond
      ).getTime();

      expect(job.timePlan.startDayTimestampMs).toBe(expectedLocalTimestamp);
    });
  });

  it('calculates seconds since Y2K epoch accurately for far future dates', () => {
    const futureRange: SearchConditions['dateRange'] = {
      ...singleDayRange,
      startYear: 2100,
      endYear: 2100,
    };

    withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(futureRange);
      const job = prepareJob(conditions);

      const expectedEpoch = new Date(2000, 0, 1, 0, 0, 0).getTime();
      const expectedStart = new Date(2100, 0, 1, 0, 0, 0).getTime();
      const expectedSeconds = Math.floor((expectedStart - expectedEpoch) / 1000);
      const actualSeconds = Math.floor((job.timePlan.startDayTimestampMs - expectedEpoch) / 1000);

      expect(actualSeconds).toBe(expectedSeconds);
    });
  });

  it('reflects time range boundaries in summary seconds and total messages', () => {
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

    const job = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(multiSecondRange);
      return prepareJob(conditions);
    });

    expect(job.summary.rangeSeconds).toBe(6);

    const summedMessages = job.segments.reduce((sum, segment) => sum + segment.messageCount, 0);
    expect(job.summary.totalMessages).toBe(summedMessages);
    expect(job.summary.totalMessages).toBe(12);
  });

  it('emits non-overlapping segments with contiguous message offsets', () => {
    const wideTimerRangeConditions = createSearchConditions({ ...singleDayRange });
    wideTimerRangeConditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC7C };

    const job = withMockedLocalTimezone(-540, () => prepareJob(wideTimerRangeConditions));

    expect(job.segments.length).toBeGreaterThan(0);

    for (let index = 1; index < job.segments.length; index += 1) {
      const previous = job.segments[index - 1]!;
      const current = job.segments[index]!;
      expect(current.globalMessageOffset).toBe(previous.globalMessageOffset + previous.messageCount);
    }

    const totalMessages = job.segments.reduce((sum, segment) => sum + segment.messageCount, 0);
    expect(totalMessages).toBe(job.summary.totalMessages);
  });

  it('filters key codes containing opposite directions (up/down)', () => {
    const job = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      conditions.keyInput = keyNamesToMask(['[↑]', '[↓]']);
      conditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC79 };
      return prepareJob(conditions);
    });

    const uniqueKeyCodes = new Set(job.segments.map((segment) => segment.keyCode));
    expect(uniqueKeyCodes.size).toBe(3);
    for (const code of uniqueKeyCodes) {
      expect(hasImpossibleKeyCombination(keyCodeToMask(code))).toBe(false);
    }
  });

  it('filters key codes containing opposite directions (left/right)', () => {
    const job = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      conditions.keyInput = keyNamesToMask(['[←]', '[→]', 'A']);
      conditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC79 };
      return prepareJob(conditions);
    });

    const uniqueKeyCodes = new Set(job.segments.map((segment) => segment.keyCode));
    expect(uniqueKeyCodes.size).toBe(6);
    for (const code of uniqueKeyCodes) {
      expect(hasImpossibleKeyCombination(keyCodeToMask(code))).toBe(false);
    }
  });

  it('filters Start+Select+L+R simultaneous presses', () => {
    const job = withMockedLocalTimezone(-540, () => {
      const conditions = createSearchConditions(singleDayRange);
      conditions.keyInput = keyNamesToMask(['Start', 'Select', 'L', 'R', 'B']);
      conditions.timer0VCountConfig.timer0Range = { min: 0xC79, max: 0xC79 };
      return prepareJob(conditions);
    });

    const uniqueKeyCodes = new Set(job.segments.map((segment) => segment.keyCode));
    expect(uniqueKeyCodes.size).toBe(30);
    for (const code of uniqueKeyCodes) {
      expect(hasImpossibleKeyCombination(keyCodeToMask(code))).toBe(false);
    }
  });
});
