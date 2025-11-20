import { describe, it, expect } from 'vitest';
import { iterateAllowedSubChunks } from '@/lib/search/time/sub-chunk-planner';
import { resolveTimePlan } from '@/lib/search/time/time-plan';
import type { SearchConditions } from '@/types/search';

const BASE_CONDITIONS: SearchConditions = {
  romVersion: 'B2',
  romRegion: 'JPN',
  hardware: 'DS',
  timer0VCountConfig: {
    useAutoConfiguration: false,
    timer0Range: { min: 0xC79, max: 0xC79 },
    vcountRange: { min: 0x90, max: 0x90 },
  },
  timeRange: {
    hour: { start: 5, end: 5 },
    minute: { start: 0, end: 0 },
    second: { start: 0, end: 59 },
  },
  dateRange: {
    startYear: 2013,
    endYear: 2013,
    startMonth: 1,
    endMonth: 1,
    startDay: 1,
    endDay: 3,
    startHour: 0,
    endHour: 23,
    startMinute: 0,
    endMinute: 59,
    startSecond: 0,
    endSecond: 59,
  },
  keyInput: 0,
  macAddress: [0, 0, 0, 0, 0, 0],
};

function createConditions(overrides: Partial<SearchConditions> = {}): SearchConditions {
  return {
    ...BASE_CONDITIONS,
    ...overrides,
    timer0VCountConfig: overrides.timer0VCountConfig ?? BASE_CONDITIONS.timer0VCountConfig,
    timeRange: overrides.timeRange ?? BASE_CONDITIONS.timeRange,
    dateRange: overrides.dateRange ?? BASE_CONDITIONS.dateRange,
  };
}

describe('iterateAllowedSubChunks', () => {
  it('generates contiguous windows when allowed seconds are continuous', () => {
    const contiguousConditions = createConditions({
      timeRange: {
        hour: { start: 5, end: 5 },
        minute: { start: 0, end: 0 },
        second: { start: 0, end: 59 },
      },
    });
    const { plan } = resolveTimePlan(contiguousConditions);
    const chunkStartMs = new Date(2013, 0, 1, 5, 0, 0).getTime();
    const chunkEndMs = new Date(2013, 0, 1, 6, 0, 0).getTime() + 1000;

    const windows = Array.from(
      iterateAllowedSubChunks({
        plan,
        chunkStartMs,
        chunkEndExclusiveMs: chunkEndMs,
        desiredAllowedSeconds: 30,
        maxSecondsPerChunk: 600,
      })
    );

    expect(windows).toHaveLength(2);
    expect(windows[0].countedSeconds).toBe(30);
    expect(windows[0].durationSeconds).toBe(30);
    expect(windows[0].startTimestampMs).toBe(chunkStartMs);

    expect(windows[1].countedSeconds).toBe(30);
    expect(windows[1].durationSeconds).toBe(30);
    expect(windows[1].startTimestampMs).toBe(chunkStartMs + 30 * 1000);
  });

  it('extends timeline duration when chunk spans multiple days', () => {
    const { plan } = resolveTimePlan(BASE_CONDITIONS);
    const chunkStartMs = new Date(2013, 0, 1, 5, 0, 0).getTime();
    const chunkEndMs = new Date(2013, 0, 3, 0, 0, 0).getTime() + 1000;

    const [window] = Array.from(
      iterateAllowedSubChunks({
        plan,
        chunkStartMs,
        chunkEndExclusiveMs: chunkEndMs,
        desiredAllowedSeconds: 90,
        maxSecondsPerChunk: 3 * 24 * 60 * 60,
      })
    );

    expect(window.countedSeconds).toBe(90);
    expect(window.durationSeconds).toBeGreaterThan(window.countedSeconds);
    expect(window.startTimestampMs).toBe(chunkStartMs);
    expect(window.durationSeconds).toBe(86430);
  });

  it('respects the maximum timeline duration per chunk', () => {
    const wideMinuteRangeConditions = createConditions({
      timeRange: {
        hour: { start: 5, end: 5 },
        minute: { start: 0, end: 9 },
        second: { start: 0, end: 59 },
      },
    });
    const { plan } = resolveTimePlan(wideMinuteRangeConditions);
    const chunkStartMs = new Date(2013, 0, 1, 5, 0, 0).getTime();
    const chunkEndMs = new Date(2013, 0, 1, 6, 0, 0).getTime() + 1000;

    const windows = Array.from(
      iterateAllowedSubChunks({
        plan,
        chunkStartMs,
        chunkEndExclusiveMs: chunkEndMs,
        desiredAllowedSeconds: 600,
        maxSecondsPerChunk: 60,
      })
    );

    expect(windows.length).toBeGreaterThan(1);
    windows.forEach((window) => {
      expect(window.durationSeconds).toBeLessThanOrEqual(60);
      expect(window.countedSeconds).toBeLessThanOrEqual(window.durationSeconds);
    });
  });
});
