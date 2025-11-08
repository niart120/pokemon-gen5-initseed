/// <reference types="@webgpu/types" />

import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { SeedCalculator } from '@/lib/core/seed-calculator';
import { SHA1 } from '@/lib/core/sha1';
import { buildSearchContext } from '@/lib/webgpu/seed-search/message-encoder';
import { createWebGpuSeedSearchRunner, type WebGpuSeedSearchRunner } from '@/lib/webgpu/seed-search/runner';
import type { SearchConditions } from '@/types/search';
import type { WebGpuSegment } from '@/lib/webgpu/seed-search/types';

const hasWebGpu = typeof navigator !== 'undefined' && navigator.gpu !== undefined && navigator.gpu !== null;
const describeWebGpu = hasWebGpu ? describe : describe.skip;

interface SimulatedIndices {
  timer0: number;
  vcount: number;
  secondOffset: number;
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  dayOfWeek: number;
}

function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

function monthDayFromDayOfYear(dayOfYear: number, leap: boolean): { month: number; day: number } {
  const lengths = leap ? [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] : [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let remaining = dayOfYear;
  for (let index = 0; index < lengths.length; index += 1) {
    const length = lengths[index]!;
    if (remaining <= length) {
      return { month: index + 1, day: remaining };
    }
    remaining -= length;
  }
  return { month: 12, day: 31 };
}

function simulateGpuIndices(segment: WebGpuSegment, messageIndex: number): SimulatedIndices {
  const safeRangeSeconds = Math.max(segment.rangeSeconds, 1);
  const safeVcountCount = Math.max(segment.config.vcountCount, 1);
  const messagesPerVcount = safeRangeSeconds;
  const messagesPerTimer0 = messagesPerVcount * safeVcountCount;

  const timer0Index = Math.floor(messageIndex / messagesPerTimer0);
  const remainderAfterTimer0 = messageIndex - timer0Index * messagesPerTimer0;
  const vcountIndex = Math.floor(remainderAfterTimer0 / messagesPerVcount);
  const secondOffset = remainderAfterTimer0 - vcountIndex * messagesPerVcount;

  const timer0 = segment.config.timer0Min + timer0Index;
  const vcount = segment.config.vcountMin + vcountIndex;

  const totalSeconds = segment.config.startSecondOfDay + secondOffset;
  const dayOffset = Math.floor(totalSeconds / 86400);
  const secondsOfDay = totalSeconds - dayOffset * 86400;

  const hour = Math.floor(secondsOfDay / 3600);
  const minute = Math.floor((secondsOfDay % 3600) / 60);
  const second = secondsOfDay % 60;

  let year = segment.config.startYear;
  let dayOfYear = segment.config.startDayOfYear + dayOffset;
  while (true) {
    const yearLength = isLeapYear(year) ? 366 : 365;
    if (dayOfYear <= yearLength) {
      break;
    }
    dayOfYear -= yearLength;
    year += 1;
  }

  const leap = isLeapYear(year);
  const { month, day } = monthDayFromDayOfYear(dayOfYear, leap);
  const dayOfWeek = (segment.config.startDayOfWeek + dayOffset) % 7;

  return {
    timer0,
    vcount,
    secondOffset,
    year,
    month,
    day,
    hour,
    minute,
    second,
    dayOfWeek,
  };
}

function swap32(value: number): number {
  return (
    ((value & 0xff) << 24) |
    (((value >>> 8) & 0xff) << 16) |
    (((value >>> 16) & 0xff) << 8) |
    ((value >>> 24) & 0xff)
  ) >>> 0;
}

describeWebGpu('WebGPU seed search runner', () => {
  let runner: WebGpuSeedSearchRunner | null = null;

  beforeAll(async () => {
    runner = createWebGpuSeedSearchRunner({ workgroupSize: 64 });
    await runner.init();
  });

  afterAll(() => {
    runner?.dispose();
    runner = null;
  });

  it('matches CPU results for a small search window', async () => {
    const calculator = new SeedCalculator();
    const sha1 = new SHA1();

    const conditions: SearchConditions = {
      romVersion: 'W2',
      romRegion: 'JPN',
      hardware: 'DS',
      keyInput: 0x0000,
      macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
      timer0VCountConfig: {
        useAutoConfiguration: false,
        timer0Range: { min: 0x10f5, max: 0x10f6 },
        vcountRange: { min: 0x82, max: 0x82 },
      },
      dateRange: {
        startYear: 2012,
        endYear: 2012,
        startMonth: 6,
        endMonth: 6,
        startDay: 12,
        endDay: 12,
        startHour: 10,
        endHour: 10,
        startMinute: 15,
        endMinute: 15,
        startSecond: 0,
        endSecond: 3,
      },
    };

    const context = buildSearchContext(conditions);
    const expected: Array<{ seed: number; timer0: number; vcount: number; iso: string }> = [];

    for (const segment of context.segments) {
      for (let index = 0; index < segment.totalMessages; index += 1) {
        const simulated = simulateGpuIndices(segment, index);
        const datetime = new Date(context.startTimestampMs + simulated.secondOffset * 1000);
        const message = calculator.generateMessage(conditions, simulated.timer0, simulated.vcount, datetime);
        const hash = sha1.calculateHash(message);
        const seed = calculator.calculateSeed(message).seed;
        expected.push({ seed, timer0: simulated.timer0, vcount: simulated.vcount, iso: datetime.toISOString() });
      }
    }

    const collected: Array<{ seed: number; timer0: number; vcount: number; iso: string }> = [];
    let error: Error | null = null;
    let completed = false;

    await runner!.run({
      context,
  targetSeeds: [],
      callbacks: {
        onProgress: () => {},
        onResult: (result) => {
          collected.push({
            seed: result.seed >>> 0,
            timer0: result.timer0,
            vcount: result.vcount,
            iso: result.datetime.toISOString(),
          });
        },
        onComplete: () => {
          completed = true;
        },
        onError: (message) => {
          error = new Error(message);
        },
        onPaused: () => {},
        onResumed: () => {},
        onStopped: () => {},
      },
    });

    if (error) {
      throw error;
    }

    expect(completed).toBe(true);
  const sortedExpected = expected
      .map((entry) => ({ ...entry, seed: entry.seed >>> 0 }))
      .sort((a, b) => a.seed - b.seed || a.timer0 - b.timer0 || a.vcount - b.vcount || a.iso.localeCompare(b.iso));
    const sortedCollected = collected
      .map((entry) => ({ ...entry, seed: entry.seed >>> 0 }))
      .sort((a, b) => a.seed - b.seed || a.timer0 - b.timer0 || a.vcount - b.vcount || a.iso.localeCompare(b.iso));

    expect(sortedCollected).toEqual(sortedExpected);
  });
});
