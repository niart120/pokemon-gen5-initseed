import { describe, it, expect } from 'vitest';
import { SeedCalculator } from '@/lib/core/seed-calculator';
import { SHA1 } from '@/lib/core/sha1';
import { buildSearchContext } from '@/lib/webgpu/seed-search/message-encoder';
import type { SearchConditions } from '@/types/search';
import type { WebGpuSegment } from '@/lib/webgpu/seed-search/types';

function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

function monthDayFromDayOfYear(dayOfYear: number, leap: boolean): { month: number; day: number } {
  const monthLengths = leap
    ? [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    : [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let remaining = dayOfYear;
  for (let index = 0; index < monthLengths.length; index += 1) {
    const length = monthLengths[index]!;
    if (remaining <= length) {
      return { month: index + 1, day: remaining };
    }
    remaining -= length;
  }
  return { month: 12, day: 31 };
}

function simulateGpuIndices(segment: WebGpuSegment, messageIndex: number) {
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

function mulExtended(a: number, b: number): { lo: number; hi: number } {
  const product = BigInt(a >>> 0) * BigInt(b >>> 0);
  const lo = Number(product & 0xffffffffn) >>> 0;
  const hi = Number((product >> 32n) & 0xffffffffn) >>> 0;
  return { lo, hi };
}

function addWithCarry(a: number, b: number): { result: number; carry: number } {
  const sum = (a >>> 0) + (b >>> 0);
  const result = sum >>> 0;
  const carry = result < (a >>> 0) ? 1 : 0;
  return { result, carry };
}

function computeGpuSeed(h0: number, h1: number): number {
  const le0 = swap32(h0 >>> 0);
  const le1 = swap32(h1 >>> 0);
  const mulLo = 0x6c078965 >>> 0;
  const mulHi = 0x5d588b65 >>> 0;
  const increment = 0x00269ec3 >>> 0;

  const prod0 = mulExtended(le0, mulLo);
  const prod1 = mulExtended(le0, mulHi);
  const prod2 = mulExtended(le1, mulLo);
  const inc = addWithCarry(prod0.lo, increment);

  let upper = prod0.hi >>> 0;
  upper = (upper + prod1.lo) >>> 0;
  upper = (upper + prod2.lo) >>> 0;
  upper = (upper + inc.carry) >>> 0;
  return upper >>> 0;
}

function toBcd(value: number): number {
  const tens = Math.floor(value / 10) >>> 0;
  const ones = value - tens * 10;
  return ((tens << 4) | ones) >>> 0;
}

function buildGpuMessage(
  segment: WebGpuSegment,
  simulated: ReturnType<typeof simulateGpuIndices>,
  hardware: SearchConditions['hardware']
): number[] {
  const dateYear = simulated.year % 100;
  const dateWord =
    (toBcd(dateYear) << 24) |
    (toBcd(simulated.month) << 16) |
    (toBcd(simulated.day) << 8) |
    toBcd(simulated.dayOfWeek);

  const isPmHardware = hardware === 'DS' || hardware === 'DS_LITE';
  const pmFlag = isPmHardware && simulated.hour >= 12 ? 1 : 0;
  const timeWord =
    (pmFlag << 30) |
    (toBcd(simulated.hour) << 24) |
    (toBcd(simulated.minute) << 16) |
    (toBcd(simulated.second) << 8);

  const nazo = segment.config.nazoSwapped;

  return [
    nazo[0]!,
    nazo[1]!,
    nazo[2]!,
    nazo[3]!,
    nazo[4]!,
    swap32(((simulated.vcount << 16) | simulated.timer0) >>> 0),
    segment.config.macLower >>> 0,
    segment.config.data7Swapped >>> 0,
    dateWord >>> 0,
    timeWord >>> 0,
    0,
    0,
    segment.config.keyInputSwapped >>> 0,
    0x80000000,
    0,
    0x000001a0,
  ];
}

describe('webgpu seed search message mapping', () => {
  const calculator = new SeedCalculator();
  const sha1 = new SHA1();
  const conditions: SearchConditions = {
    romVersion: 'W2',
    romRegion: 'JPN',
    hardware: 'DS',
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
      endSecond: 4,
    },
    keyInput: 0x0000,
    macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
  };

  const context = buildSearchContext(conditions);

  it('matches CPU enumeration order', () => {
    const segment = context.segments[0];
    const expectedOrder: Array<{ timer0: number; vcount: number; secondOffset: number }> = [];

    for (let timer0 = segment.timer0Min; timer0 <= segment.timer0Max; timer0 += 1) {
      for (let secondOffset = 0; secondOffset < segment.rangeSeconds; secondOffset += 1) {
        expectedOrder.push({
          timer0,
          vcount: segment.vcount,
          secondOffset,
        });
      }
    }

    for (let index = 0; index < segment.totalMessages; index += 1) {
      const expected = expectedOrder[index]!;
      const simulated = simulateGpuIndices(segment, index);

      expect(simulated.timer0).toBe(expected.timer0);
      expect(simulated.vcount).toBe(expected.vcount);
      expect(simulated.secondOffset).toBe(expected.secondOffset);

      const expectedDatetime = new Date(context.startTimestampMs + simulated.secondOffset * 1000);
      expect(expectedDatetime.getFullYear()).toBe(simulated.year);
      expect(expectedDatetime.getMonth() + 1).toBe(simulated.month);
      expect(expectedDatetime.getDate()).toBe(simulated.day);
      expect(expectedDatetime.getHours()).toBe(simulated.hour);
      expect(expectedDatetime.getMinutes()).toBe(simulated.minute);
      expect(expectedDatetime.getSeconds()).toBe(simulated.second);
      expect(expectedDatetime.getDay()).toBe(simulated.dayOfWeek);

  const message = calculator.generateMessage(conditions, simulated.timer0, simulated.vcount, expectedDatetime);
  const gpuMessage = buildGpuMessage(segment, simulated, conditions.hardware);
  expect(gpuMessage).toEqual(message);
  const hash = sha1.calculateHash(message);
  const seedCpu = calculator.calculateSeed(message).seed;
      const seedGpu = computeGpuSeed(hash.h0, hash.h1);
      expect(seedGpu).toBe(seedCpu);
    }
  });

  it('handles day rollover consistently', () => {
    const rolloverConditions: SearchConditions = {
      ...conditions,
      dateRange: {
        startYear: 2012,
        endYear: 2012,
        startMonth: 6,
        endMonth: 6,
        startDay: 12,
        endDay: 13,
        startHour: 23,
        endHour: 0,
        startMinute: 59,
        endMinute: 0,
        startSecond: 58,
        endSecond: 2,
      },
    };

    const rolloverContext = buildSearchContext(rolloverConditions);
    const rolloverSegment = rolloverContext.segments[0];
    for (let index = 0; index < rolloverSegment.totalMessages; index += 1) {
      const simulated = simulateGpuIndices(rolloverSegment, index);
      const expectedDatetime = new Date(rolloverContext.startTimestampMs + simulated.secondOffset * 1000);
      const message = calculator.generateMessage(
        rolloverConditions,
        simulated.timer0,
        simulated.vcount,
        expectedDatetime
      );
      const gpuMessage = buildGpuMessage(rolloverSegment, simulated, rolloverConditions.hardware);
      expect(gpuMessage).toEqual(message);
    }
  });
});
