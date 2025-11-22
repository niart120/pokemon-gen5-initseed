import { describe, it, expect } from 'vitest';
import { SeedCalculator } from '@/lib/core/seed-calculator';
import { SHA1 } from '@/lib/core/sha1';
import { prepareSearchJob } from '@/lib/webgpu/seed-search/prepare-search-job';
import { getDateFromTimePlan } from '@/lib/search/time/time-plan';
import type { SeedSearchJobSegment } from '@/lib/webgpu/seed-search/types';
import type { SearchConditions } from '@/types/search';
import { createTestSeedSearchJobLimits } from './seed-search-job-limit-helpers';

const TEST_LIMITS = createTestSeedSearchJobLimits();

interface SimulatedIndices {
  timer0: number;
  vcount: number;
  timeCombinationIndex: number;
  datetime: Date;
}

function simulateGpuIndices(
  segment: SeedSearchJobSegment,
  messageIndex: number,
  timePlanStart: (timeIndex: number) => Date
): SimulatedIndices {
  const timeCombinationIndex = segment.baseSecondOffset + messageIndex;
  const timer0 = segment.timer0;
  const vcount = segment.vcount;
  const datetime = timePlanStart(timeCombinationIndex);

  return { timer0, vcount, timeCombinationIndex, datetime };
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

function decodeSegmentConstants(segment: SeedSearchJobSegment) {
  const words = segment.uniformWords;
  return {
    macLower: words[1] >>> 0,
    data7Swapped: words[2] >>> 0,
    keyInputSwapped: words[3] >>> 0,
    nazoSwapped: words.slice(14, 19),
  };
}

function buildGpuMessage(
  segment: SeedSearchJobSegment,
  simulated: SimulatedIndices,
  hardware: SearchConditions['hardware']
): number[] {
  const config = decodeSegmentConstants(segment);
  const datetime = simulated.datetime;
  const dateYear = datetime.getFullYear() % 100;
  const dateWord =
    (toBcd(dateYear) << 24) |
    (toBcd(datetime.getMonth() + 1) << 16) |
    (toBcd(datetime.getDate()) << 8) |
    toBcd(datetime.getDay());

  const isPmHardware = hardware === 'DS' || hardware === 'DS_LITE';
  const pmFlag = isPmHardware && datetime.getHours() >= 12 ? 1 : 0;
  const timeWord =
    (pmFlag << 30) |
    (toBcd(datetime.getHours()) << 24) |
    (toBcd(datetime.getMinutes()) << 16) |
    (toBcd(datetime.getSeconds()) << 8);

  const nazo = config.nazoSwapped;

  return [
    nazo[0]!,
    nazo[1]!,
    nazo[2]!,
    nazo[3]!,
    nazo[4]!,
    swap32(((simulated.vcount << 16) | simulated.timer0) >>> 0),
    config.macLower >>> 0,
    config.data7Swapped >>> 0,
    dateWord >>> 0,
    timeWord >>> 0,
    0,
    0,
    config.keyInputSwapped >>> 0,
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
    timeRange: {
      hour: { start: 10, end: 10 },
      minute: { start: 15, end: 15 },
      second: { start: 0, end: 4 },
    },
    keyInput: 0x0000,
    macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
  };

  const job = prepareSearchJob(conditions, undefined, { limits: TEST_LIMITS });
  const primarySegment = job.segments[0]!;
  const timePlanResolver = (timeIndex: number) => getDateFromTimePlan(job.timePlan, timeIndex);

  it('matches CPU enumeration order', () => {
    const expectedOrder = Array.from({ length: primarySegment.messageCount }, (_, index) => ({
      timer0: primarySegment.timer0,
      vcount: primarySegment.vcount,
      timeIndex: primarySegment.baseSecondOffset + index,
    }));

    for (let index = 0; index < primarySegment.messageCount; index += 1) {
      const expected = expectedOrder[index]!;
      const simulated = simulateGpuIndices(primarySegment, index, timePlanResolver);

      expect(simulated.timer0).toBe(expected.timer0);
      expect(simulated.vcount).toBe(expected.vcount);
      expect(simulated.timeCombinationIndex).toBe(expected.timeIndex);

      const expectedDatetime = timePlanResolver(expected.timeIndex);
      expect(simulated.datetime.getTime()).toBe(expectedDatetime.getTime());

      const message = calculator.generateMessage(
        conditions,
        simulated.timer0,
        simulated.vcount,
        expectedDatetime,
        primarySegment.keyCode
      );
      const gpuMessage = buildGpuMessage(primarySegment, simulated, conditions.hardware);
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
        startHour: 0,
        endHour: 0,
        startMinute: 0,
        endMinute: 0,
        startSecond: 0,
        endSecond: 0,
      },
      timeRange: {
        hour: { start: 23, end: 23 },
        minute: { start: 59, end: 59 },
        second: { start: 58, end: 59 },
      },
    };

    const rolloverJob = prepareSearchJob(rolloverConditions, undefined, { limits: TEST_LIMITS });
    const rolloverSegment = rolloverJob.segments[0]!;
    const resolveRolloverTime = (timeIndex: number) => getDateFromTimePlan(rolloverJob.timePlan, timeIndex);

    for (let index = 0; index < rolloverSegment.messageCount; index += 1) {
      const simulated = simulateGpuIndices(rolloverSegment, index, resolveRolloverTime);
      const expectedDatetime = simulated.datetime;
      const message = calculator.generateMessage(
        rolloverConditions,
        simulated.timer0,
        simulated.vcount,
        expectedDatetime,
        rolloverSegment.keyCode
      );
      const gpuMessage = buildGpuMessage(rolloverSegment, simulated, rolloverConditions.hardware);
      expect(gpuMessage).toEqual(message);
    }
  });
});
