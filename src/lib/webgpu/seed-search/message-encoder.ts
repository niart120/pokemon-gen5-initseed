import romParameters from '@/data/rom-parameters';
import type { Hardware } from '@/types/rom';
import type { SearchConditions } from '@/types/search';
import type { GpuSha1WorkloadConfig, WebGpuSearchContext, WebGpuSegment } from './types';

const EPOCH_2000_MS = Date.UTC(2000, 0, 1, 0, 0, 0);
const GX_STAT = 0x06000000;

const HARDWARE_FRAME_VALUES: Record<Hardware, number> = {
  DS: 8,
  DS_LITE: 6,
  '3DS': 9,
};

export function buildSearchContext(conditions: SearchConditions): WebGpuSearchContext {
  const startDate = buildDate(
    conditions.dateRange.startYear,
    conditions.dateRange.startMonth,
    conditions.dateRange.startDay,
    conditions.dateRange.startHour,
    conditions.dateRange.startMinute,
    conditions.dateRange.startSecond
  );

  const endDate = buildDate(
    conditions.dateRange.endYear,
    conditions.dateRange.endMonth,
    conditions.dateRange.endDay,
    conditions.dateRange.endHour,
    conditions.dateRange.endMinute,
    conditions.dateRange.endSecond
  );

  if (startDate.getTime() > endDate.getTime()) {
    throw new Error('開始日時が終了日時より後ろに設定されています');
  }

  const params = resolveRomParameters(conditions);
  const timer0Segments = resolveTimer0Segments(conditions, params);

  const startEpochMs = calculateEpoch2000TimestampMs(
    conditions.dateRange.startYear,
    conditions.dateRange.startMonth,
    conditions.dateRange.startDay,
    conditions.dateRange.startHour,
    conditions.dateRange.startMinute,
    conditions.dateRange.startSecond
  );

  const endEpochMs = calculateEpoch2000TimestampMs(
    conditions.dateRange.endYear,
    conditions.dateRange.endMonth,
    conditions.dateRange.endDay,
    conditions.dateRange.endHour,
    conditions.dateRange.endMinute,
    conditions.dateRange.endSecond
  );

  const startSecondsSince2000 = Math.floor((startEpochMs - EPOCH_2000_MS) / 1000);
  if (startSecondsSince2000 < 0) {
    throw new Error('2000年より前の日時は指定できません');
  }

  const rangeSeconds = Math.floor((endEpochMs - startEpochMs) / 1000) + 1;
  if (rangeSeconds <= 0) {
    throw new Error('探索秒数が0秒以下です');
  }

  const startYear = startDate.getFullYear();
  const startDayOfYear = calculateLocalDayOfYear(startDate);
  const startSecondOfDay = calculateSecondOfDay(startDate);
  const startDayOfWeek = startDate.getDay();
  const frameValue = HARDWARE_FRAME_VALUES[conditions.hardware];

  const { macLower, data7Swapped } = computeMacWords(conditions.macAddress, frameValue);
  const keyInputSwapped = swap32(conditions.keyInput >>> 0);
  const nazoSwapped = createNazoSwapped(params.nazo);

  const segments: WebGpuSegment[] = [];
  let baseOffset = 0;
  for (let index = 0; index < timer0Segments.length; index += 1) {
    const segmentInfo = timer0Segments[index];
    const timer0Count = segmentInfo.timer0Max - segmentInfo.timer0Min + 1;
    const totalMessages = rangeSeconds * timer0Count;

    const config: GpuSha1WorkloadConfig = {
      startSecondsSince2000: startSecondsSince2000 >>> 0,
      rangeSeconds: rangeSeconds >>> 0,
      timer0Min: segmentInfo.timer0Min >>> 0,
      timer0Max: segmentInfo.timer0Max >>> 0,
      timer0Count: timer0Count >>> 0,
      vcountMin: segmentInfo.vcount >>> 0,
      vcountMax: segmentInfo.vcount >>> 0,
      vcountCount: 1 >>> 0,
      totalMessages: totalMessages >>> 0,
      hardwareType: mapHardwareToId(conditions.hardware),
      macLower: macLower >>> 0,
      data7Swapped: data7Swapped >>> 0,
      keyInputSwapped: keyInputSwapped >>> 0,
      nazoSwapped,
      startYear: startYear >>> 0,
      startDayOfYear: startDayOfYear >>> 0,
      startSecondOfDay: startSecondOfDay >>> 0,
      startDayOfWeek: startDayOfWeek >>> 0,
    };

    segments.push({
      index,
      baseOffset,
      timer0Min: segmentInfo.timer0Min,
      timer0Max: segmentInfo.timer0Max,
      timer0Count,
      vcount: segmentInfo.vcount,
      rangeSeconds,
      totalMessages,
      config,
    });

    baseOffset += totalMessages;
  }

  const totalMessages = segments.reduce((sum, segment) => sum + segment.totalMessages, 0);

  return {
    conditions,
    startDate,
    startTimestampMs: startDate.getTime(),
      rangeSeconds,
    totalMessages,
    segments,
  };
}

function buildDate(year: number, month: number, day: number, hour: number, minute: number, second: number): Date {
  return new Date(year, month - 1, day, hour, minute, second);
}

function calculateEpoch2000TimestampMs(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number
): number {
  return Date.UTC(year, month - 1, day, hour, minute, second, 0);
}

function resolveRomParameters(conditions: SearchConditions) {
  const versionData = romParameters[conditions.romVersion as keyof typeof romParameters];
  if (!versionData) {
    throw new Error(`ROMバージョン ${conditions.romVersion} は未対応です`);
  }

  const regionData = versionData[conditions.romRegion as keyof typeof versionData];
  if (!regionData) {
    throw new Error(`ROMリージョン ${conditions.romRegion} は未対応です`);
  }

  return {
    nazo: [...regionData.nazo] as [number, number, number, number, number],
    vcountTimerRanges: regionData.vcountTimerRanges.map((entry) => [...entry] as [number, number, number]),
  };
}

function resolveTimer0Segments(
  conditions: SearchConditions,
  params: { vcountTimerRanges: readonly [number, number, number][] }
): Array<{ timer0Min: number; timer0Max: number; vcount: number }> {
  const segments: Array<{ timer0Min: number; timer0Max: number; vcount: number }> = [];
  const timer0Min = conditions.timer0VCountConfig.timer0Range.min;
  const timer0Max = conditions.timer0VCountConfig.timer0Range.max;

  let current: { timer0Min: number; timer0Max: number; vcount: number } | null = null;

  for (let timer0 = timer0Min; timer0 <= timer0Max; timer0 += 1) {
    const vcount = getVCountForTimer0(params, timer0);

    if (current && current.vcount === vcount && timer0 === current.timer0Max + 1) {
      current.timer0Max = timer0;
    } else {
      if (current) {
        segments.push(current);
      }
      current = { timer0Min: timer0, timer0Max: timer0, vcount };
    }
  }

  if (current) {
    segments.push(current);
  }

  return segments;
}

function getVCountForTimer0(params: { vcountTimerRanges: readonly [number, number, number][] }, timer0: number): number {
  for (const [vcount, min, max] of params.vcountTimerRanges) {
    if (timer0 >= min && timer0 <= max) {
      return vcount;
    }
  }
  return params.vcountTimerRanges.length > 0 ? params.vcountTimerRanges[0][0] : 0x60;
}

function computeMacWords(mac: number[], frame: number): { macLower: number; data7Swapped: number } {
  const safeMac = normalizeMac(mac);
  const macLower = ((safeMac[4] & 0xff) << 8) | (safeMac[5] & 0xff);
  const macUpper =
    (safeMac[0] & 0xff) |
    ((safeMac[1] & 0xff) << 8) |
    ((safeMac[2] & 0xff) << 16) |
    ((safeMac[3] & 0xff) << 24);
  const data7 = (macUpper ^ GX_STAT ^ frame) >>> 0;
  return { macLower, data7Swapped: swap32(data7) };
}

function normalizeMac(mac: number[]): number[] {
  const result = new Array(6).fill(0);
  for (let i = 0; i < 6; i += 1) {
    const value = mac[i] ?? 0;
    result[i] = (Number(value) & 0xff) >>> 0;
  }
  return result;
}

function mapHardwareToId(hardware: Hardware): number {
  switch (hardware) {
    case 'DS':
      return 0;
    case 'DS_LITE':
      return 1;
    case '3DS':
      return 2;
    default:
      return 0;
  }
}

function swap32(value: number): number {
  return (
    ((value & 0xff) << 24) |
    (((value >>> 8) & 0xff) << 16) |
    (((value >>> 16) & 0xff) << 8) |
    ((value >>> 24) & 0xff)
  ) >>> 0;
}

function createNazoSwapped(nazo: readonly number[]): Uint32Array {
  const array = new Uint32Array(nazo.length);
  for (let i = 0; i < nazo.length; i += 1) {
    array[i] = swap32(nazo[i] >>> 0);
  }
  return array;
}

function calculateLocalDayOfYear(date: Date): number {
  const startOfYear = new Date(date.getFullYear(), 0, 1);
  const diffMs = date.getTime() - startOfYear.getTime();
  return Math.floor(diffMs / (24 * 60 * 60 * 1000)) + 1;
}

function calculateSecondOfDay(date: Date): number {
  return date.getHours() * 3600 + date.getMinutes() * 60 + date.getSeconds();
}
