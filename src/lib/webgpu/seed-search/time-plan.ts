import type { SearchConditions, TimeFieldRange } from '@/types/search';
import type { WebGpuTimePlan } from './types';

const MS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;
const HOURS_PER_DAY = 24;
const SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
const SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY;
const MS_PER_DAY = SECONDS_PER_DAY * MS_PER_SECOND;

interface ValidatedFieldRange {
  start: number;
  end: number;
  count: number;
}

export interface ResolvedTimePlan {
  plan: WebGpuTimePlan;
  firstCombinationDate: Date;
}

export function resolveTimePlan(conditions: SearchConditions): ResolvedTimePlan {
  const timeRange = conditions.timeRange;
  if (!timeRange) {
    throw new Error('timeRange is required for WebGPU search');
  }

  const hourRange = validateFieldRange('hour', timeRange.hour, 0, HOURS_PER_DAY - 1);
  const minuteRange = validateFieldRange('minute', timeRange.minute, 0, MINUTES_PER_HOUR - 1);
  const secondRange = validateFieldRange('second', timeRange.second, 0, SECONDS_PER_MINUTE - 1);

  const startDayDate = new Date(
    conditions.dateRange.startYear,
    conditions.dateRange.startMonth - 1,
    conditions.dateRange.startDay,
    0,
    0,
    0
  );
  const endDayDate = new Date(
    conditions.dateRange.endYear,
    conditions.dateRange.endMonth - 1,
    conditions.dateRange.endDay,
    0,
    0,
    0
  );

  const startDayMs = startDayDate.getTime();
  const endDayMs = endDayDate.getTime();
  if (startDayMs > endDayMs) {
    throw new Error('開始日が終了日より後に設定されています');
  }

  const dayCount = Math.floor((endDayMs - startDayMs) / MS_PER_DAY) + 1;
  if (dayCount <= 0) {
    throw new Error('探索日数が検出できませんでした');
  }

  const combosPerDay = hourRange.count * minuteRange.count * secondRange.count;
  if (combosPerDay <= 0) {
    throw new Error('時刻レンジの組み合わせ数が0です');
  }

  const firstCombinationDate = new Date(
    conditions.dateRange.startYear,
    conditions.dateRange.startMonth - 1,
    conditions.dateRange.startDay,
    hourRange.start,
    minuteRange.start,
    secondRange.start,
    0
  );

  const plan: WebGpuTimePlan = {
    dayCount,
    combosPerDay,
    hourRangeStart: hourRange.start,
    hourRangeCount: hourRange.count,
    minuteRangeStart: minuteRange.start,
    minuteRangeCount: minuteRange.count,
    secondRangeStart: secondRange.start,
    secondRangeCount: secondRange.count,
    startDayTimestampMs: startDayMs,
  };

  return {
    plan,
    firstCombinationDate,
  };
}

export function getDateFromTimePlan(plan: WebGpuTimePlan, timeIndex: number): Date {
  const safeMinuteCount = Math.max(plan.minuteRangeCount, 1);
  const safeSecondCount = Math.max(plan.secondRangeCount, 1);
  const combosPerDay = Math.max(plan.combosPerDay, 1);

  const clampedIndex = Math.max(0, Math.trunc(timeIndex));
  const dayIndex = Math.floor(clampedIndex / combosPerDay);
  const remainderAfterDay = clampedIndex - dayIndex * combosPerDay;

  const entriesPerHour = safeMinuteCount * safeSecondCount;
  const hourIndex = Math.floor(remainderAfterDay / entriesPerHour);
  const remainderAfterHour = remainderAfterDay - hourIndex * entriesPerHour;
  const minuteIndex = Math.floor(remainderAfterHour / safeSecondCount);
  const secondIndex = remainderAfterHour - minuteIndex * safeSecondCount;

  const hour = plan.hourRangeStart + hourIndex;
  const minute = plan.minuteRangeStart + minuteIndex;
  const second = plan.secondRangeStart + secondIndex;

  const dayTimestampMs = plan.startDayTimestampMs + dayIndex * MS_PER_DAY;
  const totalMs =
    dayTimestampMs +
    hour * SECONDS_PER_HOUR * MS_PER_SECOND +
    minute * SECONDS_PER_MINUTE * MS_PER_SECOND +
    second * MS_PER_SECOND;

  return new Date(totalMs);
}

function validateFieldRange(
  label: string,
  range: TimeFieldRange | undefined,
  min: number,
  max: number
): ValidatedFieldRange {
  if (!range) {
    throw new Error(`${label} range is required for WebGPU search`);
  }
  const start = Math.trunc(range.start);
  const end = Math.trunc(range.end);
  if (Number.isNaN(start) || Number.isNaN(end)) {
    throw new Error(`${label} range must be numeric`);
  }
  if (start < min || end > max) {
    throw new Error(`${label} range must be within ${min} to ${max}`);
  }
  if (start > end) {
    throw new Error(`${label} range start must be <= end`);
  }
  return {
    start,
    end,
    count: end - start + 1,
  };
}
