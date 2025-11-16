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

interface DailyWindowBounds {
  startSecond: number;
  endSecondExclusive: number;
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

export function countAllowedSecondsInInterval(
  plan: WebGpuTimePlan,
  startTimestampMs: number,
  endTimestampMsExclusive: number
): number {
  if (endTimestampMsExclusive <= startTimestampMs) {
    return 0;
  }

  const bounds = getDailyWindowBounds(plan);
  let cursor = startTimestampMs;
  let total = 0;

  while (cursor < endTimestampMsExclusive) {
    const { dayStartMs, dayEndMs } = getDayBounds(plan, cursor);
    const segmentEnd = Math.min(dayEndMs, endTimestampMsExclusive);

    const segmentStartSecond = Math.max(0, Math.floor((cursor - dayStartMs) / MS_PER_SECOND));
    const segmentEndSecondExclusive = Math.max(
      segmentStartSecond,
      Math.min(SECONDS_PER_DAY, Math.ceil((segmentEnd - dayStartMs) / MS_PER_SECOND))
    );

    const overlapStart = Math.max(segmentStartSecond, bounds.startSecond);
    const overlapEndExclusive = Math.min(segmentEndSecondExclusive, bounds.endSecondExclusive);

    if (overlapStart < overlapEndExclusive) {
      total += overlapEndExclusive - overlapStart;
    }

    cursor = segmentEnd;
  }

  return total;
}

export function advanceByAllowedSeconds(
  plan: WebGpuTimePlan,
  startTimestampMs: number,
  endTimestampMsExclusive: number,
  targetAllowedSeconds: number
): {
  endTimestampMs: number;
  countedSeconds: number;
  lastAllowedTimestampMs?: number;
} {
  if (endTimestampMsExclusive <= startTimestampMs) {
    return { endTimestampMs: endTimestampMsExclusive, countedSeconds: 0 };
  }

  const desired = Math.max(0, targetAllowedSeconds);
  const bounds = getDailyWindowBounds(plan);
  let cursor = startTimestampMs;
  let counted = 0;
  let lastAllowed: number | undefined;

  while (cursor < endTimestampMsExclusive) {
    const { dayStartMs, dayEndMs } = getDayBounds(plan, cursor);
    const segmentEnd = Math.min(dayEndMs, endTimestampMsExclusive);

    const segmentStartSecond = Math.max(0, Math.floor((cursor - dayStartMs) / MS_PER_SECOND));
    const segmentEndSecondExclusive = Math.max(
      segmentStartSecond,
      Math.min(SECONDS_PER_DAY, Math.ceil((segmentEnd - dayStartMs) / MS_PER_SECOND))
    );

    const overlapStartSecond = Math.max(segmentStartSecond, bounds.startSecond);
    const overlapEndSecondExclusive = Math.min(segmentEndSecondExclusive, bounds.endSecondExclusive);

    if (overlapStartSecond < overlapEndSecondExclusive) {
      const overlapStartMs = dayStartMs + overlapStartSecond * MS_PER_SECOND;
      const available = overlapEndSecondExclusive - overlapStartSecond;
      const needed = desired > 0 ? Math.min(available, desired - counted) : available;

      if (needed > 0) {
        counted += needed;
        lastAllowed = overlapStartMs + (needed - 1) * MS_PER_SECOND;
        const consumedEndMs = overlapStartMs + needed * MS_PER_SECOND;

        if (desired > 0 && counted >= desired) {
          return {
            endTimestampMs: consumedEndMs,
            countedSeconds: counted,
            lastAllowedTimestampMs: lastAllowed,
          };
        }

        cursor = Math.max(consumedEndMs, segmentEnd);
        continue;
      }
    }

    cursor = segmentEnd;
  }

  return {
    endTimestampMs: cursor,
    countedSeconds: counted,
    lastAllowedTimestampMs: lastAllowed,
  };
}

export function isDateWithinTimePlan(date: Date, plan: WebGpuTimePlan): boolean {
  const secondOfDay =
    date.getHours() * SECONDS_PER_HOUR + date.getMinutes() * SECONDS_PER_MINUTE + date.getSeconds();
  const bounds = getDailyWindowBounds(plan);
  return secondOfDay >= bounds.startSecond && secondOfDay < bounds.endSecondExclusive;
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

function getDailyWindowBounds(plan: WebGpuTimePlan): DailyWindowBounds {
  const startSecond =
    plan.hourRangeStart * SECONDS_PER_HOUR +
    plan.minuteRangeStart * SECONDS_PER_MINUTE +
    plan.secondRangeStart;

  const endSecondExclusive = Math.min(startSecond + plan.combosPerDay, SECONDS_PER_DAY);
  return { startSecond, endSecondExclusive };
}

function getDayBounds(plan: WebGpuTimePlan, timestampMs: number): { dayStartMs: number; dayEndMs: number } {
  const relativeMs = timestampMs - plan.startDayTimestampMs;
  const dayIndex = Math.floor(relativeMs / MS_PER_DAY);
  const dayStartMs = plan.startDayTimestampMs + dayIndex * MS_PER_DAY;
  return {
    dayStartMs,
    dayEndMs: dayStartMs + MS_PER_DAY,
  };
}
