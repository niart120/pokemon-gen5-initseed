/**
 * Egg boot timing search chunk calculator
 * Worker に割り当てるチャンク分割を計算
 * 
 * 統一されたチャンク計算APIを使用
 */

import type { EggBootTimingSearchParams } from '@/types/egg-boot-timing-search';
import type { TimeChunk } from '@/types/parallel';
import { countValidKeyCombinations } from '@/lib/utils/key-input';
import {
  calculateTimeChunks,
  calculateOperationsPerSecond,
  getDefaultWorkerCount,
} from '@/lib/search/chunk-calculator';

/**
 * EggBootTimingSearchParams から operationsPerSecond を計算
 */
export function calculateEggOperationsPerSecond(params: EggBootTimingSearchParams): number {
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  const keyCombinationCount = countValidKeyCombinations(params.keyInputMask);

  return calculateOperationsPerSecond({
    timer0Count,
    vcountCount,
    keyCombinationCount,
    advanceCount: params.advanceCount,
  });
}

/**
 * EggBootTimingSearchParams から TimeChunk[] を計算
 */
export function calculateEggBootTimingTimeChunks(
  params: EggBootTimingSearchParams,
  maxWorkers: number = getDefaultWorkerCount()
): TimeChunk[] {
  const { dateRange, timeRange } = params;
  const startDateTime = new Date(
    dateRange.startYear,
    dateRange.startMonth - 1,
    dateRange.startDay,
    timeRange.hour.start,
    timeRange.minute.start,
    timeRange.second.start
  );
  const endDateTime = new Date(
    dateRange.endYear,
    dateRange.endMonth - 1,
    dateRange.endDay,
    timeRange.hour.end,
    timeRange.minute.end,
    timeRange.second.end
  );

  const operationsPerSecond = calculateEggOperationsPerSecond(params);

  return calculateTimeChunks(
    { startDateTime, endDateTime, operationsPerSecond },
    maxWorkers
  );
}

// Re-export for convenience
export { getDefaultWorkerCount } from '@/lib/search/chunk-calculator';
