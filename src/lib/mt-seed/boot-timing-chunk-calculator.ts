/**
 * MT Seed boot timing search chunk calculator
 * Worker に割り当てるチャンク分割を計算
 * 
 * 統一されたチャンク計算APIを使用
 */

import type { MtSeedBootTimingSearchParams } from '@/types/mt-seed-boot-timing-search';
import type { TimeChunk } from '@/types/parallel';
import { countValidKeyCombinations } from '@/lib/utils/key-input';
import {
  calculateTimeChunks,
  calculateOperationsPerSecond,
  getDefaultWorkerCount,
} from '@/lib/search/chunk-calculator';

/**
 * @deprecated TimeChunk を使用してください
 */
export interface MtSeedBootTimingWorkerChunk {
  workerId: number;
  startDatetime: Date;
  endDatetime: Date;
  rangeSeconds: number;
  estimatedOperations: number;
}

/**
 * MtSeedBootTimingSearchParams から operationsPerSecond を計算
 */
export function calculateMtSeedOperationsPerSecond(params: MtSeedBootTimingSearchParams): number {
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  const keyCombinationCount = countValidKeyCombinations(params.keyInputMask);

  return calculateOperationsPerSecond({
    timer0Count,
    vcountCount,
    keyCombinationCount,
  });
}

/**
 * MtSeedBootTimingSearchParams から TimeChunk[] を計算
 */
export function calculateMtSeedBootTimingTimeChunks(
  params: MtSeedBootTimingSearchParams,
  maxWorkers: number = getDefaultWorkerCount()
): TimeChunk[] {
  const { dateRange } = params;
  const startDateTime = new Date(
    dateRange.startYear,
    dateRange.startMonth - 1,
    dateRange.startDay
  );
  const endDateTime = new Date(
    dateRange.endYear,
    dateRange.endMonth - 1,
    dateRange.endDay,
    23, 59, 59
  );

  const operationsPerSecond = calculateMtSeedOperationsPerSecond(params);

  return calculateTimeChunks(
    { startDateTime, endDateTime, operationsPerSecond },
    maxWorkers
  );
}

// ============================================
// 以下は後方互換性のための旧API（deprecated）
// ============================================

/**
 * @deprecated calculateMtSeedBootTimingTimeChunks を使用してください
 */
export function calculateMtSeedBootTimingChunks(
  params: MtSeedBootTimingSearchParams,
  maxWorkers: number = getDefaultWorkerCount()
): MtSeedBootTimingWorkerChunk[] {
  const timeChunks = calculateMtSeedBootTimingTimeChunks(params, maxWorkers);

  // TimeChunk → MtSeedBootTimingWorkerChunk への変換
  return timeChunks.map((chunk) => ({
    workerId: chunk.workerId,
    startDatetime: chunk.startDateTime,
    endDatetime: chunk.endDateTime,
    rangeSeconds: chunk.rangeSeconds,
    estimatedOperations: chunk.estimatedOperations,
  }));
}

// Re-export for convenience
export { getDefaultWorkerCount } from '@/lib/search/chunk-calculator';
