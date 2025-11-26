/**
 * Egg boot timing search chunk calculator
 * Worker に割り当てるチャンク分割を計算
 */

import type { EggBootTimingSearchParams } from '@/types/egg-boot-timing-search';
import { countValidKeyCombinations } from '@/lib/utils/key-input';

/**
 * デフォルトのWorker数を取得（環境に応じて調整）
 */
export function getDefaultWorkerCount(): number {
  return typeof navigator !== 'undefined'
    ? navigator.hardwareConcurrency || 4
    : 4;
}

/**
 * Worker に割り当てるチャンク情報
 */
export interface EggBootTimingWorkerChunk {
  workerId: number;
  startDatetime: Date;
  endDatetime: Date;
  rangeSeconds: number;
  estimatedOperations: number;
}

/**
 * 秒あたりの処理数を計算
 */
function getOperationsPerSecond(params: EggBootTimingSearchParams): number {
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  const keyCombinationCount = countValidKeyCombinations(params.keyInputMask);

  // 各秒に対して advanceCount 分の個体を検索
  return Math.max(
    1,
    timer0Count * vcountCount * keyCombinationCount * params.advanceCount
  );
}

/**
 * 最適なチャンク分割を計算
 */
export function calculateEggBootTimingChunks(
  params: EggBootTimingSearchParams,
  maxWorkers: number = getDefaultWorkerCount()
): EggBootTimingWorkerChunk[] {
  const operationsPerSecond = getOperationsPerSecond(params);

  // dateRangeから開始日と日数を計算
  const { dateRange } = params;
  const startDatetime = new Date(
    dateRange.startYear,
    dateRange.startMonth - 1,
    dateRange.startDay
  );
  const endDatetime = new Date(
    dateRange.endYear,
    dateRange.endMonth - 1,
    dateRange.endDay
  );
  // 日数計算: Math.floorを使用して日数を正確に計算（両端含む）
  const daysDiff = Math.max(1, Math.floor((endDatetime.getTime() - startDatetime.getTime()) / (1000 * 60 * 60 * 24)) + 1);
  const totalSeconds = daysDiff * 24 * 60 * 60;
  const secondsPerWorker = Math.ceil(totalSeconds / maxWorkers);

  const chunks: EggBootTimingWorkerChunk[] = [];

  for (let i = 0; i < maxWorkers; i++) {
    const chunkStartOffset = i * secondsPerWorker;
    if (chunkStartOffset >= totalSeconds) break;

    const chunkEndOffset = Math.min(
      chunkStartOffset + secondsPerWorker,
      totalSeconds
    );
    const chunkRangeSeconds = chunkEndOffset - chunkStartOffset;

    const chunkStartDatetime = new Date(
      startDatetime.getTime() + chunkStartOffset * 1000
    );
    const chunkEndDatetime = new Date(
      startDatetime.getTime() + chunkEndOffset * 1000
    );

    const estimatedOperations = chunkRangeSeconds * operationsPerSecond;

    chunks.push({
      workerId: i,
      startDatetime: chunkStartDatetime,
      endDatetime: chunkEndDatetime,
      rangeSeconds: chunkRangeSeconds,
      estimatedOperations,
    });
  }

  return chunks;
}
