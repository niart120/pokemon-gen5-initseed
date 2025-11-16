/**
 * チャンク分割計算ユーティリティ
 * 検索範囲を複数Workerに効率的に分散
 */

import type { SearchConditions } from '../../types/search';
import type { WorkerChunk } from '../../types/parallel';
import { countValidKeyCombinations } from '@/lib/utils/key-input';

const getOperationsPerSecond = (conditions: SearchConditions): number => {
  const timer0Count =
    conditions.timer0VCountConfig.timer0Range.max - conditions.timer0VCountConfig.timer0Range.min + 1;
  const vcountCount =
    conditions.timer0VCountConfig.vcountRange.max - conditions.timer0VCountConfig.vcountRange.min + 1;
  const keyCombinationCount = countValidKeyCombinations(conditions.keyInput);
  return Math.max(1, timer0Count * vcountCount * keyCombinationCount);
};

const estimateOperations = (
  startDate: Date,
  endDate: Date,
  operationsPerSecond: number
): number => {
  const seconds = Math.floor((endDate.getTime() - startDate.getTime()) / 1000) + 1;
  return Math.max(1, seconds * operationsPerSecond);
};

export function calculateOptimalChunks(
  conditions: SearchConditions,
  maxWorkers: number = navigator.hardwareConcurrency || 4
): WorkerChunk[] {
  const operationsPerSecond = getOperationsPerSecond(conditions);
  const startDate = new Date(
    conditions.dateRange.startYear,
    conditions.dateRange.startMonth - 1,
    conditions.dateRange.startDay,
    conditions.dateRange.startHour,
    conditions.dateRange.startMinute,
    conditions.dateRange.startSecond
  );

  const endDate = new Date(
    conditions.dateRange.endYear,
    conditions.dateRange.endMonth - 1,
    conditions.dateRange.endDay,
    conditions.dateRange.endHour,
    conditions.dateRange.endMinute,
    conditions.dateRange.endSecond
  );

  const totalSeconds = Math.floor((endDate.getTime() - startDate.getTime()) / 1000) + 1;
  const secondsPerWorker = Math.ceil(totalSeconds / maxWorkers);
  const baseStartMs = startDate.getTime();
  const endMs = endDate.getTime();

  const chunks: WorkerChunk[] = [];

  for (let i = 0; i < maxWorkers; i++) {
    const chunkStartMs = baseStartMs + i * secondsPerWorker * 1000;
    if (chunkStartMs > endMs) {
      break;
    }

    const chunkEndMs = Math.min(chunkStartMs + secondsPerWorker * 1000 - 1000, endMs);
    const chunkStartDate = new Date(chunkStartMs);
    const chunkEndDate = new Date(chunkEndMs);

    const estimatedOps = estimateOperations(
      chunkStartDate,
      chunkEndDate,
      operationsPerSecond
    );

    chunks.push({
      workerId: i,
      startDateTime: chunkStartDate,
      endDateTime: chunkEndDate,
      timer0Range: conditions.timer0VCountConfig.timer0Range,
      vcountRange: conditions.timer0VCountConfig.vcountRange,
      estimatedOperations: estimatedOps
    });
  }

  return chunks;
}
