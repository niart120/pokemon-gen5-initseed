/**
 * Egg boot timing search chunk calculator
 * Worker に割り当てるチャンク分割を計算
 */

import type { EggBootTimingSearchParams } from '@/types/egg-boot-timing-search';
import { countValidKeyCombinations } from '@/lib/utils/key-input';

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
  maxWorkers: number = typeof navigator !== 'undefined'
    ? navigator.hardwareConcurrency || 4
    : 4
): EggBootTimingWorkerChunk[] {
  const operationsPerSecond = getOperationsPerSecond(params);

  const startDatetime = new Date(params.startDatetimeIso);
  const totalSeconds = params.rangeSeconds;
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

/**
 * バッチサイズ計算
 * メモリと応答性のバランスを考慮してバッチサイズを決定
 */
export function calculateBatchSize(params: EggBootTimingSearchParams): number {
  // 基本バッチサイズ: 60秒分
  const BASE_BATCH_SECONDS = 60;

  // Timer0/VCount/KeyCodeの組み合わせ数
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  const keyCombinations = countValidKeyCombinations(params.keyInputMask);

  const combinationsPerSecond = timer0Count * vcountCount * keyCombinations;

  // 組み合わせ数が多い場合はバッチサイズを小さくして応答性を確保
  if (combinationsPerSecond > 1000) {
    return Math.max(
      10,
      Math.floor(BASE_BATCH_SECONDS / (combinationsPerSecond / 100))
    );
  }

  return BASE_BATCH_SECONDS;
}
