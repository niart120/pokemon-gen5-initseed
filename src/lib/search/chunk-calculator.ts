/**
 * チャンク分割計算ユーティリティ
 * 検索範囲を複数Workerに効率的に分散
 */

import type { TimeChunk } from '../../types/parallel';

/**
 * デフォルトのWorker数を取得（環境に応じて調整）
 */
export function getDefaultWorkerCount(): number {
  return typeof navigator !== 'undefined'
    ? navigator.hardwareConcurrency || 4
    : 4;
}

/**
 * operationsPerSecond 計算用の入力パラメータ
 */
export interface OperationsPerSecondInput {
  timer0Count: number;
  vcountCount: number;
  keyCombinationCount: number;
  /** オプション: Egg検索用のadvanceCount */
  advanceCount?: number;
}

/**
 * 秒あたりの処理数を計算
 */
export function calculateOperationsPerSecond(input: OperationsPerSecondInput): number {
  const { timer0Count, vcountCount, keyCombinationCount, advanceCount = 1 } = input;
  return Math.max(1, timer0Count * vcountCount * keyCombinationCount * advanceCount);
}

/**
 * 時間チャンク計算用の入力パラメータ
 */
export interface TimeChunkCalculationInput {
  startDateTime: Date;
  endDateTime: Date;
  operationsPerSecond: number;
}

/**
 * 統一されたチャンク分割計算
 * 時間範囲を指定されたWorker数で均等に分割
 */
export function calculateTimeChunks(
  input: TimeChunkCalculationInput,
  maxWorkers: number = getDefaultWorkerCount()
): TimeChunk[] {
  const { startDateTime, endDateTime, operationsPerSecond } = input;

  const totalMs = endDateTime.getTime() - startDateTime.getTime();
  const totalSeconds = Math.max(1, Math.floor(totalMs / 1000) + 1);
  const secondsPerWorker = Math.ceil(totalSeconds / maxWorkers);

  const chunks: TimeChunk[] = [];
  const baseStartMs = startDateTime.getTime();

  for (let i = 0; i < maxWorkers; i++) {
    const chunkStartOffset = i * secondsPerWorker;
    if (chunkStartOffset >= totalSeconds) break;

    const chunkEndOffset = Math.min(chunkStartOffset + secondsPerWorker, totalSeconds);
    const rangeSeconds = chunkEndOffset - chunkStartOffset;

    const chunkStartDateTime = new Date(baseStartMs + chunkStartOffset * 1000);
    const chunkEndDateTime = new Date(baseStartMs + chunkEndOffset * 1000);

    chunks.push({
      workerId: i,
      startDateTime: chunkStartDateTime,
      endDateTime: chunkEndDateTime,
      rangeSeconds,
      estimatedOperations: rangeSeconds * operationsPerSecond,
    });
  }

  return chunks;
}
