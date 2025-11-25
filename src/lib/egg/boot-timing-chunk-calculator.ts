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

/**
 * バッチサイズ計算
 *
 * 想定される検索シナリオ:
 * - 片親6V(他方不明)で色違い(1/8192)×希望性格(1/25)×夢特性(2/5)を検索
 * - 検索範囲: 1年(31,536,000秒) × 100消費程度
 * - Timer0/VCount/KeyCodeの組み合わせ数: 通常数個〜数十個程度
 *
 * 設計方針:
 * - 進捗更新のオーバーヘッドを抑えるため、十分な時間範囲をまとめて処理
 * - 最小バッチサイズは3600秒(1時間)を確保し、大規模検索での効率を維持
 * - UIの応答性より検索効率を優先（検索自体が長時間かかる想定）
 */
export function calculateBatchSize(params: EggBootTimingSearchParams): number {
  // 想定される大規模検索シナリオ（1年検索）に対応するバッチサイズ
  // 1年 = 31,536,000秒を適切な数のバッチに分割
  // 目標: 100〜1000程度のバッチ数で全体を処理
  const TARGET_BATCH_COUNT = 500;

  // 最小バッチサイズ: 1時間(3600秒) - 大規模検索での効率を確保
  const MIN_BATCH_SECONDS = 3600;

  // 最大バッチサイズ: 1日(86400秒) - メモリ使用量の上限
  const MAX_BATCH_SECONDS = 86400;

  // dateRangeから日数を計算
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
  const totalRangeSeconds = daysDiff * 24 * 60 * 60;

  // 検索範囲から目標バッチ数に基づくバッチサイズを計算
  const calculatedBatchSize = Math.ceil(totalRangeSeconds / TARGET_BATCH_COUNT);

  // 最小・最大の範囲内に収める
  return Math.max(MIN_BATCH_SECONDS, Math.min(MAX_BATCH_SECONDS, calculatedBatchSize));
}
