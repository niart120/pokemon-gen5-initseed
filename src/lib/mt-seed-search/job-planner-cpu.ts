/**
 * CPUジョブ分割ロジック
 * CPU並列計算用にSeed空間をWorker数で均等分割する
 */

import type {
  CpuJobPlannerConfig,
  JobPlan,
  MtSeedSearchJob,
} from '@/types/mt-seed-search';

/**
 * CPU処理時間の概算見積もり（ミリ秒）
 * 
 * @param totalRange - 検索範囲のサイズ
 * @param ivCodeCount - IVコード数
 * @param workerCount - Worker数
 * @returns 推定処理時間（ミリ秒）
 */
function estimateCpuTime(
  totalRange: number,
  ivCodeCount: number,
  workerCount: number
): number {
  // 概算: 1 Seed あたり約 0.001ms（SIMD最適化込み）
  // IVコード数による線形オーバーヘッドは HashSet なので無視可能
  const baseTimePerSeed = 0.001;
  const parallelFactor = Math.max(1, workerCount);
  
  return Math.ceil((totalRange * baseTimePerSeed) / parallelFactor);
}

/**
 * CPU並列計算用のジョブ分割を計画
 * 
 * @param config - CPU向けジョブ計画設定
 * @returns ジョブ計画
 */
export function planCpuJobs(config: CpuJobPlannerConfig): JobPlan {
  const { fullRange, ivCodes, mtAdvances, workerCount } = config;

  // 検索範囲サイズを計算（オーバーフロー対策）
  const totalRange = fullRange.end - fullRange.start + 1;
  
  // Worker数が0の場合は1として扱う
  const effectiveWorkerCount = Math.max(1, workerCount);
  
  // 各Workerへの割り当て範囲サイズ
  const rangePerWorker = Math.ceil(totalRange / effectiveWorkerCount);

  const jobs: MtSeedSearchJob[] = [];
  let cursor = fullRange.start;

  for (let i = 0; i < effectiveWorkerCount && cursor <= fullRange.end; i++) {
    // 範囲終端を計算（オーバーフロー対策）
    const rangeEnd = Math.min(
      cursor + rangePerWorker - 1,
      fullRange.end
    );

    jobs.push({
      searchRange: { start: cursor, end: rangeEnd },
      ivCodes,
      mtAdvances,
      jobId: i,
    });

    // 次の開始位置（オーバーフロー対策）
    cursor = rangeEnd + 1;
    if (cursor < rangeEnd) {
      // オーバーフローが発生した場合は終了
      break;
    }
  }

  return {
    jobs,
    totalSearchSpace: totalRange,
    estimatedTimeMs: estimateCpuTime(totalRange, ivCodes.length, effectiveWorkerCount),
  };
}

/**
 * デフォルトのWorker数を取得
 * navigator.hardwareConcurrency が使用可能な場合はその値を返す
 */
export function getDefaultWorkerCount(): number {
  if (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) {
    return navigator.hardwareConcurrency;
  }
  // フォールバック: 4コア想定
  return 4;
}

/**
 * 32bit全探索用のCPUジョブ計画設定を作成
 */
export function createFullSearchCpuConfig(
  ivCodes: number[],
  mtAdvances: number,
  workerCount?: number
): CpuJobPlannerConfig {
  return {
    fullRange: { start: 0, end: 0xffffffff },
    ivCodes,
    mtAdvances,
    workerCount: workerCount ?? getDefaultWorkerCount(),
  };
}
