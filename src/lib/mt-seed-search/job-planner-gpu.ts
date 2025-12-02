/**
 * GPUジョブ分割ロジック
 * GPU計算用にデバイス制約内で最大範囲のジョブを計画する
 */

import type {
  GpuJobPlannerConfig,
  JobPlan,
  MtSeedSearchJob,
} from '@/types/mt-seed-search';

/**
 * GPU処理時間の概算見積もり（ミリ秒）
 *
 * @param jobCount - ジョブ数
 * @param ivCodeCount - IVコード数
 * @returns 推定処理時間（ミリ秒）
 */
function estimateGpuTime(jobCount: number, ivCodeCount: number): number {
  // 概算: 1ジョブあたり約100ms、IVコード数による線形オーバーヘッド
  const baseTimePerJob = 100;
  const ivCodeOverhead = ivCodeCount * 0.01;

  return Math.ceil(jobCount * (baseTimePerJob + ivCodeOverhead));
}

/**
 * GPU計算用のジョブ分割を計画
 * デバイス制約の範囲内で可能な限り広い検索範囲を単一ジョブに割り当てる
 *
 * @param config - GPU向けジョブ計画設定
 * @returns ジョブ計画
 */
export function planGpuJobs(config: GpuJobPlannerConfig): JobPlan {
  const { fullRange, ivCodes, mtAdvances, deviceLimits, workgroupSize } = config;

  // 1ディスパッチあたりの最大処理数
  const maxSeedsPerDispatch =
    deviceLimits.maxComputeWorkgroupsPerDimension * workgroupSize;

  // 検索範囲サイズを計算
  const totalRange = fullRange.end - fullRange.start + 1;

  const jobs: MtSeedSearchJob[] = [];
  let cursor = fullRange.start;
  let jobId = 0;

  while (cursor <= fullRange.end) {
    // 今回のジョブで処理する範囲サイズ
    const remainingRange = fullRange.end - cursor + 1;
    const rangeSize = Math.min(maxSeedsPerDispatch, remainingRange);

    jobs.push({
      searchRange: {
        start: cursor,
        end: cursor + rangeSize - 1,
      },
      ivCodes,
      mtAdvances,
      jobId: jobId++,
    });

    // 次の開始位置（オーバーフロー対策）
    const nextCursor = cursor + rangeSize;
    if (nextCursor <= cursor) {
      // オーバーフローが発生した場合は終了
      break;
    }
    cursor = nextCursor;
  }

  return {
    jobs,
    totalSearchSpace: totalRange,
    estimatedTimeMs: estimateGpuTime(jobs.length, ivCodes.length),
  };
}

/**
 * WebGPUデバイスの制限値を取得
 * GPUAdapterから取得、または安全なデフォルト値を返す
 */
export function getDeviceLimits(device?: GPUDevice): GpuJobPlannerConfig['deviceLimits'] {
  if (device) {
    return {
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
      maxStorageBufferBindingSize:
        device.limits.maxStorageBufferBindingSize,
    };
  }

  // 安全なデフォルト値（WebGPU仕様の最小保証値）
  return {
    maxComputeWorkgroupsPerDimension: 65535,
    maxStorageBufferBindingSize: 128 * 1024 * 1024, // 128MB
  };
}

/**
 * 推奨ワークグループサイズ
 * device-context のデフォルト値に合わせて256を推奨
 */
export const RECOMMENDED_WORKGROUP_SIZE = 256;

/**
 * 32bit全探索用のGPUジョブ計画設定を作成
 */
export function createFullSearchGpuConfig(
  ivCodes: number[],
  mtAdvances: number,
  device?: GPUDevice
): GpuJobPlannerConfig {
  return {
    fullRange: { start: 0, end: 0xffffffff },
    ivCodes,
    mtAdvances,
    deviceLimits: getDeviceLimits(device),
    workgroupSize: RECOMMENDED_WORKGROUP_SIZE,
  };
}
