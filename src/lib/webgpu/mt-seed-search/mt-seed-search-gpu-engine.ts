/**
 * MT Seed 32bit全探索 GPU エンジン
 *
 * WebGPUを使用してMT Seedの全探索を実行する。
 * MT19937のstate配列（624×32bit = 2.5KB）はGPUのprivateメモリに配置。
 *
 * device-context から制限値を取得し、最適なディスパッチサイズを動的に決定する。
 */

import type {
  MtSeedSearchJob,
  MtSeedMatch,
  IvCode,
} from '@/types/mt-seed-search';
import { decodeIvCode } from '@/types/mt-seed-search';
import {
  createWebGpuDeviceContext,
  isWebGpuSupported,
  BUFFER_ALIGNMENT,
  MT_SEED_SEARCH_PARAMS_WORDS,
  MT_SEED_RESULT_HEADER_WORDS,
  MT_SEED_RESULT_RECORD_WORDS,
  MT_SEED_MAX_RESULTS_PER_DISPATCH,
  type WebGpuDeviceContext,
  type SeedSearchJobLimits,
} from '@/lib/webgpu/utils';
import shaderSource from '@/lib/webgpu/kernel/mt-seed-search.wgsl?raw';

// === 型定義 ===

export interface MtSeedSearchGpuEngineConfig {
  /** 希望するワークグループサイズ（デフォルト: 256、device-contextで調整） */
  workgroupSize?: number;
  /** ディスパッチあたりの最大結果数 */
  maxResultsPerDispatch?: number;
}

export interface MtSeedSearchGpuEngineResult {
  matches: MtSeedMatch[];
  matchCount: number;
  processedCount: number;
}

export interface MtSeedSearchGpuEngine {
  /**
   * エンジンを初期化
   */
  initialize(): Promise<void>;

  /**
   * ジョブを実行
   */
  executeJob(job: MtSeedSearchJob): Promise<MtSeedSearchGpuEngineResult>;

  /**
   * リソースを解放
   */
  dispose(): void;

  /**
   * WebGPUが利用可能かどうか
   */
  isAvailable(): boolean;

  /**
   * 実際のワークグループサイズを取得
   */
  getWorkgroupSize(): number;

  /**
   * ジョブ制限値を取得
   */
  getJobLimits(): SeedSearchJobLimits | null;
}

// === ユーティリティ ===

function alignSize(bytes: number): number {
  return Math.ceil(bytes / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
}

// === エンジン実装 ===

export function createMtSeedSearchGpuEngine(
  config?: MtSeedSearchGpuEngineConfig
): MtSeedSearchGpuEngine {
  const requestedWorkgroupSize = config?.workgroupSize ?? 256;
  const maxResults = config?.maxResultsPerDispatch ?? MT_SEED_MAX_RESULTS_PER_DISPATCH;

  let deviceContext: WebGpuDeviceContext | null = null;
  let jobLimits: SeedSearchJobLimits | null = null;
  let pipeline: GPUComputePipeline | null = null;
  let bindGroupLayout: GPUBindGroupLayout | null = null;

  // 再利用可能なバッファ
  let paramsBuffer: GPUBuffer | null = null;
  let targetCodesBuffer: GPUBuffer | null = null;
  let resultsBuffer: GPUBuffer | null = null;
  let readbackBuffer: GPUBuffer | null = null;

  let targetCodesCapacity = 0;
  let actualWorkgroupSize = requestedWorkgroupSize;

  const isAvailable = (): boolean => {
    return isWebGpuSupported();
  };

  const initialize = async (): Promise<void> => {
    if (!isAvailable()) {
      throw new Error('WebGPU is not available in this environment');
    }

    // WebGpuDeviceContext を使用してデバイスを取得
    deviceContext = await createWebGpuDeviceContext({
      powerPreference: 'high-performance',
      label: 'mt-seed-search-device',
    });

    // デバイスの制限値から検索ジョブの制限を導出
    jobLimits = deviceContext.deriveSearchJobLimits({
      workgroupSize: requestedWorkgroupSize,
      candidateCapacityPerDispatch: maxResults,
    });

    // 実際に使用するワークグループサイズを取得
    actualWorkgroupSize = jobLimits.workgroupSize;

    const device = deviceContext.getDevice();

    // シェーダーのワークグループサイズを置換
    const processedShader = shaderSource
      .replace(/WORKGROUP_SIZE_PLACEHOLDERu/g, `${actualWorkgroupSize}u`)
      .replace(/WORKGROUP_SIZE_PLACEHOLDER/g, `${actualWorkgroupSize}`);

    const shaderModule = device.createShaderModule({
      label: 'mt-seed-search-shader',
      code: processedShader,
    });

    // バインドグループレイアウトを作成
    bindGroupLayout = device.createBindGroupLayout({
      label: 'mt-seed-search-bind-group-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: 'mt-seed-search-pipeline-layout',
      bindGroupLayouts: [bindGroupLayout],
    });

    pipeline = device.createComputePipeline({
      label: 'mt-seed-search-pipeline',
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    // パラメータバッファを作成
    const paramsSize = alignSize(MT_SEED_SEARCH_PARAMS_WORDS * Uint32Array.BYTES_PER_ELEMENT);
    paramsBuffer = device.createBuffer({
      label: 'mt-seed-search-params',
      size: paramsSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 結果バッファを作成
    const resultsWords = MT_SEED_RESULT_HEADER_WORDS + maxResults * MT_SEED_RESULT_RECORD_WORDS;
    const resultsSize = alignSize(resultsWords * Uint32Array.BYTES_PER_ELEMENT);
    resultsBuffer = device.createBuffer({
      label: 'mt-seed-search-results',
      size: resultsSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    readbackBuffer = device.createBuffer({
      label: 'mt-seed-search-readback',
      size: resultsSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  };

  const ensureTargetCodesBuffer = (ivCodes: IvCode[]): void => {
    if (!deviceContext) {
      throw new Error('Engine not initialized');
    }

    const device = deviceContext.getDevice();
    const requiredCapacity = ivCodes.length;
    if (targetCodesBuffer && targetCodesCapacity >= requiredCapacity) {
      return;
    }

    targetCodesBuffer?.destroy();

    const bufferSize = alignSize(requiredCapacity * Uint32Array.BYTES_PER_ELEMENT);
    targetCodesBuffer = device.createBuffer({
      label: 'mt-seed-search-target-codes',
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    targetCodesCapacity = requiredCapacity;
  };

  const executeJob = async (job: MtSeedSearchJob): Promise<MtSeedSearchGpuEngineResult> => {
    if (!deviceContext || !pipeline || !bindGroupLayout || !paramsBuffer || !resultsBuffer || !readbackBuffer || !jobLimits) {
      throw new Error('Engine not initialized');
    }

    const device = deviceContext.getDevice();
    const { searchRange, ivCodes, mtAdvances } = job;
    const { start, end } = searchRange;

    // 検索範囲のサイズを計算
    const searchCount = end - start + 1;

    // ターゲットIVコードバッファを準備
    ensureTargetCodesBuffer(ivCodes);
    const ivCodesArray = new Uint32Array(ivCodes);
    device.queue.writeBuffer(
      targetCodesBuffer!,
      0,
      ivCodesArray.buffer,
      ivCodesArray.byteOffset,
      ivCodesArray.byteLength
    );

    // 結果集約用
    const allMatches: MtSeedMatch[] = [];
    let totalMatchCount = 0;

    // device-context から導出した制限値を使用してディスパッチサイズを計算
    // maxMessagesPerDispatch = workgroupSize * maxWorkgroupsPerDispatch
    const maxSeedsPerDispatch = jobLimits.maxMessagesPerDispatch;
    let dispatchStart = start;
    let dispatchIndex = 0;

    while (dispatchStart <= end) {
      // 今回のディスパッチで処理する範囲
      const remainingSeeds = end - dispatchStart + 1;
      const dispatchSeedCount = Math.min(maxSeedsPerDispatch, remainingSeeds);
      const dispatchEnd = dispatchStart + dispatchSeedCount - 1;
      const workgroupCount = Math.ceil(dispatchSeedCount / actualWorkgroupSize);

      // パラメータを書き込み
      const paramsData = new Uint32Array([
        dispatchStart,      // start_seed
        dispatchEnd,        // end_seed
        mtAdvances,         // advances
        ivCodes.length,     // target_count
        maxResults,         // max_results
        0,                  // reserved0
        0,                  // reserved1
        0,                  // reserved2
      ]);
      device.queue.writeBuffer(
        paramsBuffer,
        0,
        paramsData.buffer,
        paramsData.byteOffset,
        paramsData.byteLength
      );

      // 結果バッファをクリア（match_countを0に）
      const zeroHeader = new Uint32Array([0]);
      device.queue.writeBuffer(
        resultsBuffer,
        0,
        zeroHeader.buffer,
        zeroHeader.byteOffset,
        zeroHeader.byteLength
      );

      // バインドグループを作成
      const bindGroup = device.createBindGroup({
        label: `mt-seed-search-bind-group-${job.jobId}-${dispatchIndex}`,
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: targetCodesBuffer! } },
          { binding: 2, resource: { buffer: resultsBuffer } },
        ],
      });

      // コマンドエンコーダを作成
      const encoder = device.createCommandEncoder({
        label: `mt-seed-search-encoder-${job.jobId}`,
      });

      const pass = encoder.beginComputePass({
        label: `mt-seed-search-pass-${job.jobId}`,
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();

      // 結果をreadbackバッファにコピー
      const resultsWords = MT_SEED_RESULT_HEADER_WORDS + maxResults * MT_SEED_RESULT_RECORD_WORDS;
      const resultsSize = alignSize(resultsWords * Uint32Array.BYTES_PER_ELEMENT);
      encoder.copyBufferToBuffer(resultsBuffer, 0, readbackBuffer, 0, resultsSize);

      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);

      // 結果を読み取り
      await readbackBuffer.mapAsync(GPUMapMode.READ, 0, resultsSize);
      const mapped = readbackBuffer.getMappedRange(0, resultsSize);
      const resultData = new Uint32Array(mapped.slice(0));
      readbackBuffer.unmap();

      // 結果をパース
      const matchCount = Math.min(resultData[0] ?? 0, maxResults);
      totalMatchCount += matchCount;

      for (let i = 0; i < matchCount; i++) {
        const baseIdx = MT_SEED_RESULT_HEADER_WORDS + i * MT_SEED_RESULT_RECORD_WORDS;
        const mtSeed = resultData[baseIdx];
        const ivCode: IvCode = resultData[baseIdx + 1];
        const ivSet = decodeIvCode(ivCode);
        allMatches.push({ mtSeed, ivCode, ivSet });
      }

      // 次のディスパッチへ（オーバーフロー対策）
      if (dispatchEnd >= end) {
        break;
      }
      dispatchStart = dispatchEnd + 1;
      dispatchIndex++;
    }

    return {
      matches: allMatches,
      matchCount: totalMatchCount,
      processedCount: searchCount,
    };
  };

  const dispose = (): void => {
    paramsBuffer?.destroy();
    targetCodesBuffer?.destroy();
    resultsBuffer?.destroy();
    readbackBuffer?.destroy();

    paramsBuffer = null;
    targetCodesBuffer = null;
    resultsBuffer = null;
    readbackBuffer = null;
    targetCodesCapacity = 0;

    deviceContext = null;
    pipeline = null;
    bindGroupLayout = null;
    jobLimits = null;
  };

  const getWorkgroupSize = (): number => actualWorkgroupSize;

  const getJobLimits = (): SeedSearchJobLimits | null => jobLimits;

  return {
    initialize,
    executeJob,
    dispose,
    isAvailable,
    getWorkgroupSize,
    getJobLimits,
  };
}

/**
 * WebGPUが利用可能かどうかを確認
 */
export function isMtSeedSearchGpuAvailable(): boolean {
  return isWebGpuSupported();
}
