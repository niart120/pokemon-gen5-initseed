/**
 * MT Seed 32bit全探索 GPU エンジン
 *
 * WebGPUを使用してMT Seedの全探索を実行する。
 * MT19937のstate配列（624×32bit = 2.5KB）はGPUのprivateメモリに配置。
 */

import type {
  MtSeedSearchJob,
  MtSeedMatch,
  IvCode,
} from '@/types/mt-seed-search';
import { decodeIvCode } from '@/types/mt-seed-search';
import shaderSource from '@/workers/shaders/mt-seed-search.wgsl?raw';

// === 定数 ===
const BUFFER_ALIGNMENT = 256;
const DEFAULT_WORKGROUP_SIZE = 64;
const MAX_RESULTS_PER_DISPATCH = 4096;

// デバイス制限のフォールバック値（WebGPU仕様の最小保証値）
const FALLBACK_MAX_WORKGROUPS_PER_DIMENSION = 65535;

// SearchParams構造体のサイズ（8 * u32 = 32バイト）
const SEARCH_PARAMS_WORDS = 8;

// 結果バッファのヘッダサイズ（match_count: 1 word）
const RESULT_HEADER_WORDS = 1;

// 1レコードのサイズ（seed + iv_code = 2 words）
const RESULT_RECORD_WORDS = 2;

// === 型定義 ===

export interface MtSeedSearchGpuEngineConfig {
  workgroupSize?: number;
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
   * ワークグループサイズを取得
   */
  getWorkgroupSize(): number;
}

// === ユーティリティ ===

function alignSize(bytes: number): number {
  return Math.ceil(bytes / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
}

// === エンジン実装 ===

export function createMtSeedSearchGpuEngine(
  config?: MtSeedSearchGpuEngineConfig
): MtSeedSearchGpuEngine {
  const workgroupSize = config?.workgroupSize ?? DEFAULT_WORKGROUP_SIZE;
  const maxResults = config?.maxResultsPerDispatch ?? MAX_RESULTS_PER_DISPATCH;

  let device: GPUDevice | null = null;
  let pipeline: GPUComputePipeline | null = null;
  let bindGroupLayout: GPUBindGroupLayout | null = null;
  let maxWorkgroupsPerDimension = FALLBACK_MAX_WORKGROUPS_PER_DIMENSION;

  // 再利用可能なバッファ
  let paramsBuffer: GPUBuffer | null = null;
  let targetCodesBuffer: GPUBuffer | null = null;
  let resultsBuffer: GPUBuffer | null = null;
  let readbackBuffer: GPUBuffer | null = null;

  let targetCodesCapacity = 0;

  const isAvailable = (): boolean => {
    return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined';
  };

  const initialize = async (): Promise<void> => {
    if (!isAvailable()) {
      throw new Error('WebGPU is not available in this environment');
    }

    const gpu = navigator.gpu!;
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('Failed to acquire WebGPU adapter');
    }

    device = await adapter.requestDevice({
      label: 'mt-seed-search-device',
    });

    // デバイスの制限値を取得
    maxWorkgroupsPerDimension = device.limits.maxComputeWorkgroupsPerDimension
      ?? FALLBACK_MAX_WORKGROUPS_PER_DIMENSION;

    // シェーダーのワークグループサイズを置換
    const processedShader = shaderSource
      .replace(/WORKGROUP_SIZE_PLACEHOLDERu/g, `${workgroupSize}u`)
      .replace(/WORKGROUP_SIZE_PLACEHOLDER/g, `${workgroupSize}`);

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
    const paramsSize = alignSize(SEARCH_PARAMS_WORDS * Uint32Array.BYTES_PER_ELEMENT);
    paramsBuffer = device.createBuffer({
      label: 'mt-seed-search-params',
      size: paramsSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 結果バッファを作成
    const resultsWords = RESULT_HEADER_WORDS + maxResults * RESULT_RECORD_WORDS;
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
    if (!device) {
      throw new Error('Engine not initialized');
    }

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
    if (!device || !pipeline || !bindGroupLayout || !paramsBuffer || !resultsBuffer || !readbackBuffer) {
      throw new Error('Engine not initialized');
    }

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

    // ディスパッチを分割して実行
    // 1ディスパッチあたりの最大Seed数 = maxWorkgroupsPerDimension * workgroupSize
    const maxSeedsPerDispatch = maxWorkgroupsPerDimension * workgroupSize;
    let dispatchStart = start;
    let dispatchIndex = 0;

    while (dispatchStart <= end) {
      // 今回のディスパッチで処理する範囲
      const remainingSeeds = end - dispatchStart + 1;
      const dispatchSeedCount = Math.min(maxSeedsPerDispatch, remainingSeeds);
      const dispatchEnd = dispatchStart + dispatchSeedCount - 1;
      const workgroupCount = Math.ceil(dispatchSeedCount / workgroupSize);

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
      const resultsWords = RESULT_HEADER_WORDS + maxResults * RESULT_RECORD_WORDS;
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
        const baseIdx = RESULT_HEADER_WORDS + i * RESULT_RECORD_WORDS;
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

    device = null;
    pipeline = null;
    bindGroupLayout = null;
  };

  const getWorkgroupSize = (): number => workgroupSize;

  return {
    initialize,
    executeJob,
    dispose,
    isAvailable,
    getWorkgroupSize,
  };
}

/**
 * WebGPUが利用可能かどうかを確認
 */
export function isMtSeedSearchGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined';
}
