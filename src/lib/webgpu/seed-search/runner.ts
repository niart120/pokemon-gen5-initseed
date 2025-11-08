import { SeedCalculator } from '@/lib/core/seed-calculator';
import { SHA1 } from '@/lib/core/sha1';
import type { InitialSeedResult } from '@/types/search';
import {
  DEFAULT_WORKGROUP_SIZE,
  DOUBLE_BUFFER_SET_COUNT,
  MATCH_OUTPUT_HEADER_WORDS,
  MATCH_RECORD_WORDS,
} from './constants';
import {
  createWebGpuDeviceContext,
  isWebGpuSupported as isDeviceWebGpuSupported,
  type WebGpuDeviceContext,
} from './device-context';
import { createGeneratedPipeline } from './pipelines/pipeline-factory';
import { createWebGpuBufferPool, type WebGpuBufferPool } from './buffers/buffer-pool';
import { createWebGpuBatchPlanner, type WebGpuBatchPlanner } from './batch-planner';
import { createWebGpuProfilingCollector, type WebGpuProfilingCollector } from './profiling';
import type {
  WebGpuRunRequest,
  WebGpuRunnerCallbacks,
  WebGpuRunnerProgress,
  WebGpuSearchContext,
  WebGpuSegment,
} from './types';

interface DispatchContext {
  segment: WebGpuSegment;
  dispatchIndex: number;
  messageCount: number;
  slotIndex: number;
}

const CONFIG_WORD_COUNT = 20;
const PROGRESS_INTERVAL_MS = 500;
const YIELD_INTERVAL = 1024;
const ZERO_MATCH_HEADER = new Uint32Array([0]);

interface RunnerState {
  workgroupSize: number;
  deviceContext: WebGpuDeviceContext | null;
  pipeline: GPUComputePipeline | null;
  bindGroupLayout: GPUBindGroupLayout | null;
  configBuffer: GPUBuffer | null;
  configData: Uint32Array | null;
  bufferPool: WebGpuBufferPool | null;
  planner: WebGpuBatchPlanner | null;
  profilingCollector: WebGpuProfilingCollector | null;
  targetBuffer: GPUBuffer | null;
  targetBufferCapacity: number;
  readonly seedCalculator: SeedCalculator;
  isRunning: boolean;
  isPaused: boolean;
  shouldStop: boolean;
  lastProgressUpdateMs: number;
  timerState: {
    cumulativeRunTime: number;
    segmentStartTime: number;
    isPaused: boolean;
  };
}

export interface WebGpuSeedSearchRunner {
  init(): Promise<void>;
  run(request: WebGpuRunRequest): Promise<void>;
  pause(): void;
  resume(): void;
  stop(): void;
  dispose(): void;
}

export const isWebGpuSeedSearchSupported = isDeviceWebGpuSupported;

export function createWebGpuSeedSearchRunner(options?: { workgroupSize?: number }): WebGpuSeedSearchRunner {
  const state: RunnerState = {
    workgroupSize: options?.workgroupSize ?? DEFAULT_WORKGROUP_SIZE,
    deviceContext: null,
    pipeline: null,
    bindGroupLayout: null,
    configBuffer: null,
    configData: null,
    bufferPool: null,
    planner: null,
    profilingCollector: null,
  targetBuffer: null,
  targetBufferCapacity: 0,
    seedCalculator: new SeedCalculator(),
    isRunning: false,
    isPaused: false,
    shouldStop: false,
    lastProgressUpdateMs: 0,
    timerState: {
      cumulativeRunTime: 0,
      segmentStartTime: 0,
      isPaused: false,
    },
  };

  const init = async (): Promise<void> => {
    if (state.pipeline && state.bufferPool && state.planner && state.deviceContext) {
      return;
    }

    const context = await createWebGpuDeviceContext();
    const device = context.getDevice();
    const resolvedWorkgroupSize = context.getSupportedWorkgroupSize(state.workgroupSize);
    const { pipeline, bindGroupLayout } = createGeneratedPipeline(device, resolvedWorkgroupSize);

    const configData = new Uint32Array(CONFIG_WORD_COUNT);
    const configSize = alignSize(configData.byteLength);
    const configBuffer = device.createBuffer({
      label: 'gpu-seed-config-buffer',
      size: configSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferPool = createWebGpuBufferPool(device, {
      slots: DOUBLE_BUFFER_SET_COUNT,
    });

    const planner = createWebGpuBatchPlanner(context, {
      workgroupSize: resolvedWorkgroupSize,
      bufferSetCount: DOUBLE_BUFFER_SET_COUNT,
    });

    state.deviceContext = context;
    state.pipeline = pipeline;
    state.bindGroupLayout = bindGroupLayout;
    state.configBuffer = configBuffer;
    state.configData = configData;
    state.bufferPool = bufferPool;
    state.planner = planner;
    state.workgroupSize = resolvedWorkgroupSize;
  };

  const ensureTargetBuffer = (targetSeeds: readonly number[]): void => {
    if (!state.deviceContext) {
      throw new Error('WebGPU device is not initialized');
    }

    const device = state.deviceContext.getDevice();
    const seedCount = targetSeeds.length;
    const requiredWords = 1 + seedCount;
    const requiredBytes = alignSize(requiredWords * Uint32Array.BYTES_PER_ELEMENT);

    const currentCapacity = state.targetBufferCapacity;
    const needsRecreate = !state.targetBuffer || currentCapacity < seedCount;

    if (needsRecreate) {
      state.targetBuffer?.destroy();
      state.targetBuffer = device.createBuffer({
        label: 'gpu-seed-target-buffer',
        size: requiredBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const availableWords = Math.floor(requiredBytes / Uint32Array.BYTES_PER_ELEMENT) - 1;
      state.targetBufferCapacity = Math.max(availableWords, seedCount);
    }

    const uploadWords = new Uint32Array(requiredWords);
    uploadWords[0] = seedCount >>> 0;
    for (let index = 0; index < seedCount; index += 1) {
      uploadWords[1 + index] = targetSeeds[index]! >>> 0;
    }

    const bytesToWrite = uploadWords.byteLength;
    device.queue.writeBuffer(state.targetBuffer!, 0, uploadWords.buffer, uploadWords.byteOffset, bytesToWrite);
  };

  const run = async (request: WebGpuRunRequest): Promise<void> => {
    if (state.isRunning) {
      throw new Error('WebGPU search is already running');
    }

    if (!state.pipeline || !state.bufferPool || !state.configBuffer || !state.configData || !state.planner || !state.deviceContext) {
      await init();
    }

    if (!state.pipeline || !state.bufferPool || !state.configBuffer || !state.configData || !state.planner || !state.deviceContext) {
      throw new Error('WebGPU runner failed to initialize');
    }

    const { context, targetSeeds, callbacks, signal } = request;

    if (context.totalMessages === 0) {
      callbacks.onComplete('探索対象の組み合わせが存在しません');
      return;
    }

    if (!state.bindGroupLayout) {
      throw new Error('WebGPU runner missing bind group layout');
    }

  ensureTargetBuffer(targetSeeds);

    state.isRunning = true;
    state.isPaused = false;
    state.shouldStop = false;
    state.lastProgressUpdateMs = Date.now();
    state.profilingCollector = createWebGpuProfilingCollector();

    const progress: WebGpuRunnerProgress = {
      currentStep: 0,
      totalSteps: context.totalMessages,
      elapsedTime: 0,
      estimatedTimeRemaining: 0,
      matchesFound: 0,
    };

    let abortCleanup: (() => void) | undefined;
    if (signal) {
      if (signal.aborted) {
        state.shouldStop = true;
      } else {
        const onAbort = () => {
          state.shouldStop = true;
        };
        signal.addEventListener('abort', onAbort);
        abortCleanup = () => signal.removeEventListener('abort', onAbort);
      }
    }

    startTimer();

    try {
      await executeSegments(context, progress, callbacks);

      const elapsed = getElapsedTime();
      const finalProgress: WebGpuRunnerProgress = {
        ...progress,
        elapsedTime: elapsed,
        estimatedTimeRemaining: 0,
      };

      if (state.shouldStop) {
        callbacks.onStopped('検索を停止しました', finalProgress);
      } else {
        callbacks.onProgress(finalProgress);
        callbacks.onComplete(`検索が完了しました。${progress.matchesFound}件ヒットしました。`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'WebGPU検索中に不明なエラーが発生しました';
      const code = error instanceof GPUValidationError ? 'WEBGPU_VALIDATION_ERROR' : undefined;
      callbacks.onError(message, code);
      throw error;
    } finally {
      state.isRunning = false;
      state.isPaused = false;
      state.shouldStop = false;
      pauseTimer();
      state.profilingCollector = null;
      if (abortCleanup) {
        abortCleanup();
      }
    }
  };

  const pause = (): void => {
    if (!state.isRunning || state.isPaused) {
      return;
    }
    state.isPaused = true;
    pauseTimer();
  };

  const resume = (): void => {
    if (!state.isRunning || !state.isPaused) {
      return;
    }
    state.isPaused = false;
    resumeTimer();
  };

  const stop = (): void => {
    if (!state.isRunning) {
      return;
    }
    state.shouldStop = true;
    state.isPaused = false;
    resumeTimer();
  };

  const dispose = (): void => {
    state.bufferPool?.dispose();
    state.configBuffer?.destroy();
    state.configBuffer = null;
    state.configData = null;
    state.pipeline = null;
    state.bindGroupLayout = null;
    state.bufferPool = null;
    state.planner = null;
    state.deviceContext = null;
    state.targetBuffer?.destroy();
    state.targetBuffer = null;
    state.targetBufferCapacity = 0;
  };

  const executeSegments = async (
    context: WebGpuSearchContext,
    progress: WebGpuRunnerProgress,
    callbacks: WebGpuRunnerCallbacks
  ): Promise<void> => {
    if (
      !state.deviceContext ||
      !state.pipeline ||
      !state.bufferPool ||
      !state.configBuffer ||
      !state.configData ||
      !state.planner ||
      !state.targetBuffer ||
      !state.bindGroupLayout
    ) {
      throw new Error('WebGPU runner is not ready');
    }

    const device = state.deviceContext.getDevice();
    const queue = device.queue;

    const slotCount = state.bufferPool.slotCount;
    const inFlight: Promise<void>[] = Array.from({ length: slotCount }, () => Promise.resolve());
    let dispatchCounter = 0;

    for (const segment of context.segments) {
      if (state.shouldStop) {
        break;
      }

      const plan = state.planner.computePlan(segment.totalMessages);

      for (const dispatch of plan.dispatches) {
        if (state.shouldStop) {
          break;
        }

        await waitIfPaused();
        if (state.shouldStop) {
          break;
        }

        const slotIndex = dispatchCounter % slotCount;
        await inFlight[slotIndex];

        const dispatchContext: DispatchContext = {
          segment,
          dispatchIndex: dispatchCounter,
          messageCount: dispatch.messageCount,
          slotIndex,
        };

        inFlight[slotIndex] = executeDispatch(
          dispatchContext,
          dispatch.baseOffset,
          context,
          progress,
          callbacks,
          queue
        );

        dispatchCounter += 1;
      }
    }

    await Promise.all(inFlight);
  };

  const executeDispatch = async (
    dispatchContext: DispatchContext,
    segmentBaseOffset: number,
    context: WebGpuSearchContext,
    progress: WebGpuRunnerProgress,
    callbacks: WebGpuRunnerCallbacks,
    queue: GPUQueue
  ): Promise<void> => {
    if (
      !state.deviceContext ||
      !state.pipeline ||
      !state.bufferPool ||
      !state.configBuffer ||
      !state.configData ||
      !state.targetBuffer ||
      !state.bindGroupLayout
    ) {
      throw new Error('WebGPU runner is not ready');
    }

    const device = state.deviceContext.getDevice();
    const slot = state.bufferPool.acquire(dispatchContext.slotIndex, dispatchContext.messageCount);

    queue.writeBuffer(slot.output, 0, ZERO_MATCH_HEADER.buffer, ZERO_MATCH_HEADER.byteOffset, ZERO_MATCH_HEADER.byteLength);

    const uploadStart = performance.now();
    writeConfigBuffer(dispatchContext.segment, segmentBaseOffset, dispatchContext.messageCount);
    queue.writeBuffer(state.configBuffer, 0, state.configData.buffer, state.configData.byteOffset, state.configData.byteLength);
    const afterUpload = performance.now();

    const bindGroup = device.createBindGroup({
      label: `gpu-seed-bind-group-${dispatchContext.dispatchIndex}`,
      layout: state.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: state.configBuffer },
        },
        {
          binding: 1,
          resource: { buffer: state.targetBuffer },
        },
        {
          binding: 2,
          resource: { buffer: slot.output },
        },
      ],
    });

    const commandEncoder = device.createCommandEncoder({ label: `gpu-seed-encoder-${dispatchContext.dispatchIndex}` });
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(state.pipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroupCount = Math.ceil(dispatchContext.messageCount / state.workgroupSize);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    const copySize = slot.outputSize;
    commandEncoder.copyBufferToBuffer(slot.output, 0, slot.readback, 0, copySize);

    queue.submit([commandEncoder.finish()]);

    await queue.onSubmittedWorkDone();
    const afterGpuDone = performance.now();

    await slot.readback.mapAsync(GPUMapMode.READ, 0, copySize);
    const mapped = slot.readback.getMappedRange(0, copySize);
    const results = new Uint32Array(mapped.slice(0));
    slot.readback.unmap();
    const afterReadback = performance.now();

    state.profilingCollector?.recordBatch({
      uploadMs: afterUpload - uploadStart,
      dispatchMs: afterGpuDone - afterUpload,
      readbackMs: afterReadback - afterGpuDone,
    });

    const availableRecords = Math.max(
      0,
      Math.floor((results.length - MATCH_OUTPUT_HEADER_WORDS) / MATCH_RECORD_WORDS)
    );
    const rawMatchCount = results[0] ?? 0;
    const clampedMatchCount = Math.min(rawMatchCount, slot.maxRecords, availableRecords);

    await processMatchRecords(
      results,
      clampedMatchCount,
      dispatchContext,
      segmentBaseOffset,
      context,
      progress,
      callbacks
    );
  };

  const processMatchRecords = async (
    matchWords: Uint32Array,
    matchCount: number,
    dispatchContext: DispatchContext,
    segmentBaseOffset: number,
    context: WebGpuSearchContext,
    progress: WebGpuRunnerProgress,
    callbacks: WebGpuRunnerCallbacks
  ): Promise<void> => {
    const segment = dispatchContext.segment;
    const rangeSeconds = segment.rangeSeconds;
    const safeRangeSeconds = Math.max(rangeSeconds, 1);
    const timer0Min = segment.config.timer0Min;
    const vcount = segment.vcount;

    for (let i = 0; i < matchCount; i += 1) {
      if (state.shouldStop) {
        break;
      }

      if ((i % YIELD_INTERVAL) === 0) {
        await waitIfPaused();
        if (state.shouldStop) {
          break;
        }
      }

      const recordOffset = MATCH_OUTPUT_HEADER_WORDS + i * MATCH_RECORD_WORDS;
      const messageIndex = matchWords[recordOffset];
      const seed = matchWords[recordOffset + 1] >>> 0;
      const h0 = matchWords[recordOffset + 2];
      const h1 = matchWords[recordOffset + 3];
      const h2 = matchWords[recordOffset + 4];
      const h3 = matchWords[recordOffset + 5];
      const h4 = matchWords[recordOffset + 6];

      const timer0Index = Math.floor(messageIndex / safeRangeSeconds);
      const secondOffset = messageIndex - timer0Index * safeRangeSeconds;

      const timer0 = timer0Min + timer0Index;
      const datetime = new Date(context.startTimestampMs + secondOffset * 1000);
      const message = state.seedCalculator.generateMessage(context.conditions, timer0, vcount, datetime);
      const result: InitialSeedResult = {
        seed,
        datetime,
        timer0,
        vcount,
        conditions: context.conditions,
        message,
        sha1Hash: SHA1.hashToHex(h0, h1, h2, h3, h4),
        isMatch: true,
      };
      callbacks.onResult(result);
      progress.matchesFound += 1;
    }

    if (dispatchContext.messageCount > 0) {
      const finalIndex = segmentBaseOffset + dispatchContext.messageCount - 1;
      const finalTimer0Index = Math.floor(finalIndex / safeRangeSeconds);
      const finalSecondOffset = finalIndex - finalTimer0Index * safeRangeSeconds;
      const lastDateTimeIso = new Date(context.startTimestampMs + finalSecondOffset * 1000).toISOString();
      progress.currentDateTime = lastDateTimeIso;
    }

    progress.currentStep += dispatchContext.messageCount;
    emitProgress(progress, callbacks);
  };

  const emitProgress = (progress: WebGpuRunnerProgress, callbacks: WebGpuRunnerCallbacks): void => {
    const now = Date.now();
    if (now - state.lastProgressUpdateMs < PROGRESS_INTERVAL_MS && progress.currentStep < progress.totalSteps) {
      return;
    }

    const elapsed = getElapsedTime();
    const estimated = estimateRemainingTime(progress.currentStep, progress.totalSteps, elapsed);

    callbacks.onProgress({
      currentStep: progress.currentStep,
      totalSteps: progress.totalSteps,
      elapsedTime: elapsed,
      estimatedTimeRemaining: estimated,
      matchesFound: progress.matchesFound,
      currentDateTime: progress.currentDateTime,
    });

    state.lastProgressUpdateMs = now;
  };

  const estimateRemainingTime = (currentStep: number, totalSteps: number, elapsed: number): number => {
    if (currentStep === 0 || currentStep >= totalSteps) {
      return 0;
    }
    const avgTimePerStep = elapsed / currentStep;
    const remainingSteps = totalSteps - currentStep;
    return Math.round(avgTimePerStep * remainingSteps);
  };

  const writeConfigBuffer = (segment: WebGpuSegment, baseOffset: number, messageCount: number): void => {
    if (!state.configData) {
      throw new Error('config buffer not prepared');
    }

    const data = state.configData;
    data[0] = messageCount >>> 0;
    data[1] = baseOffset >>> 0;
    data[2] = segment.config.rangeSeconds >>> 0;
    data[3] = segment.config.timer0Min >>> 0;
    data[4] = segment.config.timer0Count >>> 0;
    data[5] = segment.config.vcountMin >>> 0;
    data[6] = segment.config.vcountCount >>> 0;
    data[7] = segment.config.startSecondOfDay >>> 0;
    data[8] = segment.config.startDayOfWeek >>> 0;
    data[9] = segment.config.macLower >>> 0;
    data[10] = segment.config.data7Swapped >>> 0;
    data[11] = segment.config.keyInputSwapped >>> 0;
    data[12] = segment.config.hardwareType >>> 0;
    for (let i = 0; i < segment.config.nazoSwapped.length; i += 1) {
      data[13 + i] = segment.config.nazoSwapped[i] >>> 0;
    }
    data[18] = segment.config.startYear >>> 0;
    data[19] = segment.config.startDayOfYear >>> 0;
  };

  const startTimer = (): void => {
    state.timerState.cumulativeRunTime = 0;
    state.timerState.segmentStartTime = Date.now();
    state.timerState.isPaused = false;
  };

  const pauseTimer = (): void => {
    if (!state.timerState.isPaused) {
      state.timerState.cumulativeRunTime += Date.now() - state.timerState.segmentStartTime;
      state.timerState.isPaused = true;
    }
  };

  const resumeTimer = (): void => {
    if (state.timerState.isPaused) {
      state.timerState.segmentStartTime = Date.now();
      state.timerState.isPaused = false;
    }
  };

  const getElapsedTime = (): number => {
    if (state.timerState.isPaused) {
      return state.timerState.cumulativeRunTime;
    }
    return state.timerState.cumulativeRunTime + (Date.now() - state.timerState.segmentStartTime);
  };

  const waitIfPaused = async (): Promise<void> => {
    while (state.isPaused && !state.shouldStop) {
      await sleep(25);
    }
  };

  const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

  const alignSize = (bytes: number): number => {
    const alignment = 256;
    return Math.ceil(bytes / alignment) * alignment;
  };

  return {
    init,
    run,
    pause,
    resume,
    stop,
    dispose,
  };
}
