import { SeedCalculator } from '@/lib/core/seed-calculator';
import type { InitialSeedResult } from '@/types/search';
import {
  DEFAULT_HOST_MEMORY_LIMIT_BYTES,
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
import type {
  WebGpuRunRequest,
  WebGpuRunnerCallbacks,
  WebGpuRunnerProgress,
  WebGpuRunnerInstrumentation,
  WebGpuRunnerSpanKind,
  WebGpuSearchContext,
  WebGpuSegment,
} from './types';
import { getDateFromTimePlan } from './time-plan';

interface DispatchContext {
  segment: WebGpuSegment;
  dispatchIndex: number;
  messageCount: number;
  slotIndex: number;
}

const CONFIG_WORD_COUNT = 32;
const PROGRESS_INTERVAL_MS = 500;
const YIELD_INTERVAL = 1024;
const ZERO_MATCH_HEADER = new Uint32Array([0]);

interface RunnerState {
  workgroupSize: number;
  bufferSlotCount: number;
  hostMemoryLimitBytes: number;
  hostMemoryLimitPerSlotBytes: number;
  deviceContext: WebGpuDeviceContext | null;
  pipeline: GPUComputePipeline | null;
  bindGroupLayout: GPUBindGroupLayout | null;
  configBuffer: GPUBuffer | null;
  configData: Uint32Array | null;
  bufferPool: WebGpuBufferPool | null;
  planner: WebGpuBatchPlanner | null;
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

export interface WebGpuSeedSearchRunnerOptions {
  workgroupSize?: number;
  maxMessagesPerDispatch?: number;
  instrumentation?: WebGpuRunnerInstrumentation;
  bufferSlots?: number;
  hostMemoryLimitBytes?: number;
  hostMemoryLimitPerSlotBytes?: number;
}

export function createWebGpuSeedSearchRunner(options?: WebGpuSeedSearchRunnerOptions): WebGpuSeedSearchRunner {
  const resolveBufferSlotCount = (slots: number | undefined): number => {
    if (typeof slots !== 'number' || !Number.isFinite(slots)) {
      return DOUBLE_BUFFER_SET_COUNT;
    }
    return Math.max(1, Math.floor(slots));
  };

  const totalHostMemoryLimitBytes = (() => {
    if (typeof options?.hostMemoryLimitBytes === 'number' && Number.isFinite(options.hostMemoryLimitBytes) && options.hostMemoryLimitBytes > 0) {
      return options.hostMemoryLimitBytes;
    }
    return DEFAULT_HOST_MEMORY_LIMIT_BYTES;
  })();

  const bufferSlotCount = resolveBufferSlotCount(options?.bufferSlots);
  const hostMemoryLimitPerSlotBytes = (() => {
    if (
      typeof options?.hostMemoryLimitPerSlotBytes === 'number' &&
      Number.isFinite(options.hostMemoryLimitPerSlotBytes) &&
      options.hostMemoryLimitPerSlotBytes > 0
    ) {
      return options.hostMemoryLimitPerSlotBytes;
    }
    const calculated = Math.floor(totalHostMemoryLimitBytes / bufferSlotCount);
    return Math.max(1, calculated);
  })();

  const state: RunnerState = {
    workgroupSize: options?.workgroupSize ?? DEFAULT_WORKGROUP_SIZE,
    bufferSlotCount,
    hostMemoryLimitBytes: totalHostMemoryLimitBytes,
    hostMemoryLimitPerSlotBytes,
    deviceContext: null,
    pipeline: null,
    bindGroupLayout: null,
    configBuffer: null,
    configData: null,
    bufferPool: null,
    planner: null,
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

  const instrumentation = options?.instrumentation ?? null;

  const runWithTrace = async <T>(
    kind: WebGpuRunnerSpanKind,
    metadata: Record<string, unknown>,
    operation: () => Promise<T> | T
  ): Promise<T> => {
    const execute = async () => await Promise.resolve(operation());
    if (!instrumentation?.trace) {
      return execute();
    }
    return instrumentation.trace({ kind, metadata }, execute);
  };

  const init = async (): Promise<void> => {
    if (state.pipeline && state.bufferPool && state.planner && state.deviceContext) {
      return;
    }

    const context = await createWebGpuDeviceContext();
    const device = context.getDevice();
    const resolvedWorkgroupSize = context.getSupportedWorkgroupSize(state.workgroupSize);
    const { pipeline, layout } = createGeneratedPipeline(device, resolvedWorkgroupSize);

    const configData = new Uint32Array(CONFIG_WORD_COUNT);
    const configSize = alignSize(configData.byteLength);
    const configBuffer = device.createBuffer({
      label: 'gpu-seed-config-buffer',
      size: configSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferPool = createWebGpuBufferPool(device, {
      slots: state.bufferSlotCount,
    });

    const planner = createWebGpuBatchPlanner(context, {
      workgroupSize: resolvedWorkgroupSize,
      bufferSetCount: state.bufferSlotCount,
      hostMemoryLimitBytes: state.hostMemoryLimitBytes,
      hostMemoryLimitPerSlot: state.hostMemoryLimitPerSlotBytes,
      maxMessagesOverride: options?.maxMessagesPerDispatch ?? null,
    });

    state.deviceContext = context;
    state.pipeline = pipeline;
    state.bindGroupLayout = layout;
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
    const availableSlots: number[] = Array.from({ length: slotCount }, (_, index) => slotCount - 1 - index);
    const pendingSlotResolvers: Array<(index: number) => void> = [];
    const activeDispatchPromises: Promise<void>[] = [];
    const finalizeTasks: Promise<void>[] = [];

    const registerFinalizeTask = (task: Promise<void>): void => {
      finalizeTasks.push(task);
    };

    const acquireSlot = (): Promise<number> =>
      new Promise((resolve) => {
        if (availableSlots.length > 0) {
          const slotIndex = availableSlots.pop()!;
          resolve(slotIndex);
          return;
        }
        pendingSlotResolvers.push(resolve);
      });

    const releaseSlot = (slotIndex: number): void => {
      const resolver = pendingSlotResolvers.shift();
      if (resolver) {
        resolver(slotIndex);
        return;
      }
      availableSlots.push(slotIndex);
    };

    let dispatchCounter = 0;

    for (const segment of context.segments) {
      if (state.shouldStop) {
        break;
      }

      const plan = await runWithTrace(
        'planner.computePlan',
        {
          segmentIndex: segment.index,
          totalMessages: segment.totalMessages,
        },
        () => Promise.resolve(state.planner!.computePlan(segment.totalMessages))
      );

      for (const dispatch of plan.dispatches) {
        if (state.shouldStop) {
          break;
        }

        await waitIfPaused();
        if (state.shouldStop) {
          break;
        }

        const slotIndex = await acquireSlot();
        if (state.shouldStop) {
          releaseSlot(slotIndex);
          break;
        }

        const dispatchContext: DispatchContext = {
          segment,
          dispatchIndex: dispatchCounter,
          messageCount: dispatch.messageCount,
          slotIndex,
        };

        const dispatchPromise = executeDispatch(
          dispatchContext,
          dispatch.baseOffset,
          context,
          progress,
          callbacks,
          queue,
          releaseSlot,
          registerFinalizeTask
        );

        activeDispatchPromises.push(dispatchPromise);

        dispatchCounter += 1;
      }
    }

    if (activeDispatchPromises.length > 0) {
      await Promise.all(activeDispatchPromises);
    }
    if (finalizeTasks.length > 0) {
      await Promise.all(finalizeTasks);
    }
  };

  const executeDispatch = async (
    dispatchContext: DispatchContext,
    segmentBaseOffset: number,
    context: WebGpuSearchContext,
    progress: WebGpuRunnerProgress,
    callbacks: WebGpuRunnerCallbacks,
    queue: GPUQueue,
    releaseSlot: (slotIndex: number) => void,
    registerFinalizeTask: (task: Promise<void>) => void
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
    const configBuffer = state.configBuffer!;
    const configData = state.configData!;
    const bindGroupLayout = state.bindGroupLayout!;
    const pipeline = state.pipeline!;
    const targetBuffer = state.targetBuffer!;
    const slot = state.bufferPool.acquire(dispatchContext.slotIndex, dispatchContext.messageCount);
    let slotReleased = false;
    let finalizeRegistered = false;
    const releaseSlotOnce = () => {
      if (slotReleased) {
        return;
      }
      slotReleased = true;
      releaseSlot(dispatchContext.slotIndex);
    };
    const workgroupCount = Math.ceil(dispatchContext.messageCount / state.workgroupSize);
    const groupCount = Math.max(1, workgroupCount);
    const headerCopySize = alignSize(MATCH_OUTPUT_HEADER_WORDS * Uint32Array.BYTES_PER_ELEMENT);

    const baseDispatchMetadata = {
      dispatchIndex: dispatchContext.dispatchIndex,
      messageCount: dispatchContext.messageCount,
      slotIndex: dispatchContext.slotIndex,
      workgroupCount,
      candidateCapacity: slot.candidateCapacity,
      segmentIndex: dispatchContext.segment.index,
      segmentBaseOffset,
    } satisfies Record<string, unknown>;

    try {
      await runWithTrace('dispatch', baseDispatchMetadata, async () => {
        queue.writeBuffer(
          slot.output,
          0,
          ZERO_MATCH_HEADER.buffer,
          ZERO_MATCH_HEADER.byteOffset,
          ZERO_MATCH_HEADER.byteLength
        );

        writeConfigBuffer(
          dispatchContext.segment,
          segmentBaseOffset,
          dispatchContext.messageCount,
          groupCount,
          slot.candidateCapacity
        );
        queue.writeBuffer(
          configBuffer,
          0,
          configData.buffer,
          configData.byteOffset,
          configData.byteLength
        );

        const generateBindGroup = device.createBindGroup({
          label: `gpu-seed-generate-group-${dispatchContext.dispatchIndex}`,
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: configBuffer } },
            { binding: 1, resource: { buffer: targetBuffer } },
            { binding: 2, resource: { buffer: slot.output } },
          ],
        });

        const computeEncoder = device.createCommandEncoder({
          label: `gpu-seed-compute-${dispatchContext.dispatchIndex}`,
        });
        const generatePass = computeEncoder.beginComputePass({
          label: `gpu-seed-generate-pass-${dispatchContext.dispatchIndex}`,
        });
        generatePass.setPipeline(pipeline);
        generatePass.setBindGroup(0, generateBindGroup);
        generatePass.dispatchWorkgroups(workgroupCount);
        generatePass.end();

        computeEncoder.copyBufferToBuffer(slot.output, 0, slot.matchCount, 0, headerCopySize);

        const computeCommands = computeEncoder.finish();

        await runWithTrace('dispatch.submit', { ...baseDispatchMetadata }, async () => {
          await runWithTrace('dispatch.submit.encode', { ...baseDispatchMetadata }, async () => {
            queue.submit([computeCommands]);
          });
        });

        const rawMatchCount = await runWithTrace(
          'dispatch.mapMatchCount',
          { ...baseDispatchMetadata, headerCopyBytes: headerCopySize },
          async () => {
            await slot.matchCount.mapAsync(GPUMapMode.READ, 0, headerCopySize);
            const headerView = new Uint32Array(slot.matchCount.getMappedRange(0, headerCopySize));
            const count = headerView[0] ?? 0;
            slot.matchCount.unmap();
            return count;
          }
        );

        const clampedPlannedMatchCount = Math.min(rawMatchCount, slot.maxRecords);
        const recordsBytes = clampedPlannedMatchCount * MATCH_RECORD_WORDS * Uint32Array.BYTES_PER_ELEMENT;
        const totalCopyBytes = alignSize(
          MATCH_OUTPUT_HEADER_WORDS * Uint32Array.BYTES_PER_ELEMENT + recordsBytes
        );

        await runWithTrace(
          'dispatch.copyResults',
          { ...baseDispatchMetadata, totalCopyBytes },
          async () => {
            const copyEncoder = device.createCommandEncoder({
              label: `gpu-seed-copy-${dispatchContext.dispatchIndex}`,
            });
            copyEncoder.copyBufferToBuffer(slot.output, 0, slot.readback, 0, totalCopyBytes);
            const copyCommands = copyEncoder.finish();
            await runWithTrace('dispatch.copyResults.encode', { ...baseDispatchMetadata, totalCopyBytes }, async () => {
              queue.submit([copyCommands]);
            });
          }
        );
        const finalizePromise = (async () => {
          try {
            const { results, clampedMatchCount } = await runWithTrace(
              'dispatch.mapResults',
              { ...baseDispatchMetadata, totalCopyBytes },
              async () => {
                await slot.readback.mapAsync(GPUMapMode.READ, 0, totalCopyBytes);
                const mappedRange = slot.readback.getMappedRange(0, totalCopyBytes);
                const mappedWords = new Uint32Array(mappedRange);
                const finalRawMatchCount = mappedWords[0] ?? 0;
                const availableRecords = Math.max(
                  0,
                  Math.floor((mappedWords.length - MATCH_OUTPUT_HEADER_WORDS) / MATCH_RECORD_WORDS)
                );
                const clampedMatchCountInner = Math.min(finalRawMatchCount, slot.maxRecords, availableRecords);
                const wordsToCopy = MATCH_OUTPUT_HEADER_WORDS + clampedMatchCountInner * MATCH_RECORD_WORDS;
                const resultsArray = new Uint32Array(wordsToCopy);
                resultsArray.set(mappedWords.subarray(0, wordsToCopy));
                slot.readback.unmap();
                return {
                  results: resultsArray,
                  clampedMatchCount: clampedMatchCountInner,
                };
              }
            );

            try {
              releaseSlotOnce();
              await runWithTrace(
                'dispatch.processMatches',
                { ...baseDispatchMetadata, matchCount: clampedMatchCount },
                () =>
                  processMatchRecords(
                    results,
                    clampedMatchCount,
                    dispatchContext,
                    segmentBaseOffset,
                    context,
                    progress,
                    callbacks
                  )
              );
            } finally {
              releaseSlotOnce();
            }
          } catch (error) {
            releaseSlotOnce();
            throw error;
          }
        })();

        finalizeRegistered = true;
        registerFinalizeTask(finalizePromise);
      });
    } finally {
      if (!finalizeRegistered) {
        releaseSlotOnce();
      }
    }
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
    const safeVcountCount = Math.max(segment.config.vcountCount, 1);
    const messagesPerVcount = safeRangeSeconds;
    const messagesPerTimer0 = messagesPerVcount * safeVcountCount;
    const timer0Min = segment.config.timer0Min;
    const vcountMin = segment.config.vcountMin;
    const dispatchBaseOffset = segmentBaseOffset;

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
      const localMessageIndex = matchWords[recordOffset];
      const messageIndex = dispatchBaseOffset + localMessageIndex;
      const seed = matchWords[recordOffset + 1] >>> 0;

      const timer0Index = Math.floor(messageIndex / messagesPerTimer0);
      const remainderAfterTimer0 = messageIndex - timer0Index * messagesPerTimer0;
      const vcountIndex = Math.floor(remainderAfterTimer0 / messagesPerVcount);
      const timeCombinationOffset = remainderAfterTimer0 - vcountIndex * messagesPerVcount;

      const timer0 = timer0Min + timer0Index;
      const vcount = vcountMin + vcountIndex;
      const datetime = getDateFromTimePlan(context.timePlan, timeCombinationOffset);
      const keyCode = dispatchContext.segment.keyCode;
      const message = state.seedCalculator.generateMessage(context.conditions, timer0, vcount, datetime, keyCode);
      const { hash, seed: recalculatedSeed, lcgSeed } = state.seedCalculator.calculateSeed(message);
      if (recalculatedSeed !== seed) {
        console.warn('GPU/CPU seed mismatch detected', {
          gpuSeed: seed,
          cpuSeed: recalculatedSeed,
          messageIndex,
        });
      }
      const result: InitialSeedResult = {
        seed,
        datetime,
        timer0,
        vcount,
        keyCode,
        conditions: context.conditions,
        message,
        sha1Hash: hash,
        lcgSeed,
        isMatch: true,
      };
      callbacks.onResult(result);
      progress.matchesFound += 1;
    }

    if (dispatchContext.messageCount > 0) {
      const finalLocalIndex = dispatchContext.messageCount - 1;
      const finalMessageIndex = dispatchBaseOffset + finalLocalIndex;
      const finalTimer0Index = Math.floor(finalMessageIndex / messagesPerTimer0);
      const finalRemainderAfterTimer0 = finalMessageIndex - finalTimer0Index * messagesPerTimer0;
      const finalVcountIndex = Math.floor(finalRemainderAfterTimer0 / messagesPerVcount);
      const finalTimeCombination = finalRemainderAfterTimer0 - finalVcountIndex * messagesPerVcount;
      const lastDateTimeIso = getDateFromTimePlan(context.timePlan, finalTimeCombination).toISOString();
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

  const writeConfigBuffer = (
    segment: WebGpuSegment,
    baseOffset: number,
    messageCount: number,
    groupCount: number,
    candidateCapacity: number
  ): void => {
    if (!state.configData) {
      throw new Error('config buffer not prepared');
    }

    const timeCombinationCount = Math.max(segment.config.rangeSeconds, 1);
    const safeVcountCount = Math.max(segment.config.vcountCount, 1);
    const messagesPerVcount = timeCombinationCount;
    const messagesPerTimer0 = messagesPerVcount * safeVcountCount;

    const baseTimer0Index = Math.floor(baseOffset / messagesPerTimer0);
    const remainderAfterTimer0 = baseOffset - baseTimer0Index * messagesPerTimer0;
    const baseVcountIndex = Math.floor(remainderAfterTimer0 / messagesPerVcount);
    const baseSecondOffset = remainderAfterTimer0 - baseVcountIndex * messagesPerVcount;

    const data = state.configData;
    data[0] = messageCount >>> 0;
    data[1] = baseTimer0Index >>> 0;
    data[2] = baseVcountIndex >>> 0;
    data[3] = baseSecondOffset >>> 0;
    data[4] = segment.config.rangeSeconds >>> 0;
    data[5] = segment.config.timer0Min >>> 0;
    data[6] = segment.config.timer0Count >>> 0;
    data[7] = segment.config.vcountMin >>> 0;
    data[8] = segment.config.vcountCount >>> 0;
    data[9] = segment.config.startSecondOfDay >>> 0;
    data[10] = segment.config.startDayOfWeek >>> 0;
    data[11] = segment.config.macLower >>> 0;
    data[12] = segment.config.data7Swapped >>> 0;
    data[13] = segment.config.keyInputSwapped >>> 0;
    data[14] = segment.config.hardwareType >>> 0;
    for (let i = 0; i < segment.config.nazoSwapped.length; i += 1) {
      data[15 + i] = segment.config.nazoSwapped[i] >>> 0;
    }
    data[20] = segment.config.startYear >>> 0;
    data[21] = segment.config.startDayOfYear >>> 0;
    data[22] = groupCount >>> 0;
    data[23] = state.workgroupSize >>> 0;
    data[24] = candidateCapacity >>> 0;
    data[25] = segment.config.dayCount >>> 0;
    data[26] = segment.config.hourRangeStart >>> 0;
    data[27] = segment.config.hourRangeCount >>> 0;
    data[28] = segment.config.minuteRangeStart >>> 0;
    data[29] = segment.config.minuteRangeCount >>> 0;
    data[30] = segment.config.secondRangeStart >>> 0;
    data[31] = segment.config.secondRangeCount >>> 0;
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
