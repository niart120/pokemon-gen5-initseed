import { SeedCalculator } from '@/lib/core/seed-calculator';
import type { InitialSeedResult } from '@/types/search';
import {
  MATCH_OUTPUT_HEADER_WORDS,
  MATCH_RECORD_WORDS,
} from './constants';
import { createSeedSearchEngine, type SeedSearchEngine } from './seed-search-engine';
import type {
  SeedSearchJob,
  SeedSearchJobSegment,
  WebGpuRunnerCallbacks,
  WebGpuRunnerProgress,
} from './types';
import { getDateFromTimePlan } from '@/lib/search/time/time-plan';

const YIELD_INTERVAL = 1024;
const PROGRESS_INTERVAL_MS = 500;

interface TimerState {
  cumulativeRunTime: number;
  segmentStartTime: number;
  isPaused: boolean;
}

interface ControllerState {
  isRunning: boolean;
  isPaused: boolean;
  shouldStop: boolean;
  job: SeedSearchJob | null;
  progress: WebGpuRunnerProgress | null;
  callbacks: WebGpuRunnerCallbacks | null;
  abortCleanup?: () => void;
  timer: TimerState;
  lastProgressUpdate: number;
}

export interface SeedSearchController {
  run(job: SeedSearchJob, callbacks: WebGpuRunnerCallbacks, signal?: AbortSignal): Promise<void>;
  pause(): void;
  resume(): void;
  stop(): void;
}

export function createSeedSearchController(engine?: SeedSearchEngine): SeedSearchController {
  const seedCalculator = new SeedCalculator();
  const searchEngine = engine ?? createSeedSearchEngine();
  const state: ControllerState = {
    isRunning: false,
    isPaused: false,
    shouldStop: false,
    job: null,
    progress: null,
    callbacks: null,
    timer: {
      cumulativeRunTime: 0,
      segmentStartTime: 0,
      isPaused: false,
    },
    lastProgressUpdate: 0,
  };

  const run = async (
    job: SeedSearchJob,
    callbacks: WebGpuRunnerCallbacks,
    signal?: AbortSignal
  ): Promise<void> => {
    if (state.isRunning) {
      throw new Error('Seed search is already running');
    }

    state.isRunning = true;
    state.isPaused = false;
    state.shouldStop = signal?.aborted ?? false;
    state.job = job;
    state.callbacks = callbacks;
    state.lastProgressUpdate = 0;
    state.progress = {
      currentStep: 0,
      totalSteps: job.summary.totalMessages,
      elapsedTime: 0,
      estimatedTimeRemaining: 0,
      matchesFound: 0,
      currentDateTime: job.timePlan ? new Date(job.timePlan.startDayTimestampMs).toISOString() : undefined,
    };
    startTimer();
    emitProgress(false);

    let abortCleanup: (() => void) | undefined;
    if (signal) {
      const onAbort = () => {
        state.shouldStop = true;
      };
      signal.addEventListener('abort', onAbort);
      abortCleanup = () => signal.removeEventListener('abort', onAbort);
      state.abortCleanup = abortCleanup;
    }

    try {
      if (job.summary.totalMessages === 0) {
        callbacks.onComplete('探索対象の組み合わせが存在しません');
        return;
      }

      const dispatchConcurrency = Math.max(
        1,
        Math.min(job.limits.maxDispatchesInFlight ?? 1, job.segments.length || 1)
      );
      await searchEngine.ensureConfigured(job.limits, { dispatchSlots: dispatchConcurrency });
      searchEngine.setTargetSeeds(job.targetSeeds);
      const dispatchInflight = new Set<Promise<void>>();
      const processingInflight = new Set<Promise<void>>();

      const scheduleSegment = (segment: SeedSearchJobSegment): void => {
        let trackedDispatch: Promise<void>;
        const dispatchPromise = (async () => {
          if (state.shouldStop) {
            return;
          }
          const { words, matchCount } = await searchEngine.executeSegment(segment);
          if (state.shouldStop) {
            return;
          }
          let trackedProcessing: Promise<void>;
          const processingPromise = (async () => {
            await processMatches(segment, words, matchCount);
          })();
          trackedProcessing = processingPromise.finally(() => processingInflight.delete(trackedProcessing));
          processingInflight.add(trackedProcessing);
        })();

        trackedDispatch = dispatchPromise.finally(() => dispatchInflight.delete(trackedDispatch));
        dispatchInflight.add(trackedDispatch);
      };

      for (const segment of job.segments) {
        if (state.shouldStop) {
          break;
        }

        await waitIfPaused();
        if (state.shouldStop) {
          break;
        }

        scheduleSegment(segment);

        if (dispatchInflight.size >= dispatchConcurrency) {
          await Promise.race(dispatchInflight);
        }
      }

      if (dispatchInflight.size > 0) {
        await Promise.all(dispatchInflight);
      }
      if (processingInflight.size > 0) {
        await Promise.all(processingInflight);
      }

      finalizeRun();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'WebGPU検索中に不明なエラーが発生しました';
      const validationCtor = (globalThis as unknown as {
        GPUValidationError?: { new (...args: unknown[]): Error };
      }).GPUValidationError;
      const code = validationCtor && error instanceof validationCtor ? 'WEBGPU_VALIDATION_ERROR' : undefined;
      callbacks.onError(message, code);
      throw error;
    } finally {
      abortCleanup?.();
      state.abortCleanup = undefined;
      state.isRunning = false;
      state.isPaused = false;
      stopTimer();
      state.job = null;
      state.callbacks = null;
      state.progress = null;
      state.shouldStop = false;
      state.lastProgressUpdate = 0;
    }
  };

  const finalizeRun = (): void => {
    const callbacks = state.callbacks;
    const progress = state.progress;
    if (!callbacks || !progress) {
      return;
    }

    updateProgressTime(progress);

    if (state.shouldStop) {
      callbacks.onStopped('検索を停止しました', progress);
      return;
    }

    callbacks.onProgress(progress);
    callbacks.onComplete(`検索が完了しました。${progress.matchesFound}件ヒットしました。`);
  };

  const processMatches = async (
    segment: SeedSearchJobSegment,
    matchWords: Uint32Array,
    matchCount: number
  ): Promise<void> => {
    const job = state.job;
    const callbacks = state.callbacks;
    const progress = state.progress;
    if (!job || !callbacks || !progress) {
      return;
    }

    const headerWords = MATCH_OUTPUT_HEADER_WORDS;
    const recordWords = MATCH_RECORD_WORDS;

    for (let i = 0; i < matchCount; i += 1) {
      if (state.shouldStop) {
        break;
      }

      if (i % YIELD_INTERVAL === 0) {
        await waitIfPaused();
        if (state.shouldStop) {
          break;
        }
      }

      const recordOffset = headerWords + i * recordWords;
      const localMessageIndex = matchWords[recordOffset];
      const seed = matchWords[recordOffset + 1] >>> 0;
      const messageIndex = segment.globalMessageOffset + localMessageIndex;
      const timeCombinationOffset = segment.baseSecondOffset + localMessageIndex;
      const timer0 = segment.timer0;
      const vcount = segment.vcount;
      const datetime = getDateFromTimePlan(job.timePlan, timeCombinationOffset);
      const message = seedCalculator.generateMessage(job.conditions, timer0, vcount, datetime, segment.keyCode);
      const { hash, seed: recalculatedSeed, lcgSeed } = seedCalculator.calculateSeed(message);

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
        keyCode: segment.keyCode,
        conditions: job.conditions,
        message,
        sha1Hash: hash,
        lcgSeed,
        isMatch: true,
      };
      callbacks.onResult(result);
      progress.matchesFound += 1;
    }

    if (segment.messageCount > 0) {
      const finalLocalIndex = segment.messageCount - 1;
      const finalTimeCombination = segment.baseSecondOffset + finalLocalIndex;
      progress.currentDateTime = getDateFromTimePlan(job.timePlan, finalTimeCombination).toISOString();
    }

    progress.currentStep += segment.messageCount;
    emitProgress(true);
  };

  const pause = (): void => {
    if (!state.isRunning || state.isPaused) {
      return;
    }
    state.isPaused = true;
    pauseTimer();
    state.callbacks?.onPaused();
  };

  const resume = (): void => {
    if (!state.isRunning || !state.isPaused) {
      return;
    }
    state.isPaused = false;
    resumeTimer();
    state.callbacks?.onResumed();
  };

  const stop = (): void => {
    if (!state.isRunning) {
      return;
    }
    state.shouldStop = true;
    state.isPaused = false;
    resumeTimer();
  };

  const waitIfPaused = async (): Promise<void> => {
    while (state.isPaused && !state.shouldStop) {
      await sleep(25);
    }
  };

  const emitProgress = (throttle: boolean): void => {
    const callbacks = state.callbacks;
    const progress = state.progress;
    if (!callbacks || !progress) {
      return;
    }

    const now = Date.now();
    if (throttle && progress.currentStep < progress.totalSteps) {
      if (now - state.lastProgressUpdate < PROGRESS_INTERVAL_MS) {
        return;
      }
    }

    updateProgressTime(progress);
    callbacks.onProgress(progress);
    state.lastProgressUpdate = now;
  };

  const updateProgressTime = (progress: WebGpuRunnerProgress): void => {
    const elapsed = getElapsedTime();
    progress.elapsedTime = elapsed;
    progress.estimatedTimeRemaining = estimateRemainingTime(
      progress.currentStep,
      progress.totalSteps,
      elapsed
    );
  };

  const startTimer = (): void => {
    state.timer.cumulativeRunTime = 0;
    state.timer.segmentStartTime = Date.now();
    state.timer.isPaused = false;
  };

  const pauseTimer = (): void => {
    if (!state.timer.isPaused) {
      state.timer.cumulativeRunTime += Date.now() - state.timer.segmentStartTime;
      state.timer.isPaused = true;
    }
  };

  const resumeTimer = (): void => {
    if (state.timer.isPaused) {
      state.timer.segmentStartTime = Date.now();
      state.timer.isPaused = false;
    }
  };

  const stopTimer = (): void => {
    if (!state.timer.isPaused) {
      state.timer.cumulativeRunTime += Date.now() - state.timer.segmentStartTime;
      state.timer.isPaused = true;
    }
  };

  const getElapsedTime = (): number => {
    if (state.timer.isPaused) {
      return state.timer.cumulativeRunTime;
    }
    return state.timer.cumulativeRunTime + (Date.now() - state.timer.segmentStartTime);
  };

  const estimateRemainingTime = (currentStep: number, totalSteps: number, elapsed: number): number => {
    if (currentStep === 0 || currentStep >= totalSteps) {
      return 0;
    }
    const avgTimePerStep = elapsed / currentStep;
    return Math.round(avgTimePerStep * (totalSteps - currentStep));
  };

  const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

  return {
    run,
    pause,
    resume,
    stop,
  };
}
