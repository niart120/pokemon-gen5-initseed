/// <reference lib="webworker" />

import { prepareSearchJob } from '@/lib/webgpu/seed-search/prepare-search-job';
import {
  createSeedSearchController,
  type SeedSearchController,
} from '@/lib/webgpu/seed-search/seed-search-controller';
import {
  createWebGpuDeviceContext,
  isWebGpuSeedSearchSupported,
  type WebGpuDeviceContext,
} from '@/lib/webgpu/seed-search/device-context';
import { createSeedSearchEngine } from '@/lib/webgpu/seed-search/seed-search-engine';
import type { SeedSearchJobLimits, WebGpuRunnerCallbacks } from '@/lib/webgpu/seed-search/types';
import type { WorkerRequest, WorkerResponse } from '@/types/worker';

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;

const CANDIDATE_CAPACITY_SAFETY_FACTOR = 3n;
const UINT32_RANGE = 0x1_0000_0000n;

interface WorkerState {
  isRunning: boolean;
  isPaused: boolean;
}

const workerState: WorkerState = {
  isRunning: false,
  isPaused: false,
};

let controller: SeedSearchController | null = null;
let abortController: AbortController | null = null;
let deviceContextPromise: Promise<WebGpuDeviceContext> | null = null;
let cachedLimits: SeedSearchJobLimits | null = null;

function estimateCandidateCapacityPerDispatch(
  limits: SeedSearchJobLimits,
  targetSeedCount: number,
): number {
  if (targetSeedCount <= 0 || limits.maxMessagesPerDispatch <= 0) {
    return limits.candidateCapacityPerDispatch;
  }

  const numerator =
    CANDIDATE_CAPACITY_SAFETY_FACTOR *
    BigInt(limits.maxMessagesPerDispatch) *
    BigInt(targetSeedCount);
  const estimated = Number((numerator + UINT32_RANGE - 1n) / UINT32_RANGE);
  return Math.max(1, estimated);
}

async function ensureDeviceContext(): Promise<WebGpuDeviceContext> {
  if (!deviceContextPromise) {
    deviceContextPromise = createWebGpuDeviceContext();
  }
  return deviceContextPromise;
}

async function ensureController(): Promise<SeedSearchController> {
  if (controller) {
    return controller;
  }
  const context = await ensureDeviceContext();
  const engine = createSeedSearchEngine(undefined, context);
  controller = createSeedSearchController(engine);
  return controller;
}

async function ensureJobLimits(): Promise<SeedSearchJobLimits> {
  if (cachedLimits) {
    return cachedLimits;
  }
  const context = await ensureDeviceContext();
  cachedLimits = context.deriveSearchJobLimits();
  return cachedLimits;
}

function postMessageSafely(response: WorkerResponse): void {
  ctx.postMessage(response);
}

function postReady(): void {
  postMessageSafely({ type: 'READY', message: 'WebGPU worker initialized' });
}

function resetState(): void {
  workerState.isRunning = false;
  workerState.isPaused = false;
  abortController = null;
}

function ensureWebGpuAvailable(): boolean {
  if (!isWebGpuSeedSearchSupported()) {
    postMessageSafely({
      type: 'ERROR',
      error: 'WebGPU is not supported in this environment',
      errorCode: 'WEBGPU_UNSUPPORTED',
    });
    return false;
  }
  return true;
}

async function startSearch(request: WorkerRequest): Promise<void> {
  if (workerState.isRunning) {
    postMessageSafely({ type: 'ERROR', error: 'Search is already running' });
    return;
  }

  if (!request.conditions || !request.targetSeeds) {
    postMessageSafely({ type: 'ERROR', error: 'Missing conditions or target seeds' });
    return;
  }

  if (!ensureWebGpuAvailable()) {
    return;
  }

  workerState.isRunning = true;
  workerState.isPaused = false;

  let job;
  let activeController: SeedSearchController;
  try {
    const [limits, controllerInstance] = await Promise.all([
      ensureJobLimits(),
      ensureController(),
    ]);
    const candidateEstimate = estimateCandidateCapacityPerDispatch(limits, request.targetSeeds.length);
    const adjustedLimits: SeedSearchJobLimits = {
      ...limits,
      // Keep the candidate buffer within the default margin while covering expected hits.
      candidateCapacityPerDispatch: Math.min(limits.candidateCapacityPerDispatch, candidateEstimate),
    };
    job = prepareSearchJob(request.conditions, request.targetSeeds, { limits: adjustedLimits });
    activeController = controllerInstance;
  } catch (error) {
    resetState();
    const message = error instanceof Error ? error.message : '検索条件の解析中にエラーが発生しました';
    postMessageSafely({ type: 'ERROR', error: message, errorCode: 'WEBGPU_CONTEXT_ERROR' });
    return;
  }

  abortController = new AbortController();

  const callbacks: WebGpuRunnerCallbacks = {
    onProgress: (progress) => {
      postMessageSafely({ type: 'PROGRESS', progress });
    },
    onResult: (result) => {
      postMessageSafely({ type: 'RESULT', result });
    },
    onComplete: (message) => {
      resetState();
      postMessageSafely({ type: 'COMPLETE', message });
    },
    onError: (error, errorCode) => {
      resetState();
      postMessageSafely({ type: 'ERROR', error, errorCode });
    },
    onPaused: () => {
      workerState.isPaused = true;
      postMessageSafely({ type: 'PAUSED' });
    },
    onResumed: () => {
      workerState.isPaused = false;
      postMessageSafely({ type: 'RESUMED' });
    },
    onStopped: (message, finalProgress) => {
      resetState();
      postMessageSafely({ type: 'STOPPED', message, progress: finalProgress });
    },
  };

  try {
    await activeController.run(job, callbacks, abortController.signal);
  } catch (error) {
    if (!workerState.isRunning) {
      return;
    }
    resetState();
    const message = error instanceof Error ? error.message : 'WebGPU search failed with unknown error';
    postMessageSafely({ type: 'ERROR', error: message, errorCode: 'WEBGPU_RUNTIME_ERROR' });
  }
}

function pauseSearch(): void {
  if (!workerState.isRunning || workerState.isPaused) {
    return;
  }
  controller?.pause();
}

function resumeSearch(): void {
  if (!workerState.isRunning || !workerState.isPaused) {
    return;
  }
  controller?.resume();
}

function stopSearch(): void {
  if (!workerState.isRunning) {
    return;
  }
  controller?.stop();
  abortController?.abort();
}

postReady();

ctx.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;

  switch (request.type) {
    case 'START_SEARCH':
      void startSearch(request);
      break;

    case 'PAUSE_SEARCH':
      pauseSearch();
      break;

    case 'RESUME_SEARCH':
      resumeSearch();
      break;

    case 'STOP_SEARCH':
      stopSearch();
      break;

    default:
      postMessageSafely({ type: 'ERROR', error: `Unknown request type: ${request.type}` });
  }
};
