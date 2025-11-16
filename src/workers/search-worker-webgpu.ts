/// <reference lib="webworker" />

import { buildSearchContext } from '@/lib/webgpu/seed-search/message-encoder';
import { createWebGpuSeedSearchRunner, isWebGpuSeedSearchSupported } from '@/lib/webgpu/seed-search/runner';
import type { WebGpuRunRequest, WebGpuRunnerCallbacks } from '@/lib/webgpu/seed-search/types';
import type { WorkerRequest, WorkerResponse } from '@/types/worker';

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;

interface WorkerState {
  isRunning: boolean;
  isPaused: boolean;
}

const workerState: WorkerState = {
  isRunning: false,
  isPaused: false,
};

const runner = createWebGpuSeedSearchRunner();
let abortController: AbortController | null = null;

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

  let context;
  try {
    context = buildSearchContext(request.conditions);
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

  const runRequest: WebGpuRunRequest = {
    context,
    targetSeeds: request.targetSeeds,
    callbacks,
    signal: abortController.signal,
  };

  try {
    await runner.run(runRequest);
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
  runner.pause();
  workerState.isPaused = true;
  postMessageSafely({ type: 'PAUSED' });
}

function resumeSearch(): void {
  if (!workerState.isRunning || !workerState.isPaused) {
    return;
  }
  runner.resume();
  workerState.isPaused = false;
  postMessageSafely({ type: 'RESUMED' });
}

function stopSearch(): void {
  if (!workerState.isRunning) {
    return;
  }
  runner.stop();
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
