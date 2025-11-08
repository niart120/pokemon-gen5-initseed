export interface WebGpuProfilingSample {
  totalMs: number;
  uploadMs: number;
  dispatchMs: number;
  readbackMs: number;
  batches: number;
}

export interface WebGpuBatchTiming {
  uploadMs: number;
  dispatchMs: number;
  readbackMs: number;
}

export interface WebGpuProfilingCollector {
  recordBatch(timing: WebGpuBatchTiming): void;
  toSample(): WebGpuProfilingSample;
}

export function createWebGpuProfilingCollector(): WebGpuProfilingCollector {
  const totals = {
    totalMs: 0,
    uploadMs: 0,
    dispatchMs: 0,
    readbackMs: 0,
    batches: 0,
  };

  const recordBatch = (timing: WebGpuBatchTiming): void => {
    totals.totalMs += timing.uploadMs + timing.dispatchMs + timing.readbackMs;
    totals.uploadMs += timing.uploadMs;
    totals.dispatchMs += timing.dispatchMs;
    totals.readbackMs += timing.readbackMs;
    totals.batches += 1;
  };

  const toSample = (): WebGpuProfilingSample => ({ ...totals });

  return {
    recordBatch,
    toSample,
  };
}
