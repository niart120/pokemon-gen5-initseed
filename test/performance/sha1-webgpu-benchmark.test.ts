/// <reference types="@webgpu/types" />

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { benchmarkSha1, createRandomMessages, runCpuSha1 } from '@/test-utils/perf/sha1-webgpu-harness';
import { WebGpuSha1Runner } from '@/test-utils/webgpu/webgpu-sha1-runner';

const hasWebGpu = typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined' && navigator.gpu !== null;
const describeWebGpu = hasWebGpu ? describe : describe.skip;

describeWebGpu('WebGPU SHA-1 benchmark', () => {
  let runner: WebGpuSha1Runner;

  beforeAll(async () => {
    runner = new WebGpuSha1Runner();
    await runner.init();
  });

  afterAll(() => {
    runner?.dispose();
  });

  it('produces identical hashes for a small fixture batch', async () => {
    const messages = createRandomMessages(8);
    const cpuHashes = runCpuSha1(messages);
    const gpuHashes = await runner.compute(messages);

    expect(Array.from(gpuHashes)).toEqual(Array.from(cpuHashes));
  });

  it('collects timing metrics for a stress workload', async () => {
    const messages = createRandomMessages(1024);

    const cpuStats = await benchmarkSha1({
      runner: (input) => runCpuSha1(input),
      messages,
      iterations: 5,
      warmupIterations: 1,
    });

    const gpuStats = await benchmarkSha1({
      runner: (input) => runner.compute(input),
      messages,
      iterations: 5,
      warmupIterations: 1,
    });

    const speedup = cpuStats.averageMs === 0 ? 0 : cpuStats.averageMs / gpuStats.averageMs;
    console.info(
      `[sha1] CPU avg: ${cpuStats.averageMs.toFixed(3)}ms, WebGPU avg: ${gpuStats.averageMs.toFixed(3)}ms, speedup x${speedup.toFixed(2)}`
    );

    expect(cpuStats.samples).toBeGreaterThan(0);
    expect(gpuStats.samples).toBeGreaterThan(0);
  });
});

if (!hasWebGpu) {
  describe.skip('WebGPU SHA-1 benchmark', () => {
    it('skipped because WebGPU is not available in this environment', () => {
      expect(true).toBe(true);
    });
  });
}
