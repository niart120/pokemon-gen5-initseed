import { SHA1 } from '@/lib/core/sha1';

export const WORDS_PER_MESSAGE = 16;
export const WORDS_PER_HASH = 5;

export interface Sha1BenchmarkStats {
  averageMs: number;
  minMs: number;
  maxMs: number;
  standardDeviationMs: number;
  samples: number;
}

export interface Sha1BenchmarkOptions {
  runner: (messages: Uint32Array) => Promise<Uint32Array> | Uint32Array;
  messages: Uint32Array;
  iterations: number;
  warmupIterations?: number;
}

export interface Sha1BenchmarkOutcome extends Sha1BenchmarkStats {
  durations: number[];
}

export function createRandomMessages(messageCount: number, seed = 0x12345678): Uint32Array {
  const totalWords = messageCount * WORDS_PER_MESSAGE;
  const data = new Uint32Array(totalWords);
  let state = seed >>> 0;

  for (let i = 0; i < totalWords; i++) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    data[i] = state >>> 0;
  }

  return data;
}

export function runCpuSha1(messages: Uint32Array): Uint32Array {
  if (messages.length % WORDS_PER_MESSAGE !== 0) {
    throw new Error('Message buffer must be a multiple of 16 words');
  }

  const sha1 = new SHA1();
  const messageCount = messages.length / WORDS_PER_MESSAGE;
  const results = new Uint32Array(messageCount * WORDS_PER_HASH);
  const words: number[] = new Array(WORDS_PER_MESSAGE);

  for (let index = 0; index < messageCount; index++) {
    const base = index * WORDS_PER_MESSAGE;
    for (let offset = 0; offset < WORDS_PER_MESSAGE; offset++) {
      words[offset] = messages[base + offset]!;
    }

    const hash = sha1.calculateHash(words);
    const outBase = index * WORDS_PER_HASH;
    results[outBase] = hash.h0 >>> 0;
    results[outBase + 1] = hash.h1 >>> 0;
    results[outBase + 2] = hash.h2 >>> 0;
    results[outBase + 3] = hash.h3 >>> 0;
    results[outBase + 4] = hash.h4 >>> 0;
  }

  return results;
}

export function hashesEqual(expected: Uint32Array, actual: Uint32Array): boolean {
  if (expected.byteLength !== actual.byteLength) {
    return false;
  }

  for (let i = 0; i < expected.length; i++) {
    if (expected[i] !== actual[i]) {
      return false;
    }
  }

  return true;
}

export async function benchmarkSha1(options: Sha1BenchmarkOptions): Promise<Sha1BenchmarkOutcome> {
  const iterations = Math.max(1, options.iterations);
  const warmupIterations = Math.max(0, options.warmupIterations ?? 1);
  const durations: number[] = [];

  for (let i = 0; i < warmupIterations; i++) {
    await options.runner(options.messages);
  }

  for (let i = 0; i < iterations; i++) {
    const start = now();
    await options.runner(options.messages);
    durations.push(now() - start);
  }

  return { ...calculateStats(durations), durations };
}

function now(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function calculateStats(samples: number[]): Sha1BenchmarkStats {
  if (samples.length === 0) {
    return { averageMs: 0, minMs: 0, maxMs: 0, standardDeviationMs: 0, samples: 0 };
  }

  let total = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (const sample of samples) {
    total += sample;
    if (sample < min) {
      min = sample;
    }
    if (sample > max) {
      max = sample;
    }
  }

  const mean = total / samples.length;
  let varianceSum = 0;

  for (const sample of samples) {
    const diff = sample - mean;
    varianceSum += diff * diff;
  }

  const variance = varianceSum / samples.length;
  return {
    averageMs: mean,
    minMs: min,
    maxMs: max,
    standardDeviationMs: Math.sqrt(variance),
    samples: samples.length,
  };
}
