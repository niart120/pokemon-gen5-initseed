import { SeedCalculator } from '@/lib/core/seed-calculator';
import { initWasm } from '@/lib/core/wasm-interface';
import type { WasmModule } from '@/lib/core/wasm-interface';
import { toMacUint8Array } from '@/lib/utils/mac-address';
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import type { SearchConditions } from '@/types/search';

export const WORDS_PER_MESSAGE = 16;
export const WORDS_PER_HASH = 5;
const BYTES_PER_MESSAGE = WORDS_PER_MESSAGE * Uint32Array.BYTES_PER_ELEMENT;
const MAX_WASM_BATCH_BYTES = 512 * 1024;
const MAX_WASM_BATCH_MESSAGES = Math.max(1, Math.floor(MAX_WASM_BATCH_BYTES / BYTES_PER_MESSAGE));

const seedCalculator = new SeedCalculator();
const EPOCH_2000_MS = Date.UTC(2000, 0, 1, 0, 0, 0);

const HARDWARE_FRAME_VALUES: Record<Hardware, number> = {
  DS: 8,
  DS_LITE: 6,
  '3DS': 9,
};

export interface Sha1WorkloadConfig {
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress: [number, number, number, number, number, number];
  keyInput: number;
  startDate: Date;
  rangeSeconds: number;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
}

export interface Sha1WorkloadContext {
  config: Sha1WorkloadConfig;
  nazo: [number, number, number, number, number];
  frameValue: number;
  startSecondsSince2000: number;
  totalMessages: number;
  timer0Count: number;
  vcountCount: number;
}

export interface GpuSha1WorkloadConfig {
  startSecondsSince2000: number;
  rangeSeconds: number;
  timer0Min: number;
  timer0Max: number;
  timer0Count: number;
  vcountMin: number;
  vcountMax: number;
  vcountCount: number;
  totalMessages: number;
  hardwareType: number;
  macLower: number;
  data7Swapped: number;
  keyInputSwapped: number;
  nazoSwapped: Uint32Array;
  startYear: number;
  startDayOfYear: number;
  startSecondOfDay: number;
  startDayOfWeek: number;
}

export interface Sha1BenchmarkStats {
  averageMs: number;
  minMs: number;
  maxMs: number;
  standardDeviationMs: number;
  samples: number;
}

export interface Sha1BenchmarkOutcome extends Sha1BenchmarkStats {
  durations: number[];
}

export interface Sha1BenchmarkOptions {
  runner: (context: Sha1WorkloadContext) => Promise<void> | void;
  context: Sha1WorkloadContext;
  iterations: number;
  warmupIterations?: number;
}

const DEFAULT_WORKLOAD: Sha1WorkloadConfig = {
  romVersion: 'W2',
  romRegion: 'JPN',
  hardware: 'DS',
  macAddress: [0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e],
  keyInput: 0x0000,
  startDate: new Date(Date.UTC(2012, 0, 1, 0, 0, 0)),
  rangeSeconds: 4096,
  timer0Range: { min: 0x10a0, max: 0x10bf },
  vcountRange: { min: 0x60, max: 0x61 },
};

export function createWorkloadConfig(overrides: Partial<Sha1WorkloadConfig> = {}): Sha1WorkloadConfig {
  const timer0Range = overrides.timer0Range ?? DEFAULT_WORKLOAD.timer0Range;
  const vcountRange = overrides.vcountRange ?? DEFAULT_WORKLOAD.vcountRange;
  return {
    romVersion: overrides.romVersion ?? DEFAULT_WORKLOAD.romVersion,
    romRegion: overrides.romRegion ?? DEFAULT_WORKLOAD.romRegion,
    hardware: overrides.hardware ?? DEFAULT_WORKLOAD.hardware,
    macAddress: ([
      ...(overrides.macAddress ?? DEFAULT_WORKLOAD.macAddress),
    ] as [number, number, number, number, number, number]),
    keyInput: overrides.keyInput ?? DEFAULT_WORKLOAD.keyInput,
    startDate: overrides.startDate ? new Date(overrides.startDate) : new Date(DEFAULT_WORKLOAD.startDate),
    rangeSeconds: overrides.rangeSeconds ?? DEFAULT_WORKLOAD.rangeSeconds,
    timer0Range: { min: timer0Range.min, max: timer0Range.max },
    vcountRange: { min: vcountRange.min, max: vcountRange.max },
  };
}

export function createWorkloadContext(config: Sha1WorkloadConfig): Sha1WorkloadContext {
  const romParams = seedCalculator.getROMParameters(config.romVersion, config.romRegion);
  if (!romParams) {
    throw new Error(`ROM parameters not found for ${config.romVersion} ${config.romRegion}`);
  }

  const timer0Count = config.timer0Range.max - config.timer0Range.min + 1;
  const vcountCount = config.vcountRange.max - config.vcountRange.min + 1;
  const totalMessages = config.rangeSeconds * timer0Count * vcountCount;

  if (!Number.isSafeInteger(totalMessages) || totalMessages <= 0) {
    throw new Error('Invalid workload configuration: total messages exceeds safe limits');
  }

  const startSecondsSince2000 = Math.floor((config.startDate.getTime() - EPOCH_2000_MS) / 1000);
  if (startSecondsSince2000 < 0) {
    throw new Error('Workload start date must be on or after 2000-01-01T00:00:00Z');
  }

  const nazo = [...romParams.nazo] as [number, number, number, number, number];

  return {
    config,
    nazo,
    frameValue: HARDWARE_FRAME_VALUES[config.hardware],
    startSecondsSince2000,
    totalMessages,
    timer0Count,
    vcountCount,
  };
}

export function createWorkload(overrides: Partial<Sha1WorkloadConfig> = {}): Sha1WorkloadContext {
  return createWorkloadContext(createWorkloadConfig(overrides));
}

export function createWorkloadMessages(context: Sha1WorkloadContext): Uint32Array {
  const messages = new Uint32Array(context.totalMessages * WORDS_PER_MESSAGE);
  let writeIndex = 0;

  for (const message of iterateWorkloadMessages(context)) {
    messages.set(message, writeIndex);
    writeIndex += WORDS_PER_MESSAGE;
  }

  return messages;
}

type IntegratedSeedSearcherInstance = InstanceType<WasmModule['IntegratedSeedSearcher']>;

function computeMessagesPerSecond(context: Sha1WorkloadContext): number {
  return Math.max(1, context.timer0Count * context.vcountCount);
}

function resolveSecondsPerSlice(context: Sha1WorkloadContext, batchSize: number): number {
  const secondsFromBatch = Math.max(1, Math.floor(batchSize / computeMessagesPerSecond(context)));
  return Math.min(context.config.rangeSeconds, secondsFromBatch);
}

function runSearcherSlice(
  searcher: IntegratedSeedSearcherInstance,
  config: Sha1WorkloadConfig,
  sliceStart: Date,
  sliceSeconds: number,
  targetSeeds: Uint32Array
): void {
  searcher.search_seeds_integrated_simd(
    sliceStart.getUTCFullYear(),
    sliceStart.getUTCMonth() + 1,
    sliceStart.getUTCDate(),
    sliceStart.getUTCHours(),
    sliceStart.getUTCMinutes(),
    sliceStart.getUTCSeconds(),
    sliceSeconds,
    config.timer0Range.min,
    config.timer0Range.max,
    config.vcountRange.min,
    config.vcountRange.max,
    targetSeeds
  );
}

function swap32(value: number): number {
  return (
    ((value & 0xff) << 24) |
    (((value >>> 8) & 0xff) << 16) |
    (((value >>> 16) & 0xff) << 8) |
    ((value >>> 24) & 0xff)
  ) >>> 0;
}

export function buildGpuWorkloadConfig(context: Sha1WorkloadContext): GpuSha1WorkloadConfig {
  const { config } = context;
  const mac = config.macAddress;
  const macLower = ((mac[4] & 0xff) << 8) | (mac[5] & 0xff);
  const macUpper =
    (mac[0] & 0xff) |
    ((mac[1] & 0xff) << 8) |
    ((mac[2] & 0xff) << 16) |
    ((mac[3] & 0xff) << 24);
  const gxStat = 0x06000000;
  const data7 = (macUpper ^ gxStat ^ context.frameValue) >>> 0;
  const nazoSwapped = new Uint32Array(5);
  for (let i = 0; i < 5; i++) {
    nazoSwapped[i] = swap32(context.nazo[i] >>> 0);
  }
  const startYear = config.startDate.getFullYear();
  const startMonth = config.startDate.getMonth() + 1;
  const startDay = config.startDate.getDate();
  const startHour = config.startDate.getHours();
  const startMinute = config.startDate.getMinutes();
  const startSecond = config.startDate.getSeconds();
  const startSecondOfDay = startHour * 3600 + startMinute * 60 + startSecond;
  const startDayOfWeek = new Date(startYear, startMonth - 1, startDay).getDay();
  const startDayOfYear = calculateLocalDayOfYear(config.startDate);

  return {
    startSecondsSince2000: context.startSecondsSince2000 >>> 0,
    rangeSeconds: config.rangeSeconds >>> 0,
    timer0Min: config.timer0Range.min >>> 0,
    timer0Max: config.timer0Range.max >>> 0,
    timer0Count: context.timer0Count >>> 0,
    vcountMin: config.vcountRange.min >>> 0,
    vcountMax: config.vcountRange.max >>> 0,
    vcountCount: context.vcountCount >>> 0,
    totalMessages: context.totalMessages >>> 0,
    hardwareType: mapHardwareToId(config.hardware),
    macLower: macLower >>> 0,
    data7Swapped: swap32(data7),
    keyInputSwapped: swap32(config.keyInput >>> 0),
    nazoSwapped,
    startYear: startYear >>> 0,
    startDayOfYear: startDayOfYear >>> 0,
    startSecondOfDay: startSecondOfDay >>> 0,
    startDayOfWeek: startDayOfWeek >>> 0,
  };
}

function calculateLocalDayOfYear(date: Date): number {
  const startOfYear = new Date(date.getFullYear(), 0, 1);
  const diffMs = date.getTime() - startOfYear.getTime();
  const day = Math.floor(diffMs / (24 * 60 * 60 * 1000)) + 1;
  return Math.max(1, day);
}

function mapHardwareToId(hardware: Hardware): number {
  switch (hardware) {
    case 'DS':
      return 0;
    case 'DS_LITE':
      return 1;
    case '3DS':
      return 2;
    default:
      return 0;
  }
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

let wasmModulePromise: Promise<WasmModule> | null = null;

async function ensureWasmModule(): Promise<WasmModule> {
  if (!wasmModulePromise) {
    wasmModulePromise = initWasm();
  }
  return wasmModulePromise;
}

export async function runWasmHashes(messages: Uint32Array): Promise<Uint32Array> {
  if (messages.length % WORDS_PER_MESSAGE !== 0) {
    throw new Error('Message buffer must be a multiple of 16 words');
  }

  const module = await ensureWasmModule();
  const messageCount = messages.length / WORDS_PER_MESSAGE;
  const results = new Uint32Array(messageCount * WORDS_PER_HASH);

  let inOffset = 0;
  let outOffset = 0;

  while (inOffset < messages.length) {
    const remainingWords = messages.length - inOffset;
    const remainingMessages = remainingWords / WORDS_PER_MESSAGE;
    const batchMessages = Math.min(MAX_WASM_BATCH_MESSAGES, remainingMessages);
    const batchWords = batchMessages * WORDS_PER_MESSAGE;
    const batchView = messages.subarray(inOffset, inOffset + batchWords);

    const batchResult = module.sha1_hash_batch(batchView);
    results.set(batchResult, outOffset);

    inOffset += batchWords;
    outOffset += batchResult.length;
  }

  return results;
}

export async function runWasmHashesStreaming(
  context: Sha1WorkloadContext,
  batchSize: number
): Promise<void> {
  if (!Number.isFinite(batchSize) || batchSize <= 0) {
    throw new Error('batch size must be a positive finite number');
  }

  const module = await ensureWasmModule();
  const { config } = context;
  const secondsPerSlice = resolveSecondsPerSlice(context, batchSize);
  const nazo = new Uint32Array(context.nazo);
  const mac = toMacUint8Array(config.macAddress);
  const searcher = new module.IntegratedSeedSearcher(
    mac,
    nazo,
    config.hardware,
    config.keyInput,
    context.frameValue
  );
  const targetSeeds = new Uint32Array(0);

  try {
    const rangeSeconds = config.rangeSeconds;
    const startMs = config.startDate.getTime();

    for (let offsetSeconds = 0; offsetSeconds < rangeSeconds; offsetSeconds += secondsPerSlice) {
      const sliceSeconds = Math.min(secondsPerSlice, rangeSeconds - offsetSeconds);
      const sliceStart = new Date(startMs + offsetSeconds * 1000);
      runSearcherSlice(searcher, config, sliceStart, sliceSeconds, targetSeeds);
    }
  } finally {
    searcher.free();
  }
}

export async function benchmarkWorkload(options: Sha1BenchmarkOptions): Promise<Sha1BenchmarkOutcome> {
  const iterations = Math.max(1, options.iterations);
  const warmupIterations = Math.max(0, options.warmupIterations ?? 1);
  const durations: number[] = [];

  for (let i = 0; i < warmupIterations; i++) {
    await options.runner(options.context);
  }

  for (let i = 0; i < iterations; i++) {
    const start = now();
    await options.runner(options.context);
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

function* iterateWorkloadMessages(context: Sha1WorkloadContext): Generator<number[], void, undefined> {
  const { config } = context;
  const conditions = buildSearchConditions(config);
  const baseTimestamp = config.startDate.getTime();

  for (let timer0 = config.timer0Range.min; timer0 <= config.timer0Range.max; timer0++) {
    for (let vcount = config.vcountRange.min; vcount <= config.vcountRange.max; vcount++) {
      for (let secondOffset = 0; secondOffset < config.rangeSeconds; secondOffset++) {
        const datetime = new Date(baseTimestamp + secondOffset * 1000);
        yield seedCalculator.generateMessage(conditions, timer0, vcount, datetime);
      }
    }
  }
}

function buildSearchConditions(config: Sha1WorkloadConfig): SearchConditions {
  return {
    romVersion: config.romVersion,
    romRegion: config.romRegion,
    hardware: config.hardware,
    keyInput: config.keyInput,
    macAddress: [...config.macAddress],
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: { ...config.timer0Range },
      vcountRange: { ...config.vcountRange },
    },
    dateRange: {
      startYear: config.startDate.getUTCFullYear(),
      endYear: config.startDate.getUTCFullYear(),
      startMonth: config.startDate.getUTCMonth() + 1,
      endMonth: config.startDate.getUTCMonth() + 1,
      startDay: config.startDate.getUTCDate(),
      endDay: config.startDate.getUTCDate(),
      startHour: config.startDate.getUTCHours(),
      endHour: config.startDate.getUTCHours(),
      startMinute: config.startDate.getUTCMinutes(),
      endMinute: config.startDate.getUTCMinutes(),
      startSecond: config.startDate.getUTCSeconds(),
      endSecond: config.startDate.getUTCSeconds(),
    },
  };
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
