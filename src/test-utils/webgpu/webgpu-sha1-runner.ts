/// <reference types="@webgpu/types" />

import { WORDS_PER_HASH, WORDS_PER_MESSAGE, type GpuSha1WorkloadConfig } from '@/test-utils/perf/sha1-webgpu-harness';

const DEFAULT_WORKGROUP_SIZE = 64;
const BYTES_PER_WORD = Uint32Array.BYTES_PER_ELEMENT;
const BYTES_PER_MESSAGE = WORDS_PER_MESSAGE * BYTES_PER_WORD;
const BYTES_PER_HASH = WORDS_PER_HASH * BYTES_PER_WORD;

function buildShaderSource(workgroupSize: number): string {
  return /* wgsl */ `
struct Config {
  message_count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;
@group(0) @binding(2) var<uniform> config : Config;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
}

@compute @workgroup_size(${workgroupSize})
fn sha1_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index >= config.message_count) {
    return;
  }

  let input_base = index * 16u;
  var w : array<u32, 16>;

  for (var i = 0u; i < 16u; i = i + 1u) {
    w[i] = input_words[input_base + i];
  }

  var a : u32 = 0x67452301u;
  var b : u32 = 0xEFCDAB89u;
  var c : u32 = 0x98BADCFEu;
  var d : u32 = 0x10325476u;
  var e : u32 = 0xC3D2E1F0u;

  var i : u32 = 0u;
  for (; i < 20u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + ((b & c) | ((~b) & d)) + e + 0x5A827999u + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  for (; i < 40u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0x6ED9EBA1u + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  for (; i < 60u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  for (; i < 80u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0xCA62C1D6u + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  let h0 = 0x67452301u + a;
  let h1 = 0xEFCDAB89u + b;
  let h2 = 0x98BADCFEu + c;
  let h3 = 0x10325476u + d;
  let h4 = 0xC3D2E1F0u + e;

  let output_base = index * 5u;
  output_words[output_base] = h0;
  output_words[output_base + 1u] = h1;
  output_words[output_base + 2u] = h2;
  output_words[output_base + 3u] = h3;
  output_words[output_base + 4u] = h4;
}
`;
}

function buildGeneratedShaderSource(workgroupSize: number): string {
  return /* wgsl */ `
struct GeneratedConfig {
  message_count : u32,
  base_offset : u32,
  range_seconds : u32,
  timer0_min : u32,
  timer0_count : u32,
  vcount_min : u32,
  vcount_count : u32,
  start_second_of_day : u32,
  start_day_of_week : u32,
  mac_lower : u32,
  data7_swapped : u32,
  key_input_swapped : u32,
  hardware_type : u32,
  nazo0 : u32,
  nazo1 : u32,
  nazo2 : u32,
  nazo3 : u32,
  nazo4 : u32,
  start_year : u32,
  start_day_of_year : u32,
};

const MONTH_LENGTHS_COMMON : array<u32, 12> = array<u32, 12>(
  31u, 28u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
);
const MONTH_LENGTHS_LEAP : array<u32, 12> = array<u32, 12>(
  31u, 29u, 31u, 30u, 31u, 30u, 31u, 31u, 30u, 31u, 30u, 31u
);

@group(0) @binding(0) var<storage, read> config : GeneratedConfig;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

fn left_rotate(value : u32, amount : u32) -> u32 {
  return (value << amount) | (value >> (32u - amount));
}

fn swap32(value : u32) -> u32 {
  return ((value & 0x000000FFu) << 24u) |
    ((value & 0x0000FF00u) << 8u) |
    ((value & 0x00FF0000u) >> 8u) |
    ((value & 0xFF000000u) >> 24u);
}

fn to_bcd(value : u32) -> u32 {
  let tens = value / 10u;
  let ones = value - tens * 10u;
  return (tens << 4u) | ones;
}

fn is_leap_year(year : u32) -> bool {
  return (year % 4u == 0u && year % 100u != 0u) || (year % 400u == 0u);
}

fn month_day_from_day_of_year(day_of_year : u32, leap : bool) -> vec2<u32> {
  var remaining = day_of_year;
  var month = 1u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let length = select(MONTH_LENGTHS_COMMON[i], MONTH_LENGTHS_LEAP[i], leap);
    if (remaining <= length) {
      return vec2<u32>(month, remaining);
    }
    remaining = remaining - length;
    month = month + 1u;
  }
  return vec2<u32>(12u, 31u);
}

@compute @workgroup_size(${workgroupSize})
fn sha1_generate(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let local_index = global_id.x;
  if (local_index >= config.message_count) {
    return;
  }

  let safe_range_seconds = max(config.range_seconds, 1u);
  let safe_vcount_count = max(config.vcount_count, 1u);
  let messages_per_vcount = safe_range_seconds;
  let messages_per_timer0 = messages_per_vcount * safe_vcount_count;

  let message_index = config.base_offset + local_index;
  let timer0_index = message_index / messages_per_timer0;
  let remainder_after_timer0 = message_index - timer0_index * messages_per_timer0;
  let vcount_index = remainder_after_timer0 / messages_per_vcount;
  let second_offset = remainder_after_timer0 - vcount_index * messages_per_vcount;

  let timer0 = config.timer0_min + timer0_index;
  let vcount = config.vcount_min + vcount_index;

  let total_seconds = config.start_second_of_day + second_offset;
  let day_offset = total_seconds / 86400u;
  let seconds_of_day = total_seconds - day_offset * 86400u;

  let hour = seconds_of_day / 3600u;
  let minute = (seconds_of_day % 3600u) / 60u;
  let second = seconds_of_day % 60u;

  var year = config.start_year;
  var day_of_year = config.start_day_of_year + day_offset;
  loop {
    let year_length = select(365u, 366u, is_leap_year(year));
    if (day_of_year <= year_length) {
      break;
    }
    day_of_year = day_of_year - year_length;
    year = year + 1u;
  }

  let leap = is_leap_year(year);
  let month_day = month_day_from_day_of_year(day_of_year, leap);
  let month = month_day.x;
  let day = month_day.y;

  let day_of_week = (config.start_day_of_week + day_offset) % 7u;
  let year_mod = year % 100u;
  let date_word = (to_bcd(year_mod) << 24u) | (to_bcd(month) << 16u) | (to_bcd(day) << 8u) | to_bcd(day_of_week);
  let is_pm = (config.hardware_type <= 1u) && (hour >= 12u);
  let pm_flag = select(0u, 1u, is_pm);
  let time_word = (pm_flag << 30u) | (to_bcd(hour) << 24u) | (to_bcd(minute) << 16u) | (to_bcd(second) << 8u);

  var w : array<u32, 16>;
  w[0] = config.nazo0;
  w[1] = config.nazo1;
  w[2] = config.nazo2;
  w[3] = config.nazo3;
  w[4] = config.nazo4;
  w[5] = swap32((vcount << 16u) | timer0);
  w[6] = config.mac_lower;
  w[7] = config.data7_swapped;
  w[8] = date_word;
  w[9] = time_word;
  w[10] = 0u;
  w[11] = 0u;
  w[12] = config.key_input_swapped;
  w[13] = 0x80000000u;
  w[14] = 0u;
  w[15] = 0x000001A0u;

  var a : u32 = 0x67452301u;
  var b : u32 = 0xEFCDAB89u;
  var c : u32 = 0x98BADCFEu;
  var d : u32 = 0x10325476u;
  var e : u32 = 0xC3D2E1F0u;

  var i : u32 = 0u;
  for (; i < 20u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + ((b & c) | ((~b) & d)) + e + 0x5A827999u + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  for (; i < 40u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0x6ED9EBA1u + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  for (; i < 60u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  for (; i < 80u; i = i + 1u) {
    let w_index = i & 15u;
    var w_value : u32;
    if (i < 16u) {
      w_value = w[w_index];
    } else {
      let expanded = w[(i - 3u) & 15u] ^ w[(i - 8u) & 15u] ^ w[(i - 14u) & 15u] ^ w[w_index];
      let rotated = left_rotate(expanded, 1u);
      w[w_index] = rotated;
      w_value = rotated;
    }

    let temp = left_rotate(a, 5u) + (b ^ c ^ d) + e + 0xCA62C1D6u + w_value;
    e = d;
    d = c;
    c = left_rotate(b, 30u);
    b = a;
    a = temp;
  }

  let h0 = 0x67452301u + a;
  let h1 = 0xEFCDAB89u + b;
  let h2 = 0x98BADCFEu + c;
  let h3 = 0x10325476u + d;
  let h4 = 0xC3D2E1F0u + e;

  let output_base = local_index * 5u;
  output_words[output_base] = h0;
  output_words[output_base + 1u] = h1;
  output_words[output_base + 2u] = h2;
  output_words[output_base + 3u] = h3;
  output_words[output_base + 4u] = h4;
}
`;
}

export interface WebGpuSha1ProfilingSample {
  totalMs: number;
  uploadMs: number;
  dispatchMs: number;
  readbackMs: number;
  batches: number;
  totalMessages: number;
  maxBatchMessages: number;
  details?: WebGpuSha1BatchDetail[];
}

export interface WebGpuSha1BatchDetail {
  slot: number;
  messageCount: number;
  uploadMs: number;
  dispatchMs: number;
  readbackMs: number;
  totalMs: number;
  submitToGpuDoneMs: number | null;
  gpuDoneToMapMs: number | null;
  queueSubmitMs: number;
  gpuIdleBeforeMs: number | null;
}

type WebGpuSha1ProfilingAccumulator = WebGpuSha1ProfilingSample & { details: WebGpuSha1BatchDetail[] };

interface BufferSet {
  input: GPUBuffer | null;
  inputSize: number;
  output: GPUBuffer | null;
  outputSize: number;
  readback: GPUBuffer | null;
  readbackSize: number;
  bindGroup: GPUBindGroup | null;
}

interface GeneratedBufferSet {
  output: GPUBuffer | null;
  outputSize: number;
  readback: GPUBuffer | null;
  readbackSize: number;
  bindGroup: GPUBindGroup | null;
}

export class WebGpuSha1Runner {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private configBuffer: GPUBuffer | null = null;
  private configData: Uint32Array | null = null;
  private generatedPipeline: GPUComputePipeline | null = null;
  private generatedBindGroupLayout: GPUBindGroupLayout | null = null;
  private generatedConfigBuffer: GPUBuffer | null = null;
  private generatedConfigData: Uint32Array | null = null;
  private generatedBufferSets: GeneratedBufferSet[] = [];
  private generatedConfigSignature: string | null = null;
  private generatedLastUploadedMessageCount: number | null = null;
  private generatedLastUploadedBaseOffset: number | null = null;
  private maxMessagesPerDispatch: number | null = null;
  private maxMessagesOverride: number | null = null;
  private lastProfiling: WebGpuSha1ProfilingSample | null = null;
  private bufferSets: BufferSet[] = [];
  private readonly bufferSetCount = 2;
  private workgroupSize = DEFAULT_WORKGROUP_SIZE;
  private lastGpuDoneAt: number | null = null;
  private lastUploadedConfigCount: number | null = null;

  public async init(): Promise<void> {
    if (this.device) {
      return;
    }

    const gpu = typeof navigator !== 'undefined' ? navigator.gpu : undefined;
    if (!gpu) {
      throw new Error('WebGPU is not available in this environment');
    }

    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error('Failed to acquire WebGPU adapter');
    }

    const device = await adapter.requestDevice();

    this.device = device;
    this.configBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.configData = new Uint32Array(4);
    const generatedConfigArray = new Uint32Array(24);
    this.generatedConfigData = generatedConfigArray;
    this.generatedConfigBuffer = device.createBuffer({
      size: this.alignSize(generatedConfigArray.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.initializeBufferSets();
    this.initializeGeneratedBufferSets();
    await this.recreatePipelines();
  }

  public async compute(messages: Uint32Array): Promise<Uint32Array> {
    if (messages.length === 0) {
      this.lastProfiling = null;
      return new Uint32Array(0);
    }

    await this.init();

    const totalMessages = this.validateMessageBuffer(messages);
    const maxBatchSize = this.getMaxMessagesPerDispatch();
    const results = new Uint32Array(totalMessages * WORDS_PER_HASH);
    const profiling: WebGpuSha1ProfilingAccumulator = {
      totalMs: 0,
      uploadMs: 0,
      dispatchMs: 0,
      readbackMs: 0,
      batches: 0,
      totalMessages: 0,
      maxBatchMessages: 0,
      details: [],
    };

    this.lastGpuDoneAt = null;
    this.lastUploadedConfigCount = null;

    const inFlight: Promise<void>[] = Array.from({ length: this.bufferSetCount }, () => Promise.resolve());
    let processedMessages = 0;
    let batchIndex = 0;
    while (processedMessages < totalMessages) {
      const remaining = totalMessages - processedMessages;
      const batchSize = Math.min(maxBatchSize, remaining);
      const startWord = processedMessages * WORDS_PER_MESSAGE;
      const endWord = startWord + batchSize * WORDS_PER_MESSAGE;
      const batchView = messages.subarray(startWord, endWord);

      const slot = batchIndex % this.bufferSetCount;
      await inFlight[slot];
      const bufferSet = this.ensureBufferSet(slot, batchSize);
      const outStart = processedMessages * WORDS_PER_HASH;
      inFlight[slot] = this.scheduleBatch(batchView, batchSize, results, outStart, profiling, slot, bufferSet);
      processedMessages += batchSize;
      batchIndex += 1;
    }

    await Promise.all(inFlight);
    profiling.totalMessages = processedMessages;
    this.lastProfiling = {
      ...profiling,
      details: profiling.details.map((entry) => ({ ...entry })),
    };
    return results;
  }

  public async computeGenerated(
    workload: GpuSha1WorkloadConfig,
    baseOffset: number,
    messageCount: number
  ): Promise<Uint32Array> {
    if (!Number.isFinite(baseOffset) || baseOffset < 0 || !Number.isInteger(baseOffset)) {
      throw new Error('baseOffset must be a non-negative integer');
    }
    if (!Number.isFinite(messageCount) || messageCount < 0 || !Number.isInteger(messageCount)) {
      throw new Error('messageCount must be a non-negative integer');
    }

    if (messageCount === 0) {
      this.lastProfiling = null;
      return new Uint32Array(0);
    }

    await this.init();

    if (baseOffset + messageCount > workload.totalMessages) {
      throw new Error('Requested range exceeds workload total messages');
    }

    this.configureGeneratedWorkload(workload);

    const totalResults = new Uint32Array(messageCount * WORDS_PER_HASH);
    const maxBatchSize = this.getMaxMessagesPerDispatch();
    const profiling: WebGpuSha1ProfilingAccumulator = {
      totalMs: 0,
      uploadMs: 0,
      dispatchMs: 0,
      readbackMs: 0,
      batches: 0,
      totalMessages: 0,
      maxBatchMessages: 0,
      details: [],
    };

    this.lastGpuDoneAt = null;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;

    const inFlight: Promise<void>[] = Array.from({ length: this.bufferSetCount }, () => Promise.resolve());
    let processedMessages = 0;
    let batchIndex = 0;

    while (processedMessages < messageCount) {
      const remaining = messageCount - processedMessages;
      const batchSize = Math.min(maxBatchSize, remaining);
      const slot = batchIndex % this.bufferSetCount;
      await inFlight[slot];
      const bufferSet = this.ensureGeneratedBufferSet(slot, batchSize);
      const outStart = processedMessages * WORDS_PER_HASH;
      const batchBaseOffset = baseOffset + processedMessages;
      inFlight[slot] = this.scheduleGeneratedBatch(
        batchSize,
        batchBaseOffset,
        totalResults,
        outStart,
        profiling,
        slot,
        bufferSet
      );
      processedMessages += batchSize;
      batchIndex += 1;
    }

    await Promise.all(inFlight);
    profiling.totalMessages = messageCount;
    this.lastProfiling = {
      ...profiling,
      details: profiling.details.map((entry) => ({ ...entry })),
    };
    return totalResults;
  }

  public getLastProfiling(): WebGpuSha1ProfilingSample | null {
    if (!this.lastProfiling) {
      return null;
    }
    const { details, ...rest } = this.lastProfiling;
    return {
      ...rest,
      details: details?.map((entry) => ({ ...entry })),
    };
  }

  public setMaxMessagesPerDispatch(limit: number | null | undefined): void {
    if (limit == null) {
      this.maxMessagesOverride = null;
      this.maxMessagesPerDispatch = null;
      return;
    }

    if (!Number.isFinite(limit) || limit <= 0) {
      throw new Error('maxMessagesPerDispatch override must be a positive finite number');
    }

    this.maxMessagesOverride = Math.max(1, Math.floor(limit));
    this.maxMessagesPerDispatch = null;
  }

  public async setWorkgroupSize(size: number): Promise<void> {
    if (!Number.isFinite(size) || size <= 0 || !Number.isInteger(size)) {
      throw new Error('workgroup size must be a positive integer');
    }

    if (this.workgroupSize === size) {
      return;
    }

    this.workgroupSize = size;
    this.maxMessagesPerDispatch = null;

    if (this.device) {
      await this.recreatePipelines();
    }
  }

  public dispose(): void {
    this.configBuffer?.destroy();
    this.configBuffer = null;
    this.configData = null;
    this.generatedConfigBuffer?.destroy();
    this.generatedConfigBuffer = null;
    this.generatedConfigData = null;
    this.releaseBufferSets();
    this.releaseGeneratedBufferSets();
    this.bufferSets = [];
    this.generatedBufferSets = [];
    this.pipeline = null;
    this.generatedPipeline = null;
    this.bindGroupLayout = null;
    this.generatedBindGroupLayout = null;
    this.device = null;
    this.lastUploadedConfigCount = null;
    this.generatedConfigSignature = null;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;
  }

  private getMaxMessagesPerDispatch(): number {
    if (this.maxMessagesOverride !== null) {
      return this.maxMessagesOverride;
    }

    if (this.maxMessagesPerDispatch !== null) {
      return this.maxMessagesPerDispatch;
    }

    const device = this.device!;
    const storageLimit = device.limits.maxStorageBufferBindingSize || BYTES_PER_MESSAGE;
    const maxByInput = Math.max(1, Math.floor(storageLimit / BYTES_PER_MESSAGE));
    const maxByOutput = Math.max(1, Math.floor(storageLimit / BYTES_PER_HASH));

    const supportedWorkgroupSize = this.resolveWorkgroupSizeLimit();
    const maxWorkgroups = Math.max(1, device.limits.maxComputeWorkgroupsPerDimension ?? 65535);
    const workgroupLimit = supportedWorkgroupSize * maxWorkgroups;
    const rawMax = Math.max(1, Math.min(maxByInput, maxByOutput, workgroupLimit));

    // 少し余裕を持たせて境界超過を防ぐ
    const safeMax = rawMax > 1 ? rawMax - 1 : rawMax;
    this.maxMessagesPerDispatch = safeMax;
    return safeMax;
  }

  private resolveWorkgroupSizeLimit(): number {
    const device = this.device!;
    const invocationLimit = device.limits.maxComputeInvocationsPerWorkgroup ?? this.workgroupSize;
    const axisLimit = device.limits.maxComputeWorkgroupSizeX ?? this.workgroupSize;
    if (this.workgroupSize > invocationLimit || this.workgroupSize > axisLimit) {
      throw new Error(
        `workgroup size ${this.workgroupSize} exceeds device limits (max invocations ${invocationLimit}, max x-dimension ${axisLimit}).`
      );
    }
    return this.workgroupSize;
  }

  private scheduleBatch(
    messages: Uint32Array,
    messageCount: number,
    results: Uint32Array,
    resultOffset: number,
    profiling: WebGpuSha1ProfilingAccumulator,
    slot: number,
    bufferSet: BufferSet
  ): Promise<void> {
    const device = this.device!;

    const inputSize = messages.byteLength;
    const outputSize = messageCount * BYTES_PER_HASH;
    const batchStart = performance.now();

    device.pushErrorScope?.('validation');

    device.queue.writeBuffer(bufferSet.input!, 0, messages.buffer, messages.byteOffset, inputSize);
    this.updateDispatchConfig(messageCount);

    const afterUpload = performance.now();

    const bindGroup = this.getOrCreateBindGroup(slot, bufferSet);

    return this.completeBatch({
      pipeline: this.pipeline!,
      bindGroup,
      outputBuffer: bufferSet.output!,
      readbackBuffer: bufferSet.readback!,
      outputSize,
      messageCount,
      results,
      resultOffset,
      profiling,
      slot,
      batchStart,
      afterUpload,
      onValidationError: () => this.releaseBufferSets(),
    });
  }

  private configureGeneratedWorkload(workload: GpuSha1WorkloadConfig): void {
    const signatureParts = [
      workload.startSecondsSince2000,
      workload.rangeSeconds,
      workload.timer0Min,
      workload.timer0Count,
      workload.vcountMin,
      workload.vcountCount,
      workload.macLower,
      workload.data7Swapped,
      workload.keyInputSwapped,
      workload.hardwareType,
      workload.startYear,
      workload.startDayOfYear,
      workload.startSecondOfDay,
      workload.startDayOfWeek,
      workload.nazoSwapped[0],
      workload.nazoSwapped[1],
      workload.nazoSwapped[2],
      workload.nazoSwapped[3],
      workload.nazoSwapped[4],
    ];
    const signature = signatureParts.join('|');
    if (signature === this.generatedConfigSignature) {
      return;
    }

    const data = this.generatedConfigData ?? (this.generatedConfigData = new Uint32Array(24));
    data[0] = 0;
    data[1] = 0;
    data[2] = workload.rangeSeconds >>> 0;
    data[3] = workload.timer0Min >>> 0;
    data[4] = workload.timer0Count >>> 0;
    data[5] = workload.vcountMin >>> 0;
    data[6] = workload.vcountCount >>> 0;
    data[7] = workload.startSecondOfDay >>> 0;
    data[8] = workload.startDayOfWeek >>> 0;
    data[9] = workload.macLower >>> 0;
    data[10] = workload.data7Swapped >>> 0;
    data[11] = workload.keyInputSwapped >>> 0;
    data[12] = workload.hardwareType >>> 0;
    data[13] = workload.nazoSwapped[0] >>> 0;
    data[14] = workload.nazoSwapped[1] >>> 0;
    data[15] = workload.nazoSwapped[2] >>> 0;
    data[16] = workload.nazoSwapped[3] >>> 0;
    data[17] = workload.nazoSwapped[4] >>> 0;
    data[18] = workload.startYear >>> 0;
    data[19] = workload.startDayOfYear >>> 0;
    this.generatedConfigSignature = signature;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;
    if (this.device && this.generatedConfigBuffer) {
      this.device.queue.writeBuffer(
        this.generatedConfigBuffer,
        0,
        data.buffer,
        data.byteOffset,
        data.byteLength
      );
    }
  }

  private scheduleGeneratedBatch(
    messageCount: number,
    baseOffset: number,
    results: Uint32Array,
    resultOffset: number,
    profiling: WebGpuSha1ProfilingAccumulator,
    slot: number,
    bufferSet: GeneratedBufferSet
  ): Promise<void> {
    const outputSize = messageCount * BYTES_PER_HASH;
    const batchStart = performance.now();

    this.device!.pushErrorScope?.('validation');

    this.updateGeneratedDispatchConfig(messageCount, baseOffset);

    const afterUpload = performance.now();

    const bindGroup = this.getOrCreateGeneratedBindGroup(slot, bufferSet);

    return this.completeBatch({
      pipeline: this.generatedPipeline!,
      bindGroup,
      outputBuffer: bufferSet.output!,
      readbackBuffer: bufferSet.readback!,
      outputSize,
      messageCount,
      results,
      resultOffset,
      profiling,
      slot,
      batchStart,
      afterUpload,
      onValidationError: () => this.releaseGeneratedBufferSets(),
    });
  }

  private updateDispatchConfig(messageCount: number): void {
    if (this.lastUploadedConfigCount === messageCount) {
      return;
    }
    const device = this.device!;
    const configData = this.configData ?? (this.configData = new Uint32Array(4));
    configData[0] = messageCount >>> 0;
    configData[1] = 0;
    configData[2] = 0;
    configData[3] = 0;
    device.queue.writeBuffer(this.configBuffer!, 0, configData.buffer, configData.byteOffset, configData.byteLength);
    this.lastUploadedConfigCount = messageCount;
  }

  private updateGeneratedDispatchConfig(messageCount: number, baseOffset: number): void {
    if (
      this.generatedLastUploadedMessageCount === messageCount &&
      this.generatedLastUploadedBaseOffset === baseOffset
    ) {
      return;
    }
    const device = this.device!;
    const data = this.generatedConfigData ?? (this.generatedConfigData = new Uint32Array(24));
    data[0] = messageCount >>> 0;
    data[1] = baseOffset >>> 0;
    device.queue.writeBuffer(this.generatedConfigBuffer!, 0, data.buffer, data.byteOffset, data.byteLength);
    this.generatedLastUploadedMessageCount = messageCount;
    this.generatedLastUploadedBaseOffset = baseOffset;
  }

  private completeBatch(params: {
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    outputBuffer: GPUBuffer;
    readbackBuffer: GPUBuffer;
    outputSize: number;
    messageCount: number;
    results: Uint32Array;
    resultOffset: number;
    profiling: WebGpuSha1ProfilingAccumulator;
    slot: number;
    batchStart: number;
    afterUpload: number;
    onValidationError: () => void;
  }): Promise<void> {
    const device = this.device!;
    const {
      pipeline,
      bindGroup,
      outputBuffer,
      readbackBuffer,
      outputSize,
      messageCount,
      results,
      resultOffset,
      profiling,
      slot,
      batchStart,
      afterUpload,
      onValidationError,
    } = params;

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(messageCount / this.workgroupSize));
    pass.end();

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
    const commandBuffer = commandEncoder.finish();
    const previousGpuDoneAt = this.lastGpuDoneAt;
    device.queue.submit([commandBuffer]);
    const submitTime = performance.now();

    const validationPromise = device.popErrorScope?.() ?? Promise.resolve(null);
    const mapPromise = readbackBuffer.mapAsync(GPUMapMode.READ, 0, outputSize);
    const gpuDonePromise: Promise<number | null> =
      typeof device.queue.onSubmittedWorkDone === 'function'
        ? device.queue
            .onSubmittedWorkDone()
            .then(() => performance.now())
            .catch(() => null)
        : Promise.resolve(null);

    return (async () => {
      await mapPromise;
      const mapResolvedAt = performance.now();
      const gpuDoneAt = await gpuDonePromise;
      const validationError = await validationPromise;
      if (validationError) {
        onValidationError();
        throw validationError;
      }

      const mappedRange = readbackBuffer.getMappedRange(0, outputSize);
      const mappedView = new Uint32Array(mappedRange);
      results.set(mappedView, resultOffset);
      const afterReadback = performance.now();
      readbackBuffer.unmap();

      const uploadMs = afterUpload - batchStart;
      const dispatchMs = mapResolvedAt - afterUpload;
      const readbackMs = afterReadback - mapResolvedAt;
      const totalMs = afterReadback - batchStart;
      const submitToGpuDoneMs = gpuDoneAt === null ? null : Math.max(0, gpuDoneAt - afterUpload);
      const gpuDoneToMapMs = gpuDoneAt === null ? null : Math.max(0, mapResolvedAt - gpuDoneAt);
      const queueSubmitMs = submitTime - batchStart;
      const effectiveGpuDoneAt = gpuDoneAt ?? mapResolvedAt;
      const gpuIdleBeforeMs = previousGpuDoneAt === null ? null : Math.max(0, submitTime - previousGpuDoneAt);

      profiling.uploadMs += uploadMs;
      profiling.dispatchMs += dispatchMs;
      profiling.readbackMs += readbackMs;
      profiling.totalMs += totalMs;
      profiling.batches += 1;
      profiling.maxBatchMessages = Math.max(profiling.maxBatchMessages, messageCount);
      profiling.totalMessages += messageCount;
      profiling.details.push({
        slot,
        messageCount,
        uploadMs,
        dispatchMs,
        readbackMs,
        totalMs,
        submitToGpuDoneMs,
        gpuDoneToMapMs,
        queueSubmitMs,
        gpuIdleBeforeMs,
      });

      this.lastGpuDoneAt = effectiveGpuDoneAt;
    })();
  }

  private initializeBufferSets(): void {
    if (this.bufferSets.length === this.bufferSetCount) {
      return;
    }
    this.bufferSets = Array.from({ length: this.bufferSetCount }, () => this.createEmptyBufferSet());
  }

  private ensureBufferSet(slot: number, messageCount: number): BufferSet {
    const device = this.device;
    if (!device) {
      throw new Error('WebGPU device not initialized');
    }

    if (!this.bufferSets[slot]) {
      this.bufferSets[slot] = this.createEmptyBufferSet();
    }

    const set = this.bufferSets[slot];
    const requiredInputSize = this.alignSize(messageCount * BYTES_PER_MESSAGE);
    if (!set.input || set.inputSize < requiredInputSize) {
      set.input?.destroy();
      set.input = device.createBuffer({
        size: requiredInputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      set.inputSize = requiredInputSize;
      set.bindGroup = null;
    }

    const requiredOutputSize = this.alignSize(messageCount * BYTES_PER_HASH);
    if (!set.output || set.outputSize < requiredOutputSize) {
      set.output?.destroy();
      set.output = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      set.outputSize = requiredOutputSize;
      set.bindGroup = null;
    }

    if (!set.readback || set.readbackSize < requiredOutputSize) {
      set.readback?.destroy();
      set.readback = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      set.readbackSize = requiredOutputSize;
    }

    return set;
  }

  private releaseBufferSets(): void {
    for (const set of this.bufferSets) {
      this.clearBufferSet(set);
    }
  }

  private initializeGeneratedBufferSets(): void {
    if (this.generatedBufferSets.length === this.bufferSetCount) {
      return;
    }
    this.generatedBufferSets = Array.from({ length: this.bufferSetCount }, () => this.createEmptyGeneratedBufferSet());
  }

  private ensureGeneratedBufferSet(slot: number, messageCount: number): GeneratedBufferSet {
    const device = this.device;
    if (!device) {
      throw new Error('WebGPU device not initialized');
    }

    if (!this.generatedBufferSets[slot]) {
      this.generatedBufferSets[slot] = this.createEmptyGeneratedBufferSet();
    }

    const set = this.generatedBufferSets[slot];
    const requiredOutputSize = this.alignSize(messageCount * BYTES_PER_HASH);
    if (!set.output || set.outputSize < requiredOutputSize) {
      set.output?.destroy();
      set.output = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      set.outputSize = requiredOutputSize;
      set.bindGroup = null;
    }

    if (!set.readback || set.readbackSize < requiredOutputSize) {
      set.readback?.destroy();
      set.readback = device.createBuffer({
        size: requiredOutputSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      set.readbackSize = requiredOutputSize;
    }

    return set;
  }

  private releaseGeneratedBufferSets(): void {
    for (const set of this.generatedBufferSets) {
      this.clearGeneratedBufferSet(set);
    }
  }

  private alignSize(size: number): number {
    const alignment = 256;
    return Math.ceil(size / alignment) * alignment;
  }

  private getOrCreateGeneratedBindGroup(slot: number, set: GeneratedBufferSet): GPUBindGroup {
    if (set.bindGroup) {
      return set.bindGroup;
    }

    const bindGroup = this.device!.createBindGroup({
      layout: this.generatedBindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.generatedConfigBuffer! } },
        { binding: 1, resource: { buffer: set.output! } },
      ],
    });

    set.bindGroup = bindGroup;
    return bindGroup;
  }

  private getOrCreateBindGroup(slot: number, set: BufferSet): GPUBindGroup {
    if (set.bindGroup) {
      return set.bindGroup;
    }

    const bindGroup = this.device!.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: set.input! } },
        { binding: 1, resource: { buffer: set.output! } },
        { binding: 2, resource: { buffer: this.configBuffer! } },
      ],
    });

    set.bindGroup = bindGroup;
    return bindGroup;
  }

  private createEmptyGeneratedBufferSet(): GeneratedBufferSet {
    return {
      output: null,
      outputSize: 0,
      readback: null,
      readbackSize: 0,
      bindGroup: null,
    };
  }

  private createEmptyBufferSet(): BufferSet {
    return {
      input: null,
      inputSize: 0,
      output: null,
      outputSize: 0,
      readback: null,
      readbackSize: 0,
      bindGroup: null,
    };
  }

  private clearGeneratedBufferSet(set: GeneratedBufferSet | undefined): void {
    if (!set) {
      return;
    }

    set.output?.destroy();
    set.output = null;
    set.outputSize = 0;

    set.readback?.destroy();
    set.readback = null;
    set.readbackSize = 0;

    set.bindGroup = null;
  }

  private clearBufferSet(set: BufferSet | undefined): void {
    if (!set) {
      return;
    }

    set.input?.destroy();
    set.input = null;
    set.inputSize = 0;

    set.output?.destroy();
    set.output = null;
    set.outputSize = 0;

    set.readback?.destroy();
    set.readback = null;
    set.readbackSize = 0;

    set.bindGroup = null;
  }

  private async recreatePipelines(): Promise<void> {
    const device = this.device;
    if (!device) {
      return;
    }

    const module = device.createShaderModule({ code: buildShaderSource(this.workgroupSize) });
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'sha1_main' },
    });

    const generatedModule = device.createShaderModule({ code: buildGeneratedShaderSource(this.workgroupSize) });
    const generatedPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: generatedModule, entryPoint: 'sha1_generate' },
    });

    this.pipeline = pipeline;
    this.bindGroupLayout = pipeline.getBindGroupLayout(0);
    this.generatedPipeline = generatedPipeline;
    this.generatedBindGroupLayout = generatedPipeline.getBindGroupLayout(0);
    this.maxMessagesPerDispatch = null;
    this.invalidateBindGroups();
    this.invalidateGeneratedBindGroups();
    this.lastUploadedConfigCount = null;
    this.generatedLastUploadedMessageCount = null;
    this.generatedLastUploadedBaseOffset = null;
  }

  private invalidateBindGroups(): void {
    for (const set of this.bufferSets) {
      if (set) {
        set.bindGroup = null;
      }
    }
  }

  private invalidateGeneratedBindGroups(): void {
    for (const set of this.generatedBufferSets) {
      if (set) {
        set.bindGroup = null;
      }
    }
  }

  private validateMessageBuffer(messages: Uint32Array): number {
    if (messages.length % WORDS_PER_MESSAGE !== 0) {
      throw new Error('Message buffer must contain complete SHA-1 blocks (16 words each)');
    }
    return messages.length / WORDS_PER_MESSAGE;
  }
}
