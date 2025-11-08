/// <reference types="@webgpu/types" />

import { WORDS_PER_HASH, WORDS_PER_MESSAGE } from '@/test-utils/perf/sha1-webgpu-harness';

const WORKGROUP_SIZE = 64;

const SHADER_SOURCE = /* wgsl */ `
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

@compute @workgroup_size(${WORKGROUP_SIZE})
fn sha1_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index >= config.message_count) {
    return;
  }

  let input_base = index * 16u;
  var w : array<u32, 80>;

  for (var i = 0u; i < 16u; i = i + 1u) {
    w[i] = input_words[input_base + i];
  }

  for (var i = 16u; i < 80u; i = i + 1u) {
    let value = w[i - 3u] ^ w[i - 8u] ^ w[i - 14u] ^ w[i - 16u];
    w[i] = left_rotate(value, 1u);
  }

  var a : u32 = 0x67452301u;
  var b : u32 = 0xEFCDAB89u;
  var c : u32 = 0x98BADCFEu;
  var d : u32 = 0x10325476u;
  var e : u32 = 0xC3D2E1F0u;

  for (var i = 0u; i < 80u; i = i + 1u) {
    var f : u32;
    var k : u32;

    if (i < 20u) {
      f = (b & c) | ((~b) & d);
      k = 0x5A827999u;
    } else if (i < 40u) {
      f = b ^ c ^ d;
      k = 0x6ED9EBA1u;
    } else if (i < 60u) {
      f = (b & c) | (b & d) | (c & d);
      k = 0x8F1BBCDCu;
    } else {
      f = b ^ c ^ d;
      k = 0xCA62C1D6u;
    }

    let temp = left_rotate(a, 5u) + f + e + k + w[i];
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

export class WebGpuSha1Runner {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private configBuffer: GPUBuffer | null = null;

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
    const module = device.createShaderModule({ code: SHADER_SOURCE });
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'sha1_main' },
    });

    this.device = device;
    this.pipeline = pipeline;
    this.bindGroupLayout = pipeline.getBindGroupLayout(0);
    this.configBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  public async compute(messages: Uint32Array): Promise<Uint32Array> {
    if (messages.length === 0) {
      return new Uint32Array(0);
    }

    await this.init();

    const device = this.device!;
    const messageCount = this.validateMessageBuffer(messages);
    const inputSize = messages.byteLength;
    const outputSize = messageCount * WORDS_PER_HASH * Uint32Array.BYTES_PER_ELEMENT;

    const inputBuffer = device.createBuffer({
      size: inputSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const outputBuffer = device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readbackBuffer = device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    device.queue.writeBuffer(inputBuffer, 0, messages.buffer, messages.byteOffset, inputSize);
    device.queue.writeBuffer(this.configBuffer!, 0, new Uint32Array([messageCount, 0, 0, 0]));

    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: this.configBuffer! } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(messageCount / WORKGROUP_SIZE));
    pass.end();

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
    device.queue.submit([commandEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const copy = readbackBuffer.getMappedRange().slice(0);
    readbackBuffer.unmap();

    inputBuffer.destroy();
    outputBuffer.destroy();
    readbackBuffer.destroy();

    return new Uint32Array(copy);
  }

  public dispose(): void {
    this.configBuffer?.destroy();
    this.configBuffer = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.device = null;
  }

  private validateMessageBuffer(messages: Uint32Array): number {
    if (messages.length % WORDS_PER_MESSAGE !== 0) {
      throw new Error('Message buffer must contain complete SHA-1 blocks (16 words each)');
    }
    return messages.length / WORDS_PER_MESSAGE;
  }
}
