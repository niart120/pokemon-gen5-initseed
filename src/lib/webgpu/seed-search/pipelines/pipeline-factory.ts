import shaderTemplate from './sha1-generate.wgsl?raw';

export interface GeneratedPipeline {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

function buildShaderSource(workgroupSize: number): string {
  return shaderTemplate.replace('@compute @workgroup_size(128)', `@compute @workgroup_size(${workgroupSize})`);
}

export function createGeneratedPipeline(device: GPUDevice, workgroupSize: number): GeneratedPipeline {
  const module = device.createShaderModule({
    label: 'gpu-seed-sha1-generated-module',
    code: buildShaderSource(workgroupSize),
  });
  module.getCompilationInfo?.().then((info) => {
    if (info.messages.length > 0) {
      console.debug('[pipeline-factory] compilation diagnostics', info.messages.map((msg) => ({
        message: msg.message,
        line: msg.lineNum,
        column: msg.linePos,
        type: msg.type,
      })));
    }
  }).catch((error) => {
    console.warn('[pipeline-factory] compilation info failed', error);
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'gpu-seed-bind-layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
        },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: 'gpu-seed-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: 'gpu-seed-generated-pipeline',
    layout: pipelineLayout,
    compute: {
      module,
      entryPoint: 'sha1_generate',
    },
  });

  return { pipeline, bindGroupLayout };
}
