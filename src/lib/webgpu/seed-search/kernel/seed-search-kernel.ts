import shaderTemplate from './sha1-generate.wgsl?raw';

export interface SeedSearchKernel {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
}

const WORKGROUP_SIZE_TOKEN = /WORKGROUP_SIZE_PLACEHOLDER/g;
const MODULE_LABEL = 'seed-search-kernel-module';
const PIPELINE_LABEL = 'seed-search-kernel';
const PIPELINE_LAYOUT_LABEL = 'seed-search-kernel-layout';
const BIND_GROUP_LAYOUT_LABEL = 'seed-search-kernel-bind-layout';

function buildShaderSource(workgroupSize: number): string {
  return shaderTemplate.replace(WORKGROUP_SIZE_TOKEN, String(workgroupSize));
}

export function createSeedSearchKernel(device: GPUDevice, workgroupSize: number): SeedSearchKernel {
  const module = device.createShaderModule({
    label: MODULE_LABEL,
    code: buildShaderSource(workgroupSize),
  });

  module
    .getCompilationInfo?.()
    .then((info) => {
      if (info.messages.length > 0) {
        console.warn('[seed-search-kernel] compilation diagnostics', info.messages.map((message) => ({
          message: message.message,
          line: message.lineNum,
          column: message.linePos,
          type: message.type,
        })));
      }
    })
    .catch((error) => {
      console.warn('[seed-search-kernel] compilation info failed', error);
    });

  const layout = device.createBindGroupLayout({
    label: BIND_GROUP_LAYOUT_LABEL,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: PIPELINE_LAYOUT_LABEL,
    bindGroupLayouts: [layout],
  });

  const pipeline = device.createComputePipeline({
    label: PIPELINE_LABEL,
    layout: pipelineLayout,
    compute: {
      module,
      entryPoint: 'sha1_generate',
    },
  });

  return {
    pipeline,
    layout,
  };
}