import shaderTemplate from './sha1-generate.wgsl?raw';

export interface GeneratedPipelineSet {
  pipelines: {
    generate: GPUComputePipeline;
    scan: GPUComputePipeline;
    scatter: GPUComputePipeline;
  };
  layouts: {
    generate: GPUBindGroupLayout;
    scan: GPUBindGroupLayout;
    scatter: GPUBindGroupLayout;
  };
}

function buildShaderSource(workgroupSize: number): string {
  return shaderTemplate.replace(/WORKGROUP_SIZE_PLACEHOLDER/g, String(workgroupSize));
}

export function createGeneratedPipeline(device: GPUDevice, workgroupSize: number): GeneratedPipelineSet {
  const module = device.createShaderModule({
    label: 'gpu-seed-sha1-generated-module',
    code: buildShaderSource(workgroupSize),
  });
  module.getCompilationInfo?.().then((info) => {
    if (info.messages.length > 0) {
      console.warn('[pipeline-factory] compilation diagnostics', info.messages.map((msg) => ({
        message: msg.message,
        line: msg.lineNum,
        column: msg.linePos,
        type: msg.type,
      })));
    }
  }).catch((error) => {
    console.warn('[pipeline-factory] compilation info failed', error);
  });

  const generateLayout = device.createBindGroupLayout({
    label: 'gpu-seed-generate-bind-layout',
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
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  const scanLayout = device.createBindGroupLayout({
    label: 'gpu-seed-scan-bind-layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  const scatterLayout = device.createBindGroupLayout({
    label: 'gpu-seed-scatter-bind-layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });

  const generatePipeline = device.createComputePipeline({
    label: 'gpu-seed-generate-pipeline',
    layout: device.createPipelineLayout({
      label: 'gpu-seed-generate-pipeline-layout',
      bindGroupLayouts: [generateLayout],
    }),
    compute: {
      module,
      entryPoint: 'sha1_generate',
    },
  });

  const scanPipeline = device.createComputePipeline({
    label: 'gpu-seed-scan-pipeline',
    layout: device.createPipelineLayout({
      label: 'gpu-seed-scan-pipeline-layout',
      bindGroupLayouts: [scanLayout],
    }),
    compute: {
      module,
      entryPoint: 'exclusive_scan_groups',
    },
  });

  const scatterPipeline = device.createComputePipeline({
    label: 'gpu-seed-scatter-pipeline',
    layout: device.createPipelineLayout({
      label: 'gpu-seed-scatter-pipeline-layout',
      bindGroupLayouts: [scatterLayout],
    }),
    compute: {
      module,
      entryPoint: 'scatter_matches',
    },
  });

  return {
    pipelines: {
      generate: generatePipeline,
      scan: scanPipeline,
      scatter: scatterPipeline,
    },
    layouts: {
      generate: generateLayout,
      scan: scanLayout,
      scatter: scatterLayout,
    },
  };
}
