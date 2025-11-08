#!/usr/bin/env node
import { spawn } from 'node:child_process';

const batchLimits = [undefined, 16384, 32768, 49152, 65536, 81920];
const workgroupSizes = [undefined, 128, 192];
const combinations = [];

for (const batchLimit of batchLimits) {
  for (const workgroupSize of workgroupSizes) {
    combinations.push({ batchLimit, workgroupSize });
  }
}

function formatLabel(value) {
  return value === undefined ? 'default' : String(value);
}

function parseMetrics(output) {
  const lines = output.split(/\r?\n/);
  const metrics = { summary: null, profile: null };
  for (const line of lines) {
    if (line.includes('[sha1] ') && !line.includes('[sha1-gpu-profile]')) {
      metrics.summary = line.trim();
    } else if (line.includes('[sha1-gpu-profile]')) {
      metrics.profile = line.trim();
    }
  }
  return metrics;
}

async function runCombination(batchLimit, workgroupSize) {
  return new Promise((resolve) => {
    const env = { ...process.env };
    if (batchLimit === undefined) {
      delete env.VITE_SHA1_GPU_BATCH_LIMIT;
    } else {
      env.VITE_SHA1_GPU_BATCH_LIMIT = String(batchLimit);
    }
    if (workgroupSize === undefined) {
      delete env.VITE_SHA1_GPU_WORKGROUP_SIZE;
    } else {
      env.VITE_SHA1_GPU_WORKGROUP_SIZE = String(workgroupSize);
    }

    const args = ['vitest', 'run', '--config', 'vitest.browser.config.ts'];
    const child = spawn('npx', args, {
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: process.platform === 'win32',
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdout += text;
      process.stdout.write(text);
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(text);
    });

    child.on('close', (code) => {
      const metrics = parseMetrics(stdout);
      resolve({
        code,
        stdout,
        stderr,
        metrics,
      });
    });
  });
}

(async () => {
  const results = [];
  for (const combo of combinations) {
    const label = `batch=${formatLabel(combo.batchLimit)}, workgroup=${formatLabel(combo.workgroupSize)}`;
    console.info(`\n=== Running ${label} ===`);
    const result = await runCombination(combo.batchLimit, combo.workgroupSize);
    results.push({ label, ...result });
    if (result.code !== 0) {
      console.error(`Command failed for ${label}`);
      process.exitCode = result.code;
      break;
    }
  }

  console.info('\n=== Summary ===');
  for (const result of results) {
    const { label, metrics } = result;
    const summary = metrics.summary ?? 'summary missing';
    const profile = metrics.profile ?? 'profile missing';
    console.info(`- ${label}`);
    console.info(`  ${summary}`);
    console.info(`  ${profile}`);
  }
})();
