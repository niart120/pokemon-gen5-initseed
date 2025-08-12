import { GenerationWorkerManager } from '@/lib/generation/generation-worker-manager';
import type { GenerationParams, GenerationCompletion } from '@/types/generation';

export interface RunGenerationResult {
  completion: GenerationCompletion;
  progressSamples: number;
}

export async function runGenerationScenario(params: GenerationParams, timeoutMs = 7000): Promise<RunGenerationResult> {
  const manager = new GenerationWorkerManager();
  let progressSamples = 0;
  return await new Promise<RunGenerationResult>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout')), timeoutMs);
    manager.onProgress(() => { progressSamples++; });
    manager.onComplete(c => { clearTimeout(timer); resolve({ completion: c, progressSamples }); });
    manager.onError(e => { clearTimeout(timer); reject(new Error('worker-error:'+e)); });
    manager.start(params).catch(err => { clearTimeout(timer); reject(err); });
  });
}

export function baseParams(overrides: Partial<GenerationParams>): GenerationParams {
  return {
    baseSeed: 0x12345678n,
    offset: 0n,
    maxAdvances: 5000,
    maxResults: 500,
    version: 'B',
    encounterType: 0,
    tid: 1,
    sid: 2,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
    batchSize: 500,
    ...overrides,
  };
}
