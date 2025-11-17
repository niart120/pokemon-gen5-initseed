import { describe, it, expect } from 'vitest';
import { baseParams, runGenerationScenario } from './integration-helpers';

const hasWorker = typeof Worker !== 'undefined';

describe('Generation Integration Scenarios (D1)', () => {
  it('Scenario 1: completes by max-advances', async () => {
  if (!hasWorker) { expect(true).toBe(true); return; }
    // maxResults は validation で maxAdvances 以下が必須
    const p = baseParams({ maxAdvances: 3000, maxResults: 3000, stopAtFirstShiny: false, stopOnCap: true });
    const { completion, totalResults } = await runGenerationScenario(p);
    expect(completion.reason).toBe('max-advances');
    expect(completion.processedAdvances).toBe(3000);
    expect(completion.resultsCount).toBeLessThanOrEqual(p.maxResults);
    expect(totalResults).toBe(completion.resultsCount);
  });

  it('Scenario 2: completes by max-results', async () => {
  if (!hasWorker) { expect(true).toBe(true); return; }
    // max-results を先に達成するように maxResults を小さく, maxAdvances は十分大きく
    const p = baseParams({ maxAdvances: 20000, maxResults: 150, stopOnCap: true, stopAtFirstShiny: false });
    const { completion } = await runGenerationScenario(p);
    expect(completion.reason).toBe('max-results');
    expect(completion.resultsCount).toBe(p.maxResults);
    expect(completion.processedAdvances).toBeGreaterThanOrEqual(completion.resultsCount);
  });

  it('Scenario 3: completes by first-shiny (or fallback)', async () => {
  if (!hasWorker) { expect(true).toBe(true); return; }
    const p = baseParams({ stopAtFirstShiny: true, maxAdvances: 20000, maxResults: 20000, stopOnCap: true });
    const { completion } = await runGenerationScenario(p, 12000);
    if (completion.shinyFound) {
      expect(completion.reason).toBe('first-shiny');
    } else {
      // 想定 fallback
      expect(['max-advances']).toContain(completion.reason);
    }
  });
});
