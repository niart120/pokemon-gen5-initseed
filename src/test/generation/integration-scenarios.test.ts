import { describe, it, expect } from 'vitest';
import { baseParams, runGenerationScenario } from './integration-helpers';

describe('Generation Integration Scenarios (D1)', () => {
  it('Scenario 1: completes by max-advances', async () => {
    const p = baseParams({ maxAdvances: 3000, maxResults: 4000, stopAtFirstShiny: false, stopOnCap: true });
    const { completion, progressSamples } = await runGenerationScenario(p);
    expect(completion.reason).toBe('max-advances');
    expect(completion.processedAdvances).toBe(3000);
    expect(completion.resultsCount).toBeLessThanOrEqual(p.maxResults);
    expect(progressSamples).toBeGreaterThan(0);
  });

  it('Scenario 2: completes by max-results', async () => {
    const p = baseParams({ maxAdvances: 100000, maxResults: 150, stopOnCap: true, stopAtFirstShiny: false, batchSize: 100 });
    const { completion } = await runGenerationScenario(p);
    expect(completion.reason).toBe('max-results');
    expect(completion.resultsCount).toBe(p.maxResults);
    expect(completion.processedAdvances).toBeGreaterThanOrEqual(completion.resultsCount);
  });

  it('Scenario 3: completes by first-shiny', async () => {
    const p = baseParams({ stopAtFirstShiny: true, maxAdvances: 20000, maxResults: 1000, stopOnCap: true });
    const { completion } = await runGenerationScenario(p, 12000);
    if (completion.shinyFound) {
      expect(completion.reason).toBe('first-shiny');
    } else {
      expect(['max-advances']).toContain(completion.reason);
    }
  });
});
