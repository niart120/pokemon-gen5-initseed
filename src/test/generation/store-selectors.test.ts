import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/store/app-store';
import { selectShinyCount, selectThroughputEma, selectEtaFormatted } from '@/store/generation-store';

// Helper to reset store between tests
function resetStore() {
  useAppStore.setState({ results: [], progress: null, metrics: { shinyCount: 0 } as any });
}

describe('generation store selectors (B1)', () => {
  beforeEach(() => resetStore());

  it('shinyCount increments correctly on batch append simulation', () => {
    // Simulate internal batch handler by directly mutating via setState (public API not yet exposing custom append)
    useAppStore.setState((s) => ({
      results: s.results.concat([
        { advance: 1, shiny_type: 0, seed: 1n, pid: 1, nature: 0, sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 1n },
        { advance: 2, shiny_type: 2, seed: 2n, pid: 2, nature: 1, sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 2n },
      ]),
      metrics: { ...(s as any).metrics, shinyCount: 1 }
    }));
    useAppStore.setState((s) => ({
      results: s.results.concat([
        { advance: 3, shiny_type: 1, seed: 3n, pid: 3, nature: 2, sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 3n },
      ]),
      metrics: { ...(s as any).metrics, shinyCount: 2 }
    }));
    const shiny = selectShinyCount(useAppStore.getState());
    expect(shiny).toBe(2);
  });

  it('throughput EMA selector falls back correctly', () => {
    useAppStore.setState({ progress: { processedAdvances: 100, totalAdvances: 1000, resultsCount: 10, elapsedMs: 5000, throughput: 50, etaMs: 0, status: 'running' } as any });
    expect(selectThroughputEma(useAppStore.getState())).toBe(50);
    useAppStore.setState({ progress: { processedAdvances: 200, totalAdvances: 1000, resultsCount: 20, elapsedMs: 6000, throughput: 40, throughputRaw: 55, etaMs: 0, status: 'running' } as any });
    expect(selectThroughputEma(useAppStore.getState())).toBe(55);
    useAppStore.setState({ progress: { processedAdvances: 300, totalAdvances: 1000, resultsCount: 30, elapsedMs: 7000, throughput: 40, throughputRaw: 55, throughputEma: 52, etaMs: 0, status: 'running' } as any });
    expect(selectThroughputEma(useAppStore.getState())).toBe(52);
  });

  it('ETA formatted selector returns mm:ss for <1h', () => {
    useAppStore.setState({ progress: { processedAdvances: 100, totalAdvances: 160, resultsCount: 0, elapsedMs: 1000, throughput: 0, throughputEma: 10, etaMs: 0, status: 'running' } as any });
    const eta = selectEtaFormatted(useAppStore.getState());
    // remaining = 60, ema=10 -> 6s -> 00:06
    expect(eta).toBe('00:06');
  });
});
