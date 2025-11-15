import { describe, it, expect } from 'vitest';
import { selectResolvedResults, selectUiReadyResults, createDefaultGenerationFilters } from '@/store/generation-store';
import type { GenerationSlice } from '@/store/generation-store';
import type { GenerationResult } from '@/types/generation';

// Minimal dummy encounter table
const dummyEncounterTable = {
  slots: [
    { speciesId: 1, levelRange: { min: 5, max: 5 } },
    { speciesId: 4, levelRange: { min: 7, max: 9 } },
  ],
};

function makeState(results: GenerationResult[]): Partial<GenerationSlice> & { results: GenerationResult[] } {
  return {
    params: null,
    draftParams: {},
    validationErrors: [],
    status: 'idle',
    progress: null,
    results,
    lastCompletion: null,
    error: null,
    filters: createDefaultGenerationFilters(),
    metrics: {},
    internalFlags: { receivedAnyBatch: false },
    encounterTable: dummyEncounterTable as any,
    genderRatios: new Map([[1,{ threshold: 128, genderless: false }], [4,{ threshold: 256, genderless: true }]]),
    abilityCatalog: undefined,
  };
}

describe('generation resolution selectors', () => {
  it('resolves species / level / gender', () => {
    const results: GenerationResult[] = [
      { seed: 1n, pid: 123, nature: 2, sync_applied: false, ability_slot: 0, gender_value: 10, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 1000n, shiny_type: 0, advance: 0 },
      { seed: 2n, pid: 456, nature: 5, sync_applied: true, ability_slot: 1, gender_value: 200, encounter_slot_value: 1, encounter_type: 0, level_rand_value: 2000n, shiny_type: 2, advance: 1 },
    ];
    const state = makeState(results) as any;
    const resolved = selectResolvedResults(state);
    expect(resolved.length).toBe(2);
    expect(resolved[0].speciesId).toBe(1);
    expect(resolved[1].speciesId).toBe(4);
    // gender: species 1 threshold 128 -> gender_value 10 => F
    expect(resolved[0].gender).toBe('F');
    // species 4 genderless => 'N'
    expect(resolved[1].gender).toBe('N');
    const ui = selectUiReadyResults(state,'en');
    expect(ui[0].speciesName).toBeDefined();
    expect(ui[0].gender).toBe('F');
  });
});
