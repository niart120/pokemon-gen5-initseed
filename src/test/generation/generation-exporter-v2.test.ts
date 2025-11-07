import { describe, it, expect } from 'vitest';
import { exportGenerationResults, adaptGenerationResults } from '@/lib/export/generation-exporter';
import type { GenerationResult } from '@/types/generation';

const dummyEncounterTable = { slots: [ { speciesId: 1, levelRange: { min: 5, max: 5 } } ] } as any;

function make(i:number): GenerationResult {
  return { seed: BigInt(i+1), pid: 100+i, nature: i%25, sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 1234n, shiny_type: 0, advance: i };
}

describe('generation-exporter v2', () => {
  it('includes extended columns when context provided', () => {
    const data = [make(0)];
    const adapted = adaptGenerationResults(data, { encounterTable: dummyEncounterTable, genderRatios: new Map([[1,{ threshold: 128, genderless: false }]]) });
    expect(adapted[0].speciesName).toBeDefined();
    const csv = exportGenerationResults(data, { format:'csv' });
    expect(csv.split('\n')[0]).toContain('SpeciesName');
    const json = exportGenerationResults(data, { format:'json' });
    expect(json).toContain('generation-v2');
  });
});
