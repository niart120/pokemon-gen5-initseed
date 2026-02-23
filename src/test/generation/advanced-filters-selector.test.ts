import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/store/app-store';
import { selectFilteredSortedResults, createDefaultGenerationFilters } from '@/store/generation-store';
import type { GenerationResult } from '@/types/generation';
import { resolveBatch } from '@/lib/generation/pokemon-resolver';
import type { ResolutionContext } from '@/types/pokemon-resolved';

// We simulate species/level/gender/ability resolution by injecting encounterTable + ratios + abilityCatalog
// For selector we only need resolved speciesId, level, gender, abilityIndex mapping.
// We'll craft results with distinct pid/seed and mock resolver context via store setters if needed later.

// Minimal stubs: since selectResolvedResults uses encounterTable mapping logic, simpler approach is to
// monkey patch encounterTable with predictable slot -> species mapping and ability slots by ability_slot field.

const makeResult = (advance:number, seed:bigint, pid:number, nature:number, shiny_type:number, ability_slot:number, gender_value:number, encounter_slot_value:number, level_rand_value:bigint): GenerationResult => ({
  advance, seed, pid, nature, shiny_type, ability_slot, gender_value, encounter_slot_value, encounter_type: 0, level_rand_value, sync_applied: false,
});

const results: GenerationResult[] = [
  // encounter_slot_value 0 -> species 495, 1 -> species 498
  makeResult(10, 10n, 0x10000001, 0, 0, 0, 0x00, 0, 0n), // slot0 species 495 female (0x00 < 128)
  makeResult(11, 11n, 0x10010002, 0, 0, 1, 0xFF, 1, 0n), // slot1 species 498 male   (0xFF >= 32)
  makeResult(12, 12n, 0x10020003, 0, 0, 2, 0x80, 0, 0n), // slot0 species 495 male hidden ability (slot 2)
];

// Fake encounter table with two slots mapping to species 495 & 498 and level range 10-12
const encounterTable: any = {
  slots: [
    { slotId: 0, speciesId: 495, levelRange: { min: 10, max: 10 } },
    { slotId: 1, speciesId: 498, levelRange: { min: 11, max: 12 } },
  ],
};

// Gender ratios map (resolver expects GenderRatio { threshold, genderless }):
// threshold: female if gender_value < threshold
// 50% female species uses threshold 128 (values 0..127 female). For clarity use 128 not 127.
// 12.5% female (87.5% male) species uses threshold 32 (0..31 female).
const genderRatios = new Map<number, any>([
  [495, { threshold: 128, genderless: false }], // 50% female threshold
  [498, { threshold: 32, genderless: false }],  // 12.5% female threshold
]);

// Ability catalog: species 495 has ability1+hidden, species 498 ability1+ability2
const abilityCatalog = new Map<number, string[]>([
  [495, ['A1', undefined as unknown as string, 'AH']],
  [498, ['B1', 'B2']],
]);

function resetStore() {
  const ctx: ResolutionContext = {
    encounterTable,
    genderRatios: genderRatios as any,
    abilityCatalog: abilityCatalog as any,
  };
  const resolved = resolveBatch(results as any, ctx);
  useAppStore.setState({
    results: results.map((entry) => ({ ...entry })),
    resolvedResults: resolved,
    filters: createDefaultGenerationFilters(),
  });
  useAppStore.getState().setEncounterTable(encounterTable);
  useAppStore.getState().setGenderRatios(genderRatios as any);
  useAppStore.getState().setAbilityCatalog(abilityCatalog as any);
}

describe('advanced resolved filters', () => {
  beforeEach(() => resetStore());

  it('filters by speciesIds includes only species 495 advances', () => {
    useAppStore.setState((s) => ({ filters: { ...s.filters, speciesIds: [495] } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    const adv = out.map(r=>r.advance).sort();
    expect(adv).toEqual([10,12]);
  });

  it('filters by speciesIds + abilityIndices (hidden ability only)', () => {
    useAppStore.setState((s) => ({ filters: { ...s.filters, speciesIds: [495], abilityIndices: [2] } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(out.map(r=>r.advance)).toEqual([12]);
  });

  it('filters by speciesIds + genders (male only)', () => {
    useAppStore.setState((s) => ({ filters: { ...s.filters, speciesIds: [495, 498], genders: ['M'] } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    const adv = out.map(r=>r.advance).sort();
    expect(adv).toEqual([11,12]); // males only
  });
});
