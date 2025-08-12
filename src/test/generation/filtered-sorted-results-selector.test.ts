import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '@/store/app-store';
import { selectFilteredSortedResults } from '@/store/generation-store';

// サンプル GenerationResult データ
const baseResults = [
  { advance: 5, shiny_type: 0, seed: 5n, pid: 0xAAAA0001, nature: 3,  sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
  { advance: 2, shiny_type: 2, seed: 2n, pid: 0x00000002, nature: 1,  sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
  { advance: 8, shiny_type: 1, seed: 8n, pid: 0xBBBB0003, nature: 7,  sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
  { advance: 1, shiny_type: 0, seed: 1n, pid: 0x00000004, nature: 21, sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
];

function reset() {
  useAppStore.setState({ results: [...baseResults] });
  useAppStore.setState({ filters: { shinyOnly: false, natureIds: [], sortField: 'advance', sortOrder: 'asc', advanceRange: undefined, shinyTypes: undefined } });
}

describe('selectFilteredSortedResults', () => {
  beforeEach(() => reset());

  it('default sorted by advance asc', () => {
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(out.map(r => r.advance)).toEqual([1,2,5,8]);
  });

  it('shinyOnly filters non-shiny', () => {
    useAppStore.setState(s => ({ filters: { ...s.filters, shinyOnly: true } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(out.map(r => r.advance)).toEqual([2,8]);
  });

  it('shinyTypes subset filter', () => {
    useAppStore.setState(s => ({ filters: { ...s.filters, shinyTypes: [1] } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(out.map(r => r.shiny_type)).toEqual([1]);
  });

  it('natureIds filter + pid desc sort', () => {
    useAppStore.setState(s => ({ filters: { ...s.filters, natureIds: [1,3,7], sortField: 'pid', sortOrder: 'desc' } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(out.length).toBe(3);
    expect(out.map(r => r.pid >>> 0)).toEqual([0xBBBB0003, 0xAAAA0001, 0x00000002]);
  });

  it('advanceRange bounds', () => {
    useAppStore.setState(s => ({ filters: { ...s.filters, advanceRange: { min: 2, max: 5 } } }));
    const out = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(out.map(r => r.advance)).toEqual([2,5]);
  });

  it('memoization stable reference', () => {
    const a = selectFilteredSortedResults(useAppStore.getState() as any);
    const b = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(a).toBe(b);
    useAppStore.setState({ progress: { processedAdvances: 0 } as any });
    const c = selectFilteredSortedResults(useAppStore.getState() as any);
    expect(a).toBe(c);
  });
});
