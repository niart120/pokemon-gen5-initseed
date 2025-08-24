import { describe, it, expect, beforeAll } from 'vitest';
import { listEncounterLocations, listEncounterSpeciesOptions, isLocationBasedEncounter } from '@/data/encounters/helpers';
import { DomainEncounterType } from '@/types/domain';

// NOTE: Fixture depends on generated registry JSON already bundled via import.meta.glob eager load.

describe('encounter helpers (dynamic UI)', () => {
  const version = 'B' as const;
  const method = DomainEncounterType.Normal;

  it('isLocationBasedEncounter: Normal should be location based', () => {
    expect(isLocationBasedEncounter(method)).toBe(true);
  });

  it('listEncounterLocations should return non-empty list for Normal', () => {
    const locs = listEncounterLocations(version, method);
    expect(Array.isArray(locs)).toBe(true);
    expect(locs.length).toBeGreaterThan(0);
    // shape
    const first = locs[0];
    expect(first).toHaveProperty('key');
    expect(first).toHaveProperty('displayName');
  });

  it('listEncounterSpeciesOptions returns empty until locationKey provided', () => {
    const speciesNone = listEncounterSpeciesOptions(version, method, undefined);
    expect(speciesNone).toEqual([]);
  });

  it('listEncounterSpeciesOptions aggregates species for a known location', () => {
    const locs = listEncounterLocations(version, method);
    const target = locs[0];
    const species = listEncounterSpeciesOptions(version, method, target.key);
    expect(species.length).toBeGreaterThan(0);
    // aggregated fields
    const s0 = species[0];
    expect(s0).toHaveProperty('speciesId');
    expect(s0).toHaveProperty('firstSlotIndex');
    expect(s0).toHaveProperty('appearances');
    expect(s0).toHaveProperty('totalRate');
    expect(s0).toHaveProperty('minLevel');
    expect(s0).toHaveProperty('maxLevel');
  });

  it('caching: second call returns same reference', () => {
    const locs = listEncounterLocations(version, method);
    const key = locs[0].key;
    const s1 = listEncounterSpeciesOptions(version, method, key);
    const s2 = listEncounterSpeciesOptions(version, method, key);
    expect(s1).toBe(s2);
  });

  it('static encounter method (StaticStarter) should not be location based and return empty lists', () => {
    const staticStarter = DomainEncounterType.StaticStarter;
    expect(isLocationBasedEncounter(staticStarter)).toBe(false);
    const locs = listEncounterLocations(version, staticStarter);
    expect(locs).toEqual([]);
    const species = listEncounterSpeciesOptions(version, staticStarter, undefined);
    expect(species).toEqual([]); // placeholder WIP
  });
});
