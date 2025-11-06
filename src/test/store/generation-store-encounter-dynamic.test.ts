import { describe, it, expect } from 'vitest';
import { create } from 'zustand';
import { createGenerationSlice, type GenerationSlice, bindGenerationManager } from '@/store/generation-store';
import { DomainEncounterType } from '@/types/domain';

function setupStore() {
  const useStore = create<GenerationSlice>()((set, get) => ({
    ...createGenerationSlice(set as any, get as any),
  }));
  bindGenerationManager(useStore.getState as any); // ensure callbacks wired (not strictly needed for these tests)
  return useStore;
}

describe('generation-store dynamic encounter fields', () => {
  it('encounterField + species reset when encounterType changes', () => {
    const store = setupStore();
    // set initial draft encounterType and dynamic fields
    store.getState().setDraftParams({ encounterType: DomainEncounterType.Normal });
    store.getState().setEncounterField('route1');
    store.getState().setEncounterSpeciesId(495); // Snivy example
    store.getState().setStaticEncounterId('route1-snivy');
    expect(store.getState().encounterField).toBe('route1');
    expect(store.getState().encounterSpeciesId).toBe(495);
    expect(store.getState().staticEncounterId).toBe('route1-snivy');
    // change encounterType
    store.getState().setDraftParams({ encounterType: DomainEncounterType.StaticStarter });
    expect(store.getState().draftParams.encounterType).toBe(DomainEncounterType.StaticStarter);
    // dynamic fields cleared
    expect(store.getState().encounterField).toBeUndefined();
    expect(store.getState().encounterSpeciesId).toBeUndefined();
    expect(store.getState().staticEncounterId).toBeNull();
  });

  it('encounterField sets also clears speciesId (location change)', () => {
    const store = setupStore();
    store.getState().setDraftParams({ encounterType: DomainEncounterType.Normal });
    store.getState().setEncounterField('route1');
    store.getState().setEncounterSpeciesId(501); // Oshawott
    expect(store.getState().encounterSpeciesId).toBe(501);
    // simulate user selecting different location
    store.getState().setEncounterField('route2');
    expect(store.getState().encounterField).toBe('route2');
    expect(store.getState().encounterSpeciesId).toBeUndefined();
    expect(store.getState().staticEncounterId).toBeNull();
  });

  it('setStaticEncounterId stores selection outside draft params', () => {
    const store = setupStore();
    store.getState().setStaticEncounterId('legendary-zekrom');
    expect(store.getState().staticEncounterId).toBe('legendary-zekrom');
    expect('staticEncounterId' in store.getState().draftParams).toBe(false);
    store.getState().setStaticEncounterId(null);
    expect(store.getState().staticEncounterId).toBeNull();
    expect('staticEncounterId' in store.getState().draftParams).toBe(false);
  });
});
