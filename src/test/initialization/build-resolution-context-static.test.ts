import { describe, it, expect } from 'vitest';
import { buildResolutionContext } from '@/lib/initialization/build-resolution-context';
import { DomainEncounterType } from '@/types/domain';

describe('buildResolutionContext (static encounters)', () => {
  it('creates a synthetic encounter table for a static entry', () => {
    const ctx = buildResolutionContext({
      version: 'B2',
      encounterType: DomainEncounterType.StaticLegendary,
      staticEncounter: {
        id: 'dragonspiral-zekrom',
        speciesId: 644,
        level: 70,
      },
    });
    expect(ctx.encounterTable).toBeDefined();
    const table = ctx.encounterTable!;
    expect(table.slots).toHaveLength(1);
    expect(table.slots[0].speciesId).toBe(644);
    expect(table.slots[0].levelRange.min).toBe(70);
    expect(table.slots[0].levelRange.max).toBe(70);
  });
});
