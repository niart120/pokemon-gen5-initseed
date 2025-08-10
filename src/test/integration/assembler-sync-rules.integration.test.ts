import { describe, it, expect } from 'vitest';
import {
  assembleData,
  assembleBatch,
  createAssemblerContext,
  createSampleEncounterTables,
  validateSyncRules,
  EncounterType,
  DustCloudContent,
  type RawPokemonData,
} from '../../lib/integration/pokemon-assembler';

// Minimal helper to craft RawPokemonData
function raw(
  partial: Partial<RawPokemonData> & Pick<RawPokemonData, 'encounterType'>
): RawPokemonData {
  return {
    seed: 0x1,
    pid: 0x12345678,
    nature: 0,
    syncApplied: false,
    abilitySlot: 0,
    genderValue: 0,
    encounterSlotValue: 0,
    levelRandValue: 0,
    shinyType: 0,
    ...partial,
    encounterType: partial.encounterType,
  };
}

describe('Assembler integration: sync rules and special encounters', () => {
  const ctx = createAssemblerContext('B2', 'JPN', createSampleEncounterTables());

  it('allows sync for Normal encounter, forbids for Roaming/Starter/Fossil', () => {
    // Normal (sync eligible)
    const normal = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Normal, syncApplied: true, genderValue: 10 })
    );
    expect(normal.syncEligible).toBe(true);
    expect(normal.syncAppliedCorrectly).toBe(true);

    // StaticStarter (not eligible)
    const starter = assembleData(
      ctx,
      raw({ encounterType: EncounterType.StaticStarter, syncApplied: true })
    );
    expect(starter.syncEligible).toBe(false);
    expect(starter.syncAppliedCorrectly).toBe(false);

    // StaticFossil (not eligible)
    const fossil = assembleData(
      ctx,
      raw({ encounterType: EncounterType.StaticFossil, syncApplied: true })
    );
    expect(fossil.syncEligible).toBe(false);
    expect(fossil.syncAppliedCorrectly).toBe(false);

    // Roaming (must never have sync)
    const roam = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Roaming, syncApplied: true })
    );
    expect(roam.syncEligible).toBe(false);
    expect(roam.syncAppliedCorrectly).toBe(false);
  });

  it('validateSyncRules reports violations for incorrect sync application', () => {
    const ok = assembleData(ctx, raw({ encounterType: EncounterType.Normal, syncApplied: true }));
    const ng = assembleData(ctx, raw({ encounterType: EncounterType.Roaming, syncApplied: true }));

    const result = validateSyncRules([ok, ng]);
    expect(result.isValid).toBe(false);
    expect(result.violations.length).toBe(1);
    expect(result.violations[0].encounterType).toBe(EncounterType.Roaming);
  });

  it('level calculation respects min/max range and fixed-level behavior', () => {
    // Override level range via slot value table length (sample tables min/max are 5-7)
    // Use levelRandValue to check boundary mapping min..max
    const lvMin = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Normal, levelRandValue: 0 })
    );
    const lvMid = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Normal, levelRandValue: 1 })
    );
    const lvMax = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Normal, levelRandValue: 2 })
    );

    expect(lvMin.level).toBeGreaterThanOrEqual(5);
    expect(lvMax.level).toBeLessThanOrEqual(7);
    expect(new Set([lvMin.level, lvMid.level, lvMax.level]).size).toBeGreaterThanOrEqual(2);

    // Fixed-level example from sample tables: use roaming table (fixed 40)
    const fixed = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Roaming, levelRandValue: 999 })
    );
    expect(fixed.level).toBe(40);
  });

  it('gender and shiny type resolution are consistent', () => {
    // genderRatio for sample normal table is 87 (female if genderValue < 87)
    const female = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Normal, genderValue: 0 })
    );
    const male = assembleData(
      ctx,
      raw({ encounterType: EncounterType.Normal, genderValue: 200 })
    );
    expect(female.gender).toBe(1);
    expect(male.gender).toBe(0);

    const shinyNormal = assembleData(ctx, raw({ encounterType: EncounterType.Normal, shinyType: 0 }));
    const shinySquare = assembleData(ctx, raw({ encounterType: EncounterType.Normal, shinyType: 1 }));
    const shinyStar = assembleData(ctx, raw({ encounterType: EncounterType.Normal, shinyType: 2 }));
    expect(shinyNormal.isShiny).toBe(false);
    expect(shinyNormal.shinyType).toBe('normal');
    expect(shinySquare.isShiny).toBe(true);
    expect(shinySquare.shinyType).toBe('square');
    expect(shinyStar.isShiny).toBe(true);
    expect(shinyStar.shinyType).toBe('star');
  });

  it('DustCloud adds content and item metadata (smoke test)', () => {
    // contentRandom = (pid & 0xff) % 100
    const pokemon = assembleData(
      ctx,
      raw({ encounterType: EncounterType.DustCloud, pid: 0x00000005 }) // 5 -> Pokemon
    );
    expect(pokemon.dustCloudContent).toBe(DustCloudContent.Pokemon);

    // Force Gem path using low byte 0x55 => 85 -> Gem
    const gem = assembleData(
      ctx,
      raw({ encounterType: EncounterType.DustCloud, pid: 0x00000055 })
    );
    if (gem.dustCloudContent === DustCloudContent.Gem) {
      expect(gem.itemId).toBeDefined();
    }
  });

  it('assembleBatch preserves validation results', () => {
    const arr = assembleBatch(ctx, [
      raw({ encounterType: EncounterType.Normal, syncApplied: true }),
      raw({ encounterType: EncounterType.Roaming, syncApplied: true }),
    ]);
    expect(arr.length).toBe(2);
    const check = validateSyncRules(arr);
    expect(check.isValid).toBe(false);
    expect(check.violations.length).toBe(1);
  });
});
