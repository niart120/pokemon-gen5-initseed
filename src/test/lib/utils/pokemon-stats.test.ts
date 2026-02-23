import { describe, expect, it } from 'vitest';

import { computeIndividualValuesFromSeed, determineIvOffset, calculatePokemonStats } from '@/lib/utils/pokemon-stats';
import { generateIndividualValues } from '@/lib/utils/individual-values';
import type { GeneratedSpecies } from '@/data/species/generated';

function createSpecies(overrides: Partial<GeneratedSpecies> = {}): GeneratedSpecies {
  return {
    nationalDex: overrides.nationalDex ?? 1,
    names: overrides.names ?? { en: 'Test', ja: 'テスト' },
    gender: overrides.gender ?? { type: 'genderless' },
    baseStats: overrides.baseStats ?? {
      hp: 45,
      attack: 49,
      defense: 49,
      specialAttack: 65,
      specialDefense: 65,
      speed: 45,
    },
    abilities: overrides.abilities ?? {
      ability1: null,
      ability2: null,
      hidden: null,
    },
    heldItems: overrides.heldItems ?? { black: [], white: [], 'black-2': [], 'white-2': [] },
  };
}

describe('determineIvOffset', () => {
  it('returns roamer offset', () => {
    const result = determineIvOffset({ version: 'B', encounterType: 20 });
    expect(result).toEqual({ offset: 1, isRoamer: true });
  });

  it('returns BW2 offset for non-roamers', () => {
    const result = determineIvOffset({ version: 'B2', encounterType: 0 });
    expect(result).toEqual({ offset: 2, isRoamer: false });
  });

  it('returns zero offset for BW wild encounters', () => {
    const result = determineIvOffset({ version: 'B', encounterType: 0 });
    expect(result).toEqual({ offset: 0, isRoamer: false });
  });
});

describe('computeIndividualValuesFromSeed', () => {
  it('matches manual generation', () => {
    const seed = 0x123456789ABCDEFn;
    const context = { version: 'B2' as const, encounterType: 0 };
    const mtSeed = Number(((seed * 0x5d588b656c078965n + 0x269ec3n) >> 32n) & 0xffffffffn);
    const manual = generateIndividualValues(mtSeed, 2, false);
    expect(computeIndividualValuesFromSeed(seed, context)).toEqual(manual);
  });
});

describe('calculatePokemonStats', () => {
  it('computes neutral stats correctly', () => {
    const species = createSpecies();
    const stats = calculatePokemonStats({
      species,
      ivs: { h: 31, a: 31, b: 31, c: 31, d: 31, s: 31 },
      level: 50,
      natureId: 0,
    });
    expect(stats).toEqual({
      hp: 120,
      attack: 69,
      defense: 69,
      specialAttack: 85,
      specialDefense: 85,
      speed: 65,
    });
  });

  it('applies nature multipliers', () => {
    const species = createSpecies();
    const stats = calculatePokemonStats({
      species,
      ivs: { h: 31, a: 31, b: 31, c: 31, d: 31, s: 31 },
      level: 50,
      natureId: 3,
    });
    expect(stats.attack).toBe(75);
    expect(stats.specialAttack).toBe(76);
  });

  it('handles Shedinja HP special case', () => {
    const species = createSpecies({
      nationalDex: 292,
      baseStats: {
        hp: 1,
        attack: 90,
        defense: 45,
        specialAttack: 30,
        specialDefense: 30,
        speed: 40,
      },
    });
    const stats = calculatePokemonStats({
      species,
      ivs: { h: 31, a: 0, b: 31, c: 31, d: 31, s: 31 },
      level: 50,
      natureId: 0,
    });
    expect(stats.hp).toBe(1);
  });
});
