import { generateIndividualValues, type IndividualValues } from './individual-values';
import { lcgSeedToMtSeed } from './lcg-seed';
import type { GeneratedSpecies } from '@/data/species/generated';

export type StatKey = 'hp' | 'attack' | 'defense' | 'specialAttack' | 'specialDefense' | 'speed';
export type NonHpStatKey = Exclude<StatKey, 'hp'>;

export interface CalculatedStats {
  hp: number;
  attack: number;
  defense: number;
  specialAttack: number;
  specialDefense: number;
  speed: number;
}

interface NatureEffect {
  increase?: NonHpStatKey;
  decrease?: NonHpStatKey;
}

const NATURE_EFFECTS: readonly NatureEffect[] = [
  {},
  { increase: 'attack', decrease: 'defense' },
  { increase: 'attack', decrease: 'speed' },
  { increase: 'attack', decrease: 'specialAttack' },
  { increase: 'attack', decrease: 'specialDefense' },
  { increase: 'defense', decrease: 'attack' },
  {},
  { increase: 'defense', decrease: 'speed' },
  { increase: 'defense', decrease: 'specialAttack' },
  { increase: 'defense', decrease: 'specialDefense' },
  { increase: 'speed', decrease: 'attack' },
  { increase: 'speed', decrease: 'defense' },
  {},
  { increase: 'speed', decrease: 'specialAttack' },
  { increase: 'speed', decrease: 'specialDefense' },
  { increase: 'specialAttack', decrease: 'attack' },
  { increase: 'specialAttack', decrease: 'defense' },
  { increase: 'specialAttack', decrease: 'speed' },
  {},
  { increase: 'specialAttack', decrease: 'specialDefense' },
  { increase: 'specialDefense', decrease: 'attack' },
  { increase: 'specialDefense', decrease: 'defense' },
  { increase: 'specialDefense', decrease: 'speed' },
  { increase: 'specialDefense', decrease: 'specialAttack' },
  {},
] as const;

function natureMultiplier(natureId: number, stat: NonHpStatKey): number {
  const effect = NATURE_EFFECTS[natureId];
  if (!effect) return 1;
  if (effect.increase === stat) return 1.1;
  if (effect.decrease === stat) return 0.9;
  return 1;
}

function calculateHp(base: number, iv: number, level: number, speciesId?: number): number {
  if (speciesId === 292) return 1;
  const baseTerm = Math.floor(((base * 2 + iv) * level) / 100);
  return baseTerm + level + 10;
}

function calculateNonHp(base: number, iv: number, level: number, stat: NonHpStatKey, natureId: number): number {
  const baseTerm = Math.floor(((base * 2 + iv) * level) / 100);
  const preNature = baseTerm + 5;
  const modifier = natureMultiplier(natureId, stat);
  return Math.floor(preNature * modifier);
}

export function calculatePokemonStats(params: {
  species: GeneratedSpecies;
  ivs: IndividualValues;
  level: number;
  natureId: number;
}): CalculatedStats {
  const { species, ivs, level, natureId } = params;
  const { baseStats } = species;

  return {
    hp: calculateHp(baseStats.hp, ivs.h, level, species.nationalDex),
    attack: calculateNonHp(baseStats.attack, ivs.a, level, 'attack', natureId),
    defense: calculateNonHp(baseStats.defense, ivs.b, level, 'defense', natureId),
    specialAttack: calculateNonHp(baseStats.specialAttack, ivs.c, level, 'specialAttack', natureId),
    specialDefense: calculateNonHp(baseStats.specialDefense, ivs.d, level, 'specialDefense', natureId),
    speed: calculateNonHp(baseStats.speed, ivs.s, level, 'speed', natureId),
  };
}

export interface IvComputationContext {
  version: 'B' | 'W' | 'B2' | 'W2';
  encounterType: number;
}

export function determineIvOffset(context: IvComputationContext): { offset: number; isRoamer: boolean } {
  const isRoamer = context.encounterType === 20;
  if (isRoamer) {
    return { offset: 1, isRoamer: true };
  }
  const isBw2 = context.version === 'B2' || context.version === 'W2';
  return { offset: isBw2 ? 2 : 0, isRoamer: false };
}

export function computeIndividualValuesFromSeed(seed: bigint, context: IvComputationContext): IndividualValues {
  const mtSeed = lcgSeedToMtSeed(seed);
  const { offset, isRoamer } = determineIvOffset(context);
  return generateIndividualValues(mtSeed, offset, isRoamer);
}
