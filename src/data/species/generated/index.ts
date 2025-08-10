import dataset from './gen5-species.json';

export type LocalizedName = { en: string; ja: string };

export interface GeneratedAbilities {
  ability1: { key: string; names: LocalizedName } | null;
  ability2: { key: string; names: LocalizedName } | null;
  hidden: { key: string; names: LocalizedName } | null;
}

export interface GeneratedGenderSpec {
  type: 'genderless' | 'fixed' | 'ratio';
  fixed?: 'male' | 'female';
  femaleThreshold?: number;
}

export interface GeneratedSpecies {
  nationalDex: number;
  names: LocalizedName;
  gender: GeneratedGenderSpec;
  baseStats: {
    hp: number; attack: number; defense: number; specialAttack: number; specialDefense: number; speed: number;
  };
  abilities: GeneratedAbilities;
  heldItems: Record<'black' | 'white' | 'black-2' | 'white-2', Array<{ key: string; names: LocalizedName; rarity?: number }>>;
}

const byId: Record<string, GeneratedSpecies> = dataset as any;

export function getGeneratedSpeciesById(id: number): GeneratedSpecies | null {
  const s = (byId as any)[String(id)] as GeneratedSpecies | undefined;
  return s ?? null;
}

/** ability_slot 0 -> ability1, 1 -> ability2 (hiddenはここでは選ばない) */
export function selectAbilityBySlot(slot: number, abilities: GeneratedAbilities): { key: string; names: LocalizedName } | null {
  if (slot === 0) return abilities.ability1;
  if (slot === 1) return abilities.ability2;
  return null;
}
