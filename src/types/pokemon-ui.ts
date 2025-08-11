/**
 * UI-facing Pokemon types and helpers (compat shim replacing pokemon-enhanced.ts)
 *
 * Note: This module preserves the previous camelCase RawPokemonData shape for UI/tests
 * while the domain layer uses snake_case in src/types/pokemon-raw.ts.
 */

import { DomainEncounterType, DomainShinyType, DomainNatureNames } from './domain';
import { parseWasmLikeToRawPokemonData } from '@/lib/integration/raw-parser';

export { DomainEncounterType as EncounterType };
export { DomainShinyType as ShinyType };

export interface RawPokemonData {
  seed: bigint;
  pid: number;
  nature: number;
  syncApplied: boolean;
  abilitySlot: number;
  genderValue: number;
  encounterSlotValue: number;
  encounterType: number;
  levelRandValue: number;
  shinyType: number;
}

export type ShinyStatusName = 'Normal' | 'Square Shiny' | 'Star Shiny';

export interface EnhancedPokemonData extends RawPokemonData {
  species: PokemonSpecies;
  ability: PokemonAbility;
  gender: 'Male' | 'Female' | 'Genderless';
  level: number;
  encounter: EncounterDetails;
  natureName: string;
  shinyStatus: ShinyStatusName;
}

export interface PokemonSpecies {
  nationalDex: number;
  name: string;
  baseStats: {
    hp: number;
    attack: number;
    defense: number;
    specialAttack: number;
    specialDefense: number;
    speed: number;
  };
  types: [string] | [string, string];
  genderRatio: number;
  abilities: {
    ability1: string;
    ability2?: string;
    hiddenAbility?: string;
  };
}

export interface PokemonAbility {
  name: string;
  description: string;
  isHidden: boolean;
}

export interface EncounterDetails {
  method: string;
  location: string;
  rate?: number;
  levelRange: { min: number; max: number };
}

// Nature名は domain.ts の単一ソースを使用
export const NATURE_NAMES = DomainNatureNames;

// UI層のエントリポイント: WASMライク入力 → UI向けの camelCase RawPokemonData
export function parseRawPokemonData(wasmData: unknown): RawPokemonData {
  try {
    const snake = parseWasmLikeToRawPokemonData(wasmData as Record<string, unknown>);
    return {
      seed: snake.seed,
      pid: snake.pid,
      nature: snake.nature,
      syncApplied: snake.sync_applied,
      abilitySlot: snake.ability_slot,
      genderValue: snake.gender_value,
      encounterSlotValue: snake.encounter_slot_value,
      encounterType: snake.encounter_type,
      levelRandValue: snake.level_rand_value,
      shinyType: snake.shiny_type,
    };
  } catch (error) {
    throw new Error(`Failed to parse WASM pokemon data: ${error}`);
  }
}

export function getNatureName(natureId: number): string {
  if (natureId < 0 || natureId >= DomainNatureNames.length) {
    throw new Error(`Invalid nature ID: ${natureId}`);
  }
  return DomainNatureNames[natureId];
}

export function getShinyStatusName(shinyType: number): ShinyStatusName {
  switch (shinyType) {
    case DomainShinyType.Normal:
      return 'Normal';
    case DomainShinyType.Square:
      return 'Square Shiny';
    case DomainShinyType.Star:
      return 'Star Shiny';
    default:
      throw new Error(`Invalid shiny type: ${shinyType}`);
  }
}

export function getEncounterTypeName(encounterType: number): string {
  const typeNames: Record<number, string> = {
    [DomainEncounterType.Normal]: 'Wild Encounter',
    [DomainEncounterType.Surfing]: 'Surfing',
    [DomainEncounterType.Fishing]: 'Fishing',
    [DomainEncounterType.ShakingGrass]: 'Shaking Grass',
    [DomainEncounterType.DustCloud]: 'Dust Cloud',
    [DomainEncounterType.PokemonShadow]: 'Pokemon Shadow',
    [DomainEncounterType.SurfingBubble]: 'Surfing (Bubble)',
    [DomainEncounterType.FishingBubble]: 'Fishing (Bubble)',
    [DomainEncounterType.StaticSymbol]: 'Static Symbol',
    [DomainEncounterType.StaticStarter]: 'Starter Pokemon',
    [DomainEncounterType.StaticFossil]: 'Fossil Pokemon',
    [DomainEncounterType.StaticEvent]: 'Event Pokemon',
    [DomainEncounterType.Roaming]: 'Roaming Pokemon',
  };

  const name = typeNames[encounterType];
  if (!name) {
    throw new Error(`Unknown encounter type: ${encounterType}`);
  }
  return name;
}
