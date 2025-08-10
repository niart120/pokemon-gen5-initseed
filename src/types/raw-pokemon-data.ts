/**
 * Raw Pokemon Data types and parser for WASM integration
 * 
 * This module provides TypeScript types and parsing utilities for the RawPokemonData
 * structure returned by the WASM pokemon_generator module.
 * 
 * Source of Truth: wasm-pkg/src/pokemon_generator.rs RawPokemonData struct
 */
import { DomainEncounterType, DomainShinyType } from './domain';
// Provide local value aliases to use in code below
const EncounterType = DomainEncounterType;
const ShinyType = DomainShinyType;
// Re-export to keep existing import paths working
export { DomainEncounterType as EncounterType };
export { DomainShinyType as ShinyType };

/**
 * Raw Pokemon data structure (mirrors WASM RawPokemonData)
 * 
 * This represents the minimal data returned directly from WASM calculations
 * before enrichment with species/encounter table data.
 */
export interface RawPokemonData {
  /** Initial seed value */
  seed: bigint;
  /** Personality ID (32-bit) */
  pid: number;
  /** Nature ID (0-24) */
  nature: number;
  /** Whether synchronize was applied */
  syncApplied: boolean;
  /** Ability slot (0-1) */
  abilitySlot: number;
  /** Gender value (0-255) */
  genderValue: number;
  /** Encounter slot value for encounter table lookup */
  encounterSlotValue: number;
  /** Encounter type ID (numerical representation) */
  encounterType: number;
  /** Level random value for level calculation */
  levelRandValue: number;
  /** Shiny type (0: Normal, 1: Square, 2: Star) */
  shinyType: number;
}

/**
 * Human-readable shiny status type
 */
export type ShinyStatusName = 'Normal' | 'Square Shiny' | 'Star Shiny';

/**
 * Enhanced Pokemon data with detailed information
 * 
 * This represents the final Pokemon data after combining raw WASM output
 * with encounter table data and species information.
 */
export interface EnhancedPokemonData extends RawPokemonData {
  /** Species information */
  species: PokemonSpecies;
  /** Ability information */
  ability: PokemonAbility;
  /** Gender (derived from gender value and species gender ratio) */
  gender: 'Male' | 'Female' | 'Genderless';
  /** Actual level (calculated from level random value) */
  level: number;
  /** Encounter location/method details */
  encounter: EncounterDetails;
  /** Human-readable nature name */
  natureName: string;
  /** Human-readable shiny status */
  shinyStatus: ShinyStatusName;
}

/**
 * Pokemon species data
 */
export interface PokemonSpecies {
  /** National Dex number */
  nationalDex: number;
  /** Species name */
  name: string;
  /** Base stats */
  baseStats: {
    hp: number;
    attack: number;
    defense: number;
    specialAttack: number;
    specialDefense: number;
    speed: number;
  };
  /** Type(s) */
  types: [string] | [string, string];
  /** Gender ratio (percent male, -1 for genderless) */
  genderRatio: number;
  /** Available abilities */
  abilities: {
    ability1: string;
    ability2?: string;
    hiddenAbility?: string;
  };
}

/**
 * Pokemon ability data
 */
export interface PokemonAbility {
  /** Ability name */
  name: string;
  /** Ability description */
  description: string;
  /** Whether this is a hidden ability */
  isHidden: boolean;
}

/**
 * Encounter details
 */
export interface EncounterDetails {
  /** Encounter method name */
  method: string;
  /** Location name */
  location: string;
  /** Encounter rate (if applicable) */
  rate?: number;
  /** Level range */
  levelRange: {
    min: number;
    max: number;
  };
}

/**
 * Encounter type enumeration (matches WASM values)
 */
// Use shared domain enums to avoid duplication (see exports above)

/**
 * Shiny type enumeration (matches WASM values)
 */
// see exports above

/**
 * Nature names array (index corresponds to nature ID)
 */
export const NATURE_NAMES = [
  'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
  'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
  'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
  'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
  'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
] as const;

/**
 * Parse raw Pokemon data from WASM module
 * 
 * Converts WASM RawPokemonData object to TypeScript RawPokemonData interface
 * 
 * @param wasmData Raw data object from WASM module
 * @returns Parsed RawPokemonData
 */
export function parseRawPokemonData(wasmData: unknown): RawPokemonData {
  if (!wasmData) {
    throw new Error('WASM data is null or undefined');
  }

  // getter関数/プロパティ両対応のフィールド取得ヘルパ
  const readField = (obj: Record<string, unknown>, key: string) => {
    const candidate = obj[key];
    const val = typeof candidate === 'function' ? (candidate as () => unknown)() : candidate;
    if (val === undefined) {
      throw new Error(`Missing required property or method: ${key}`);
    }
    return val;
  };

  try {
    const obj = wasmData as Record<string, unknown>;
    const toBigInt = (v: unknown): bigint => {
      if (typeof v === 'bigint') return v;
      if (typeof v === 'number') return BigInt(Math.trunc(v));
      if (typeof v === 'string') return BigInt(v);
      if (typeof v === 'boolean') return BigInt(v ? 1 : 0);
      throw new Error(`Invalid bigint-like value: ${String(v)}`);
    };
    const toNumber = (v: unknown): number => {
      if (typeof v === 'number') return v;
      if (typeof v === 'bigint') return Number(v);
      if (typeof v === 'string') {
        const n = Number(v);
        if (!Number.isFinite(n)) throw new Error(`Invalid number string: ${v}`);
        return n;
      }
      if (typeof v === 'boolean') return v ? 1 : 0;
      throw new Error(`Invalid number-like value: ${String(v)}`);
    };
  const seedVal = readField(obj, 'get_seed');
  const pid = readField(obj, 'get_pid');
  const nature = readField(obj, 'get_nature');
  const syncApplied = readField(obj, 'get_sync_applied');
  const abilitySlot = readField(obj, 'get_ability_slot');
  const genderValue = readField(obj, 'get_gender_value');
  const encounterSlotValue = readField(obj, 'get_encounter_slot_value');
  const encounterType = readField(obj, 'get_encounter_type');
  const levelRandValue = readField(obj, 'get_level_rand_value');
  const shinyType = readField(obj, 'get_shiny_type');

    return {
      seed: toBigInt(seedVal),
      pid: toNumber(pid),
      nature: toNumber(nature),
      syncApplied: Boolean(syncApplied),
      abilitySlot: toNumber(abilitySlot),
      genderValue: toNumber(genderValue),
      encounterSlotValue: toNumber(encounterSlotValue),
      encounterType: toNumber(encounterType),
      levelRandValue: toNumber(levelRandValue),
      shinyType: toNumber(shinyType),
    };
  } catch (error) {
    throw new Error(`Failed to parse WASM pokemon data: ${error}`);
  }
}

/**
 * Get human-readable nature name from nature ID
 */
export function getNatureName(natureId: number): string {
  if (natureId < 0 || natureId >= NATURE_NAMES.length) {
    throw new Error(`Invalid nature ID: ${natureId}`);
  }
  return NATURE_NAMES[natureId];
}

/**
 * Get human-readable shiny status from shiny type
 */
export function getShinyStatusName(shinyType: number): ShinyStatusName {
  switch (shinyType) {
    case ShinyType.Normal:
      return 'Normal';
    case ShinyType.Square:
      return 'Square Shiny';
    case ShinyType.Star:
      return 'Star Shiny';
    default:
      throw new Error(`Invalid shiny type: ${shinyType}`);
  }
}

/**
 * Get encounter type name from encounter type ID
 */
export function getEncounterTypeName(encounterType: number): string {
  const typeNames: Record<number, string> = {
    [EncounterType.Normal]: 'Wild Encounter',
    [EncounterType.Surfing]: 'Surfing',
    [EncounterType.Fishing]: 'Fishing',
    [EncounterType.ShakingGrass]: 'Shaking Grass',
    [EncounterType.DustCloud]: 'Dust Cloud',
    [EncounterType.PokemonShadow]: 'Pokemon Shadow',
    [EncounterType.SurfingBubble]: 'Surfing (Bubble)',
    [EncounterType.FishingBubble]: 'Fishing (Bubble)',
    [EncounterType.StaticSymbol]: 'Static Symbol',
    [EncounterType.StaticStarter]: 'Starter Pokemon',
    [EncounterType.StaticFossil]: 'Fossil Pokemon',
    [EncounterType.StaticEvent]: 'Event Pokemon',
    [EncounterType.Roaming]: 'Roaming Pokemon',
  };

  const name = typeNames[encounterType];
  if (!name) {
    throw new Error(`Unknown encounter type: ${encounterType}`);
  }
  return name;
}

/**
 * Determine gender from gender value and species gender ratio
 * 
 * @param genderValue Gender value (0-255) from PID
 * @param genderRatio Species gender ratio (percent male, -1 for genderless)
 * @returns Gender string
 */
// Legacy determineGender(genderRatio) is removed.
// Use determineGenderFromSpec(genderValue, { type: 'genderless' | 'fixed' | 'ratio', ... })