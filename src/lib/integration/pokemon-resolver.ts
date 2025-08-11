/**
 * Pokemon Resolver - domain layer to convert raw WASM output into resolved data
 *
 * Scope:
 * - Input: RawPokemonData (snake_case) from src/types/pokemon-raw
 * - Output: ResolvedPokemonData (ID/enum中心の機械可読モデル)
 * - No UI/i18n concerns here. Names/labels are handled in UI adapter layer.
 *
 * Notes:
 * - Keep this file UI-agnostic and side-effect free (pure functions preferred).
 * - Encounter table/gender ratio/ability catalog are provided via context.
 */

import type {
  RawPokemonData as WasmRawPokemonData,
  EncounterTable,
  GenderRatio,
} from '@/types/pokemon-raw';
import { DomainEncounterType, DomainNatureNames, DomainShinyType } from '@/types/domain';

// Context to supply reference data and environment for resolution
export interface ResolutionContext {
  encounterTable?: EncounterTable; // Selected table for current area/version/type
  genderRatios?: Map<number, GenderRatio>; // species_id -> gender ratio info
  abilityCatalog?: Map<number, string[]>; // species_id -> [ability1, ability2?, hidden?]
}

// Machine-readable resolved output (no localized strings)
export interface ResolvedPokemonData {
  // echoes from raw
  seed: bigint;
  pid: number;
  natureId: number;
  syncApplied: boolean;
  abilitySlot: number;
  genderValue: number;
  encounterSlotValue: number;
  encounterType: number; // DomainEncounterType numeric
  levelRandValue: number;
  shinyType: number; // DomainShinyType numeric

  // resolved fields
  speciesId?: number;
  level?: number;
  gender?: 'M' | 'F' | 'N';
  // For domain layer keep ability index; name mapping is UI concern
  abilityIndex?: 0 | 1 | 2; // 0: ability1, 1: ability2, 2: hidden
}

// Lightweight UI adapter output (still minimal, names handled by UI layer higher up)
export interface UiReadyPokemonData extends ResolvedPokemonData {
  natureName: string;
  shinyStatus: 'normal' | 'square' | 'star';
}

// Public API
export function resolvePokemon(
  raw: WasmRawPokemonData,
  ctx: ResolutionContext
): ResolvedPokemonData {
  const speciesId = resolveSpeciesId(raw, ctx.encounterTable);
  const level = resolveLevel(raw, ctx.encounterTable);
  const gender = resolveGender(raw, speciesId, ctx.genderRatios);
  const abilityIndex = resolveAbilityIndex(raw);

  return {
    seed: raw.seed,
    pid: raw.pid,
    natureId: raw.nature,
    syncApplied: raw.sync_applied,
    abilitySlot: raw.ability_slot,
    genderValue: raw.gender_value,
    encounterSlotValue: raw.encounter_slot_value,
    encounterType: raw.encounter_type,
    levelRandValue: raw.level_rand_value,
    shinyType: raw.shiny_type,
    speciesId,
    level,
    gender,
    abilityIndex,
  };
}

export function resolveBatch(
  raws: WasmRawPokemonData[],
  ctx: ResolutionContext
): ResolvedPokemonData[] {
  return raws.map((r) => resolvePokemon(r, ctx));
}

// UI adapter helpers (kept here for convenience but still UI-agnostic)
export function toUiReadyPokemon(data: ResolvedPokemonData): UiReadyPokemonData {
  return {
    ...data,
    natureName: getNatureName(data.natureId),
    shinyStatus: toShinyStatus(data.shinyType),
  };
}

// =============== internal helpers ===============

function resolveSpeciesId(
  raw: WasmRawPokemonData,
  table?: EncounterTable
): number | undefined {
  if (!table || !table.species_list?.length) return undefined;

  // Basic policy: use encounter_slot_value as index if within range.
  // If values encode probability or different mapping, replace here later.
  const idx = raw.encounter_slot_value;
  const list = table.species_list;
  if (idx >= 0 && idx < list.length) return list[idx]?.species_id;

  // Fallback: modulo mapping to avoid throwing in early wiring
  return list[Math.abs(idx) % list.length]?.species_id;
}

function resolveLevel(
  raw: WasmRawPokemonData,
  table?: EncounterTable
): number | undefined {
  // Static encounters (>= 10) use fixed levels when provided
  if (raw.encounter_type >= DomainEncounterType.StaticSymbol && table) {
    const index = resolveSpeciesIndexSafe(raw, table);
    if (index != null) {
      const cfg = table.species_list[index]?.level_config;
      if (cfg?.fixed_level != null) return cfg.fixed_level;
    }
  }

  if (!table) return undefined;
  const index = resolveSpeciesIndexSafe(raw, table);
  if (index == null) return undefined;

  const cfg = table.species_list[index]?.level_config;
  if (!cfg) return undefined;
  if (cfg.fixed_level != null) return cfg.fixed_level;

  const min = cfg.min_level;
  const max = cfg.max_level;
  if (min == null || max == null || max < min) return undefined;

  const range = max - min + 1;
  // level_rand_value is 0..0xFFFFFFFF → [0,1) ratio
  const ratio = raw.level_rand_value / 0x100000000;
  const offset = Math.floor(ratio * range);
  return min + offset;
}

function resolveGender(
  raw: WasmRawPokemonData,
  speciesId: number | undefined,
  ratios?: Map<number, GenderRatio>
): 'M' | 'F' | 'N' | undefined {
  if (!speciesId || !ratios) return undefined;
  const r = ratios.get(speciesId);
  if (!r) return undefined;
  if (r.genderless) return 'N';
  // female if value < female_threshold
  const femaleThreshold = r.male_ratio; // naming in schema; threshold semantics per docs
  return raw.gender_value < femaleThreshold ? 'F' : 'M';
}

function resolveAbilityIndex(raw: WasmRawPokemonData): 0 | 1 | 2 | undefined {
  if (raw.ability_slot === 0) return 0;
  if (raw.ability_slot === 1) return 1;
  if (raw.ability_slot === 2) return 2;
  return undefined;
}

function resolveSpeciesIndexSafe(
  raw: WasmRawPokemonData,
  table: EncounterTable
): number | undefined {
  const idx = raw.encounter_slot_value;
  if (idx >= 0 && idx < table.species_list.length) return idx;
  return Math.abs(idx) % table.species_list.length;
}

function getNatureName(natureId: number): string {
  if (natureId < 0 || natureId >= DomainNatureNames.length) {
    return 'Unknown';
  }
  return DomainNatureNames[natureId];
}

function toShinyStatus(shinyType: number): 'normal' | 'square' | 'star' {
  switch (shinyType) {
    case DomainShinyType.Normal:
      return 'normal';
    case DomainShinyType.Square:
      return 'square';
    case DomainShinyType.Star:
      return 'star';
    default:
      return 'normal';
  }
}
