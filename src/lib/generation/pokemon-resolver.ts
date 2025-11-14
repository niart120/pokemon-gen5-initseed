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

import type { UnresolvedPokemonData, GenderRatio } from '@/types/pokemon-raw';
import type { EncounterTable } from '@/data/encounter-tables';
import { natureName as formatNatureName, shinyDomainStatus } from '@/lib/utils/format-display';
import { getGeneratedSpeciesById, type GeneratedAbilities } from '@/data/species/generated';
import type { IndividualValues } from '@/lib/utils/individual-values';
import {
  calculatePokemonStats,
  computeIndividualValuesFromSeed,
  type CalculatedStats,
} from '@/lib/utils/pokemon-stats';
import { formatHexDisplay } from '@/lib/utils/hex-parser';

// Context to supply reference data and environment for resolution
export interface ResolutionContext {
  encounterTable?: EncounterTable; // Selected table for current area/version/type
  genderRatios?: Map<number, GenderRatio>; // species_id -> gender ratio info
  abilityCatalog?: Map<number, string[]>; // species_id -> [ability1, ability2?, hidden?]
}

// Machine-readable resolved output (no localized strings)
// For domain layer keep species ID and ability index; name mapping is UI concern
export type ResolvedPokemonData = Readonly<{
  // echoes from raw
  seed: bigint;
  pid: number;
  natureId: number;
  shinyType: number; // DomainShinyType numeric
  speciesId?: number;
  level?: number;
  gender?: 'M' | 'F' | 'N';
  abilityIndex?: 0 | 1 | 2; // 0: ability1, 1: ability2, 2: hidden
  encounterType: number;
}>;

// Lightweight UI output: only fields needed for display
export interface UiReadyPokemonData {
  seedHex: string; // 16進数表記 (0x...)
  pidHex: string; // 16進数表記 (0x...)
  speciesName: string; // ローカライズ済み名
  natureName: string; // ローカライズ済み名
  natureId: number;
  abilityName: string; // ローカライズ済み名（隠れ特性含む）
  gender: 'M' | 'F' | '-' | '?'; // '-'=性別不明/N, '?'=未解決
  level?: number;
  shinyStatus: 'normal' | 'square' | 'star';
  stats?: CalculatedStats;
  ivs?: IndividualValues;
}

// Public API
export function resolvePokemon(
  raw: UnresolvedPokemonData,
  ctx: ResolutionContext
): ResolvedPokemonData {
  const speciesId = resolveSpeciesId(raw, ctx.encounterTable);
  const level = resolveLevel(raw, ctx.encounterTable);
  const gender = resolveGender(raw, speciesId, ctx.genderRatios);
  const abilityIndex = normalizeAbilityIndex(
    speciesId,
    resolveAbilityIndex(raw)
  );

  return {
    seed: raw.seed,
    pid: raw.pid,
    natureId: raw.nature,
    shinyType: raw.shiny_type,
    speciesId,
    level,
    gender,
    abilityIndex,
    encounterType: raw.encounter_type,
  };
}

export function resolveBatch(
  raws: UnresolvedPokemonData[],
  ctx: ResolutionContext
): ResolvedPokemonData[] {
  return raws.map((r) => resolvePokemon(r, ctx));
}

// UI adapter helpers (kept here for convenience but still UI-agnostic)
export function toUiReadyPokemon(
  data: ResolvedPokemonData,
  opts: { locale?: 'ja' | 'en'; version?: 'B' | 'W' | 'B2' | 'W2'; baseSeed?: bigint } = {}
): UiReadyPokemonData {
  const locale = opts.locale ?? 'ja';
  const speciesName = getSpeciesName(data.speciesId, locale);
  const abilityName = getAbilityName(data.speciesId, data.abilityIndex, locale);
  let gender: 'M' | 'F' | '-' | '?';
  if (data.gender === 'M' || data.gender === 'F') {
    gender = data.gender;
  } else if (data.gender === 'N') {
    gender = '-';
  } else {
    gender = '?';
  }

  let ivs: IndividualValues | undefined;
  let stats: CalculatedStats | undefined;
  const species = data.speciesId ? getGeneratedSpeciesById(data.speciesId) : null;
  const levelReady = typeof data.level === 'number' && Number.isFinite(data.level);
  const baseSeed = opts.baseSeed;
  if (species && levelReady && baseSeed !== undefined) {
    const version = opts.version ?? 'B';
    ivs = computeIndividualValuesFromSeed(baseSeed, {
      version,
      encounterType: data.encounterType,
    });
    stats = calculatePokemonStats({
      species,
      ivs,
      level: data.level as number,
      natureId: data.natureId,
    });
  }

  return {
    seedHex: formatHexDisplay(data.seed, 16, true),
    pidHex: formatHexDisplay(data.pid >>> 0, 8, true),
    speciesName,
    natureName: formatNatureName(data.natureId, locale),
    natureId: data.natureId,
    abilityName,
    gender,
    level: data.level,
    shinyStatus: shinyDomainStatus(data.shinyType),
    stats,
    ivs,
  };
}

// =============== internal helpers ===============

function resolveSpeciesId(
  raw: UnresolvedPokemonData,
  table?: EncounterTable
): number | undefined {
  if (!table || !table.slots?.length) return undefined;

  // Basic policy: use encounter_slot_value as index if within range.
  // If values encode probability or different mapping, replace here later.
  const idx = raw.encounter_slot_value;
  const list = table.slots;
  if (idx >= 0 && idx < list.length) return list[idx]?.speciesId;

  // Fallback: modulo mapping to avoid throwing in early wiring
  return list[Math.abs(idx) % list.length]?.speciesId;
}

function resolveLevel(
  raw: UnresolvedPokemonData,
  table?: EncounterTable
): number | undefined {
  // 仕様（BW/BW2）のレベル計算ロジックに基づく一元実装
  // 式: (rand >> 32) * 0xFFFF / 0x290 >> 32 % (max - min + 1) + min
  // 前提: 現在の raw.level_rand_value(u32) が (rand >> 32) に相当する

  // 固定レベル（min===max）指定がある場合はそれを優先
  if (table) {
    const index = resolveSpeciesIndexSafe(raw, table);
    if (index != null) {
      const range0 = table.slots[index]?.levelRange;
      if (range0 && range0.min === range0.max) return range0.min;
    }
  }

  if (!table) return undefined;
  const index = resolveSpeciesIndexSafe(raw, table);
  if (index == null) return undefined;

  const cfg = table.slots[index]?.levelRange;
  if (!cfg) return undefined;

  const min = cfg.min;
  const max = cfg.max;
  if (min == null || max == null || max < min) return undefined;
  const range = max - min + 1;

  // BigIntで厳密に演算
  const upper32 = BigInt(Number(raw.level_rand_value) >>> 0); // (rand >> 32) に対応するu32
  const t1 = upper32 * 0xFFFFn;
  const t2 = t1 / 0x290n;
  const x = t2 >> 32n; // ((upper32 * 0xFFFF) / 0x290) の上位32bit
  const offset = Number(x % BigInt(range));
  return min + offset;
}

function resolveGender(
  raw: UnresolvedPokemonData,
  speciesId: number | undefined,
  ratios?: Map<number, GenderRatio>
): 'M' | 'F' | 'N' | undefined {
  if (!speciesId || !ratios) return undefined;
  const r = ratios.get(speciesId);
  if (!r) return undefined;
  if (r.genderless) return 'N';
  // female if value < threshold
  const femaleThreshold = r.threshold;
  return raw.gender_value < femaleThreshold ? 'F' : 'M';
}

function resolveAbilityIndex(raw: UnresolvedPokemonData): 0 | 1 | 2 | undefined {
  if (raw.ability_slot === 0) return 0;
  if (raw.ability_slot === 1) return 1;
  if (raw.ability_slot === 2) return 2;
  return undefined;
}

function normalizeAbilityIndex(
  speciesId: number | undefined,
  abilityIndex: 0 | 1 | 2 | undefined
): 0 | 1 | 2 | undefined {
  if (speciesId == null || abilityIndex == null) return abilityIndex;
  const species = getGeneratedSpeciesById(speciesId);
  if (!species) return abilityIndex;

  const { ability1, ability2, hidden } = species.abilities;

  // Align ability index with available slots so UI filters remain consistent.
  if (abilityIndex === 0) return ability1 ? 0 : undefined;
  if (abilityIndex === 1) {
    if (ability2) return 1;
    return ability1 ? 0 : undefined;
  }
  if (abilityIndex === 2) {
    if (hidden) return 2;
    if (ability2) return 1;
    return ability1 ? 0 : undefined;
  }
  return abilityIndex;
}

function resolveSpeciesIndexSafe(
  raw: UnresolvedPokemonData,
  table: EncounterTable
): number | undefined {
  const idx = raw.encounter_slot_value;
  if (idx >= 0 && idx < table.slots.length) return idx;
  return Math.abs(idx) % table.slots.length;
}

// (moved) nature/shiny formatting now lives in format-display.ts

// ======== UI adapter helpers (name/formatting) ========

// Hex formatting unified via formatHexDisplay()

function getSpeciesName(id: number | undefined, locale: 'ja' | 'en'): string {
  if (!id) return 'Unknown';
  const s = getGeneratedSpeciesById(id);
  if (!s) return 'Unknown';
  return locale === 'ja' ? s.names.ja : s.names.en;
}

function getAbilityName(
  speciesId: number | undefined,
  abilityIndex: 0 | 1 | 2 | undefined,
  locale: 'ja' | 'en'
): string {
  if (!speciesId || abilityIndex == null) return 'Unknown';
  const s = getGeneratedSpeciesById(speciesId);
  if (!s) return 'Unknown';
  const a = selectAbilityByIndex(abilityIndex, s.abilities);
  if (!a) return 'Unknown';
  return locale === 'ja' ? a.names.ja : a.names.en;
}

function selectAbilityByIndex(
  idx: 0 | 1 | 2,
  abilities: GeneratedAbilities
): { key: string; names: { en: string; ja: string } } | null {
  if (idx === 0) return abilities.ability1;
  if (idx === 1) return abilities.ability2 ?? abilities.ability1;
  if (idx === 2) return abilities.hidden ?? abilities.ability2 ?? abilities.ability1;
  return null;
}
