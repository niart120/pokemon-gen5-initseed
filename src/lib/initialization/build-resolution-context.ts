/**
 * ResolutionContext builder
 *
 * Purpose: Provide a single place to construct resolver context
 * from generated datasets (encounters/species/abilities/gender).
 *
 * This is a minimal, non-invasive builder to enable M1 service refactor.
 * It can be extended later to load richer catalogs.
 */

import type { ResolutionContext } from '@/lib/generation/pokemon-resolver';
import type { ROMVersion } from '@/types/rom';
import type { EncounterTable as UiEncounterTable } from '@/data/encounter-tables';
import { getEncounterTable } from '@/data/encounter-tables';
import type { DomainEncounterType } from '@/types/domain';

// Species/generated dataset adapters
// For M1, we gather only what resolver uses immediately (gender ratios, ability names placeholder)
// The real, species-scoped catalogs can be wired here later without changing service code.
import { getGeneratedSpeciesById } from '@/data/species/generated';

export interface BuildContextOptions {
  version: ROMVersion;
  location: string;
  encounterType: DomainEncounterType;
}

/** Lightweight cache to avoid repeated construction */
const cache = new Map<string, ResolutionContext>();

function cacheKey(opts: BuildContextOptions): string {
  return `${opts.version}__${opts.location}__${opts.encounterType}`;
}

/**
 * Build ResolutionContext from datasets.
 * - encounterTable: UI側loaderの出力を resolver 用 EncounterTable 相当に供給
 * - genderRatios / abilityCatalog: 必要最低限を species/generated から動的参照
 */
export function buildResolutionContext(opts: BuildContextOptions): ResolutionContext {
  const key = cacheKey(opts);
  const hit = cache.get(key);
  if (hit) return hit;

    const table: UiEncounterTable | null = getEncounterTable(
      opts.version,
      opts.location,
      Number(opts.encounterType) // numeric enum
    );

  const ctx: ResolutionContext = {
    encounterTable: table ?? undefined,
    // For now, we don’t prebuild these large maps; resolver helpers receive undefined gracefully.
  // Follow-ups: Precompute gender/abilities maps by scanning slots once.
    genderRatios: undefined,
    abilityCatalog: undefined,
  };

  cache.set(key, ctx);
  return ctx;
}

/** Optional helper: enrich context lazily for a given speciesId */
export function enrichForSpecies(ctx: ResolutionContext, speciesId: number): void {
  const s = getGeneratedSpeciesById(speciesId);
  if (!s) return;

  // genderRatios map
  if (!ctx.genderRatios) ctx.genderRatios = new Map();
  if (!ctx.genderRatios.has(speciesId)) {
    const gender = s.gender;
    if (gender.type === 'genderless') {
  ctx.genderRatios.set(speciesId, { threshold: 0, genderless: true });
    } else if (gender.type === 'fixed') {
      // fixed male/female: map to threshold extremes (male: 0 -> always male, female: 256 -> always female)
      const thr = gender.fixed === 'male' ? 0 : 256;
      ctx.genderRatios.set(speciesId, { threshold: thr, genderless: false });
    } else if (gender.type === 'ratio') {
      // femaleThreshold: value in [0..255], female if gender_value < threshold
      const thr = Math.max(0, Math.min(255, gender.femaleThreshold ?? 127));
  ctx.genderRatios.set(speciesId, { threshold: thr, genderless: false });
    }
  }

  // abilityCatalog map
  if (!ctx.abilityCatalog) ctx.abilityCatalog = new Map();
  if (!ctx.abilityCatalog.has(speciesId)) {
    const names: string[] = [];
    if (s.abilities.ability1) names.push(s.abilities.ability1.names.en);
    if (s.abilities.ability2) names.push(s.abilities.ability2.names.en);
    if (s.abilities.hidden) names.push(s.abilities.hidden.names.en);
    ctx.abilityCatalog.set(speciesId, names);
  }
}
