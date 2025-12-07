import type { GenerationSlice, GenerationFilters } from '@/store/generation-store';
import type { GenerationResult } from '@/types/generation';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import type { ResolvedPokemonData, UiReadyPokemonData } from '@/types/pokemon-resolved';
import { resolveBatch, toUiReadyPokemon } from '@/lib/generation/pokemon-resolver';
import { buildResolutionContextFromSources } from '@/lib/initialization/build-resolution-context';
import { formatKeyInputForDisplay, KEY_INPUT_DISPLAY_FALLBACK } from '@/lib/utils/key-input';
import { DomainShinyType } from '@/types/domain';
import { parseHexFilterValue } from '@/lib/utils/hex-filter';

export type FilteredRowsCache = {
  resultsRef: GenerationResult[];
  resolvedRef: ResolvedPokemonData[];
  filtersRef: GenerationFilters;
  encounterTableRef?: EncounterTable;
  genderRatiosRef?: Map<number, GenderRatio>;
  abilityCatalogRef?: Map<number, string[]>;
  locale: 'ja' | 'en';
  baseSeedRef?: bigint;
  versionRef: 'B' | 'W' | 'B2' | 'W2';
  ui: UiReadyPokemonData[];
  raw: GenerationResult[];
} | null;

let _filteredRowsCache: FilteredRowsCache = null;

function computeFilteredRowsCache(s: GenerationSlice, locale: 'ja' | 'en'): NonNullable<FilteredRowsCache> {
  const {
    results,
    filters,
    encounterTable,
    genderRatios,
    abilityCatalog,
  } = s as GenerationSlice & {
    encounterTable?: EncounterTable;
    genderRatios?: Map<number, GenderRatio>;
    abilityCatalog?: Map<number, string[]>;
    locale?: 'ja' | 'en';
  };

  const cache = _filteredRowsCache;
  const version = (s.params?.version ?? s.draftParams.version ?? 'B') as 'B' | 'W' | 'B2' | 'W2';
  const baseSeed = s.params?.baseSeed;
  const resolvedFromState = s.resolvedResults ?? [];
  const requiresFallback = results.length > 0 && resolvedFromState.length !== results.length;

  if (
    cache &&
    cache.resultsRef === results &&
    cache.filtersRef === filters &&
    cache.encounterTableRef === encounterTable &&
    cache.genderRatiosRef === genderRatios &&
    cache.abilityCatalogRef === abilityCatalog &&
    cache.locale === locale &&
    cache.versionRef === version &&
    cache.baseSeedRef === baseSeed &&
    (
      (!requiresFallback && cache.resolvedRef === resolvedFromState) ||
      (requiresFallback && cache.resolvedRef.length === results.length)
    )
  ) {
    return cache;
  }

  let resolved = resolvedFromState;

  if (requiresFallback) {
    const context = buildResolutionContextFromSources({ encounterTable, genderRatios, abilityCatalog }) ?? {};
    resolved = resolveBatch(results, context);
  }

  const shinyMode = filters.shinyMode;
  const natureSet = filters.natureIds.length ? new Set(filters.natureIds) : null;
  const speciesSet = filters.speciesIds.length ? new Set(filters.speciesIds) : null;
  const abilitySet = speciesSet && filters.abilityIndices.length ? new Set(filters.abilityIndices) : null;
  const genderSet = speciesSet && filters.genders.length ? new Set(filters.genders) : null;
  const statKeys: Array<keyof GenerationFilters['statRanges']> = ['hp', 'attack', 'defense', 'specialAttack', 'specialDefense', 'speed'];
  const hasStatFilters = statKeys.some((key) => {
    const range = filters.statRanges[key];
    return !!range && (range.min != null || range.max != null);
  });
  const levelRange = filters.levelRange;
  const hasLevelFilter = Boolean(levelRange && (levelRange.min != null || levelRange.max != null));

  const timer0FilterValue = parseHexFilterValue(filters.timer0Filter, { maxValue: 0xFFFF });
  const vcountFilterValue = parseHexFilterValue(filters.vcountFilter, { maxValue: 0xFF });
  const hasTimer0Filter = timer0FilterValue != null;
  const hasVcountFilter = vcountFilterValue != null;

  const entries: Array<{ raw: GenerationResult; ui: UiReadyPokemonData }> = [];

  for (let i = 0; i < results.length; i += 1) {
    const raw = results[i];
    const resolvedData = resolved[i];
    if (!resolvedData) continue;
    const perRowBaseSeed = raw.baseSeed ?? baseSeed;
    const uiData = {
      ...toUiReadyPokemon(resolvedData, { locale, version, baseSeed: perRowBaseSeed }),
      reportNeedleDirection: raw.report_needle_direction,
    } as UiReadyPokemonData;

    const shinyType = uiData.shinyType ?? DomainShinyType.Normal;
    if (shinyMode === 'shiny' && shinyType === DomainShinyType.Normal) continue;
    if (shinyMode === 'non-shiny' && shinyType !== DomainShinyType.Normal) continue;
    if (shinyMode === 'star' && shinyType !== DomainShinyType.Star) continue;
    if (shinyMode === 'square' && shinyType !== DomainShinyType.Square) continue;
    if (natureSet && !natureSet.has(uiData.natureId)) continue;

    if (speciesSet) {
      const speciesId = uiData.speciesId;
      if (!speciesId || !speciesSet.has(speciesId)) continue;
    }

    if (abilitySet) {
      const abilityIndex = uiData.abilityIndex;
      if (abilityIndex == null || !abilitySet.has(abilityIndex)) continue;
    }

    if (genderSet) {
      const gender = uiData.genderCode;
      if (!gender || !genderSet.has(gender)) continue;
    }

    if (hasLevelFilter) {
      const level = uiData.level;
      if (level == null) continue;
      if (levelRange?.min != null && level < levelRange.min) continue;
      if (levelRange?.max != null && level > levelRange.max) continue;
    }

    if (hasStatFilters) {
      const stats = uiData.stats;
      if (!stats) continue;
      let ok = true;
      for (const key of statKeys) {
        const range = filters.statRanges[key];
        if (!range) continue;
        const value = stats[key as keyof typeof stats];
        if (range.min != null && value < range.min) {
          ok = false;
          break;
        }
        if (range.max != null && value > range.max) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
    }

    if (hasTimer0Filter) {
      const value = raw.timer0;
      if (typeof value !== 'number' || value !== timer0FilterValue) continue;
    }

    if (hasVcountFilter) {
      const value = raw.vcount;
      if (typeof value !== 'number' || value !== vcountFilterValue) continue;
    }

    if (raw.seedSourceMode) {
      uiData.seedSourceMode = raw.seedSourceMode;
      uiData.derivedSeedIndex = raw.derivedSeedIndex;
      uiData.seedSourceSeedHex = raw.seedSourceSeedHex;
    } else {
      uiData.seedSourceMode = undefined;
      uiData.derivedSeedIndex = undefined;
      uiData.seedSourceSeedHex = undefined;
    }
    uiData.timer0 = raw.timer0;
    uiData.vcount = raw.vcount;
    uiData.bootTimestampIso = raw.bootTimestampIso;
    const keyDisplay = formatKeyInputForDisplay(null, raw.keyInputNames);
    if (keyDisplay !== KEY_INPUT_DISPLAY_FALLBACK) {
      uiData.keyInputNames = raw.keyInputNames;
      uiData.keyInputDisplay = keyDisplay;
    } else {
      uiData.keyInputNames = undefined;
      uiData.keyInputDisplay = undefined;
    }

    entries.push({ raw, ui: uiData });
  }

  const field = filters.sortField ?? 'advance';
  const order = filters.sortOrder === 'desc' ? -1 : 1;

  entries.sort((a, b) => {
    const au = a.ui;
    const bu = b.ui;
    let av: number;
    let bv: number;
    switch (field) {
      case 'pid':
        av = au.pid >>> 0;
        bv = bu.pid >>> 0;
        break;
      case 'nature':
        av = au.natureId;
        bv = bu.natureId;
        break;
      case 'shiny':
        av = au.shinyType;
        bv = bu.shinyType;
        break;
      case 'species':
        av = au.speciesId ?? Number.MAX_SAFE_INTEGER;
        bv = bu.speciesId ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'ability':
        av = au.abilityIndex ?? Number.MAX_SAFE_INTEGER;
        bv = bu.abilityIndex ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'level':
        av = au.level ?? Number.MAX_SAFE_INTEGER;
        bv = bu.level ?? Number.MAX_SAFE_INTEGER;
        break;
      case 'advance':
      default:
        av = au.advance;
        bv = bu.advance;
        break;
    }
    if (av < bv) return -1 * order;
    if (av > bv) return 1 * order;
    return 0;
  });

  const ui = entries.map(entry => entry.ui);
  const raw = entries.map(entry => entry.raw);
  const nextCache: NonNullable<FilteredRowsCache> = {
    resultsRef: results,
    resolvedRef: resolved,
    filtersRef: filters,
    encounterTableRef: encounterTable,
    genderRatiosRef: genderRatios,
    abilityCatalogRef: abilityCatalog,
    locale,
    versionRef: version,
    baseSeedRef: baseSeed,
    ui,
    raw,
  };
  _filteredRowsCache = nextCache;
  return nextCache;
}

export const selectFilteredDisplayRows = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja'): UiReadyPokemonData[] => {
  return computeFilteredRowsCache(s, locale).ui;
};

export const selectFilteredSortedResults = (s: GenerationSlice, locale: 'ja' | 'en' = 'ja') => {
  return computeFilteredRowsCache(s, locale).raw;
};

export const selectResolvedResults = (s: GenerationSlice): ResolvedPokemonData[] => {
  return s.resolvedResults ?? [];
};
