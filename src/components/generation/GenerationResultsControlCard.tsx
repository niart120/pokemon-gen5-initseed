import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { useAppStore } from '@/store/app-store';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { FunnelSimple, Trash, ArrowCounterClockwise } from '@phosphor-icons/react';
import { GenerationExportButton } from './GenerationExportButton';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { cn } from '@/lib/utils/cn';
import { natureName } from '@/lib/utils/format-display';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import type { ShinyFilterMode, StatRangeFilters } from '@/store/generation-store';
import { selectFilteredSortedResults, selectResolvedResults } from '@/store/generation-store';
import { getGeneratedSpeciesById } from '@/data/species/generated';
import type { StatKey } from '@/lib/utils/pokemon-stats';
import { useLocale } from '@/lib/i18n/locale-context';
import { anyOptionLabel } from '@/lib/i18n/strings/common';
import {
  abilityPreviewJoiner,
  abilitySlotLabels,
  genderOptionLabels,
  noAbilitySelectionLabel,
  noGenderSelectionLabel,
  shinyModeOptionLabels,
} from '@/lib/i18n/strings/generation-filters';
import {
  formatGenerationResultsControlStatAria,
  generationResultsControlAbilityPreviewEllipsis,
  generationResultsControlClearResultsLabel,
  generationResultsControlFieldLabels,
  generationResultsControlFiltersHeading,
  generationResultsControlLevelAriaLabel,
  generationResultsControlResetFiltersLabel,
  generationResultsControlStatLabels,
  generationResultsControlTitle,
} from '@/lib/i18n/strings/generation-results-control';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import type { SupportedLocale } from '@/types/i18n';

type AppStoreState = ReturnType<typeof useAppStore.getState>;

const STAT_KEYS: StatKey[] = ['hp', 'attack', 'defense', 'specialAttack', 'specialDefense', 'speed'];

interface AbilityMeta {
  options: Array<{ index: 0 | 1 | 2; label: string }>;
  available: Set<0 | 1 | 2>;
}

export const GenerationResultsControlCard: React.FC = () => {
  const locale = useLocale();
  const optionLocale: SupportedLocale = locale;
  const anyLabel = resolveLocaleValue(anyOptionLabel, optionLocale);
  const noAbilitiesLabel = resolveLocaleValue(noAbilitySelectionLabel, optionLocale);
  const noGendersLabel = resolveLocaleValue(noGenderSelectionLabel, optionLocale);
  const cardTitle = resolveLocaleValue(generationResultsControlTitle, optionLocale);
  const filtersHeading = resolveLocaleValue(generationResultsControlFiltersHeading, optionLocale);
  const resetFiltersLabel = resolveLocaleValue(generationResultsControlResetFiltersLabel, optionLocale);
  const clearResultsLabel = resolveLocaleValue(generationResultsControlClearResultsLabel, optionLocale);
  const fieldLabels = resolveLocaleValue(generationResultsControlFieldLabels, optionLocale);
  const statLabels = resolveLocaleValue(generationResultsControlStatLabels, optionLocale);
  const levelAriaLabel = resolveLocaleValue(generationResultsControlLevelAriaLabel, optionLocale);
  const abilityPreviewEllipsis = resolveLocaleValue(generationResultsControlAbilityPreviewEllipsis, optionLocale);
  const filters = useAppStore((state) => state.filters);
  const applyFilters = useAppStore((state) => state.applyFilters);
  const resetGenerationFilters = useAppStore((state) => state.resetGenerationFilters);
  const results = useAppStore((state) => state.results);
  const clearResults = useAppStore((state) => state.clearResults);
  const encounterTable = useAppStore((state) => state.encounterTable);
  const genderRatios = useAppStore((state) => state.genderRatios);
  const abilityCatalog = useAppStore((state) => state.abilityCatalog);
  const version = useAppStore((state) => (state.params?.version ?? state.draftParams.version ?? 'B') as 'B' | 'W' | 'B2' | 'W2');
  const baseSeed = useAppStore((state) => {
    if (state.params?.baseSeed !== undefined) return state.params.baseSeed;
    const hex = state.draftParams.baseSeedHex;
    if (typeof hex === 'string') {
      const normalized = hex.trim();
      if (normalized !== '') {
        try {
          return BigInt('0x' + normalized.replace(/^0x/i, ''));
        } catch {
          return undefined;
        }
      }
    }
    return undefined;
  });
  const statsAvailable = useAppStore((state) => Boolean(state.params?.baseSeed));

  const resolvedResults = useAppStore((state: AppStoreState) => selectResolvedResults(state));
  const filteredRawRows = useAppStore((state: AppStoreState) => selectFilteredSortedResults(state, locale));

  const { isStack } = useResponsiveLayout();

  const natureOptions = React.useMemo(
    () => Array.from({ length: 25 }, (_, id) => ({ id, label: natureName(id, optionLocale) })),
    [optionLocale],
  );

  const pokemonOptions = React.useMemo(() => {
    const speciesIds = new Set<number>();
    if (encounterTable?.slots?.length) {
      for (const slot of encounterTable.slots) {
        if (slot?.speciesId) speciesIds.add(slot.speciesId);
      }
    }
    if (speciesIds.size === 0) {
      for (const entry of resolvedResults) {
        if (entry?.speciesId) speciesIds.add(entry.speciesId);
      }
    }
    const formatter = optionLocale;
    return Array.from(speciesIds)
      .map((id) => {
        const species = getGeneratedSpeciesById(id);
        const fallback = species?.names.en ?? `#${id}`;
        const name = species?.names[optionLocale] ?? fallback;
        return { id, label: name };
      })
      .sort((a, b) => a.label.localeCompare(b.label, formatter));
  }, [encounterTable, resolvedResults, optionLocale]);

  const computeAbilityMeta = React.useCallback((speciesIds: number[]): AbilityMeta => {
    const available = new Set<0 | 1 | 2>();
    const labelBuckets: Record<0 | 1 | 2, Set<string>> = {
      0: new Set(),
      1: new Set(),
      2: new Set(),
    };

    for (const speciesId of speciesIds) {
      const species = getGeneratedSpeciesById(speciesId);
      if (!species) continue;
      const { ability1, ability2, hidden } = species.abilities;
      if (ability1) {
        available.add(0);
        labelBuckets[0].add(ability1.names[optionLocale] ?? ability1.names.en);
      }
      if (ability2) {
        available.add(1);
        labelBuckets[1].add(ability2.names[optionLocale] ?? ability2.names.en);
      }
      if (hidden) {
        available.add(2);
        labelBuckets[2].add(hidden.names[optionLocale] ?? hidden.names.en);
      }
    }

    const joiner = resolveLocaleValue(abilityPreviewJoiner, optionLocale);
    const slotLabels = resolveLocaleValue(abilitySlotLabels, optionLocale);

    const options: Array<{ index: 0 | 1 | 2; label: string }> = [];
    ([0, 1, 2] as const).forEach((slot) => {
      if (!available.has(slot)) return;
      const names = Array.from(labelBuckets[slot]).filter(Boolean);
      const preview = names.slice(0, 3).join(joiner);
      const suffix = names.length > 3 ? `${preview}${abilityPreviewEllipsis}` : preview;
      const slotTitle = slotLabels[slot];
      const label = suffix ? `${slotTitle} (${suffix})` : slotTitle;
      options.push({ index: slot, label });
    });

    return { options, available };
  }, [abilityPreviewEllipsis, optionLocale]);
  const abilityMeta = React.useMemo(
    () => computeAbilityMeta(filters.speciesIds),
    [computeAbilityMeta, filters.speciesIds],
  );
  const levelValue = filters.levelRange?.min != null ? String(filters.levelRange.min) : '';

  const computeAvailableGenders = React.useCallback((speciesIds: number[]): Set<'M' | 'F' | 'N'> => {
    const genders = new Set<'M' | 'F' | 'N'>();
    for (const speciesId of speciesIds) {
      const species = getGeneratedSpeciesById(speciesId);
      if (!species) continue;
      const info = species.gender;
      if (info.type === 'genderless') {
        genders.add('N');
        continue;
      }
      if (info.type === 'fixed') {
        genders.add(info.fixed === 'male' ? 'M' : 'F');
        continue;
      }
      if (info.type === 'ratio') {
        const threshold = info.femaleThreshold ?? 127;
        if (threshold <= 0) {
          genders.add('M');
        } else if (threshold >= 256) {
          genders.add('F');
        } else {
          genders.add('M');
          genders.add('F');
        }
        continue;
      }
      genders.add('M');
      genders.add('F');
    }
    return genders;
  }, []);

  const availableGenders = React.useMemo(
    () => computeAvailableGenders(filters.speciesIds),
    [computeAvailableGenders, filters.speciesIds],
  );

  const shinyOptions = React.useMemo(() => {
    const labels = resolveLocaleValue(shinyModeOptionLabels, optionLocale);
    return [
      { value: 'all', label: labels.all },
      { value: 'shiny', label: labels.shiny },
      { value: 'non-shiny', label: labels['non-shiny'] },
    ];
  }, [optionLocale]);

  const genderOptions = React.useMemo(() => {
    const labels = resolveLocaleValue(genderOptionLabels, optionLocale);
    return [
      { value: 'M' as const, label: labels.M },
      { value: 'F' as const, label: labels.F },
      { value: 'N' as const, label: labels.N },
    ];
  }, [optionLocale]);

  const handleShinyModeChange = React.useCallback(
    (value: string) => {
      applyFilters({ shinyMode: value as ShinyFilterMode });
    },
    [applyFilters],
  );

  const handleSpeciesSelect = React.useCallback(
    (value: string) => {
      if (value === 'any') {
        applyFilters({ speciesIds: [], abilityIndices: [], genders: [] });
        return;
      }

      const speciesId = Number(value);
      if (Number.isNaN(speciesId)) {
        applyFilters({ speciesIds: [], abilityIndices: [], genders: [] });
        return;
      }

      const selectedIds = [speciesId];
      const nextAbilityMeta = computeAbilityMeta(selectedIds);
      const nextGenderSet = computeAvailableGenders(selectedIds);
      const abilitySelection = filters.abilityIndices
        .filter((idx) => nextAbilityMeta.available.has(idx))
        .slice(0, 1) as (0 | 1 | 2)[];
      const genderSelection = filters.genders
        .filter((g) => nextGenderSet.has(g))
        .slice(0, 1) as ('M' | 'F' | 'N')[];

      applyFilters({
        speciesIds: selectedIds,
        abilityIndices: abilitySelection,
        genders: genderSelection,
      });
    },
    [applyFilters, filters.abilityIndices, filters.genders, computeAbilityMeta, computeAvailableGenders],
  );

  const handleNatureSelect = React.useCallback(
    (value: string) => {
      if (value === 'any') {
        applyFilters({ natureIds: [] });
        return;
      }

      const natureId = Number(value);
      if (Number.isNaN(natureId)) {
        applyFilters({ natureIds: [] });
        return;
      }

      applyFilters({ natureIds: [natureId] });
    },
    [applyFilters],
  );

  const handleAbilitySelect = React.useCallback(
    (value: string) => {
      if (value === 'any') {
        applyFilters({ abilityIndices: [] });
        return;
      }

      const index = Number(value) as 0 | 1 | 2;
      if (!abilityMeta.available.has(index)) {
        applyFilters({ abilityIndices: [] });
        return;
      }

      applyFilters({ abilityIndices: [index] });
    },
    [abilityMeta.available, applyFilters],
  );

  const handleGenderSelect = React.useCallback(
    (value: string) => {
      if (value === 'any') {
        applyFilters({ genders: [] });
        return;
      }

      const gender = value as 'M' | 'F' | 'N';
      if (!availableGenders.has(gender)) {
        applyFilters({ genders: [] });
        return;
      }

      applyFilters({ genders: [gender] });
    },
    [applyFilters, availableGenders],
  );

  const handleStatValueChange = React.useCallback(
    (stat: StatKey) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const raw = event.target.value.trim();
      if (raw !== '' && !/^[0-9]+$/.test(raw)) {
        return;
      }

      const cloned: StatRangeFilters = {};
      for (const key of STAT_KEYS) {
        const range = filters.statRanges[key];
        if (range && (range.min != null || range.max != null)) {
          cloned[key] = { ...range };
        }
      }

      if (raw === '') {
        if (cloned[stat]) {
          delete cloned[stat];
        }
      } else {
        const value = Number(raw);
        cloned[stat] = { min: value, max: value };
      }

      applyFilters({ statRanges: cloned });
    },
    [applyFilters, filters.statRanges],
  );

  const handleLevelValueChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const raw = event.target.value.trim();
      if (raw !== '' && !/^[0-9]+$/.test(raw)) {
        return;
      }

      if (raw === '') {
        if (filters.levelRange) {
          applyFilters({ levelRange: undefined });
        }
        return;
      }

      const value = Number(raw);
      applyFilters({ levelRange: { min: value, max: value } });
    },
    [applyFilters, filters.levelRange],
  );

  const hasPokemonSelection = filters.speciesIds.length > 0;
  const abilityDisabled = !hasPokemonSelection || abilityMeta.options.length === 0;
  const genderDisabled = !hasPokemonSelection || availableGenders.size === 0;

  return (
    <PanelCard
      icon={<FunnelSimple size={20} className="opacity-80" />}
      title={<span id="gen-results-control-title">{cardTitle}</span>}
      headerActions={
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={resetGenerationFilters}
            className="gap-1"
          >
            <ArrowCounterClockwise size={14} />
            {resetFiltersLabel}
          </Button>
          <GenerationExportButton
            rows={filteredRawRows}
            encounterTable={encounterTable}
            genderRatios={genderRatios}
            abilityCatalog={abilityCatalog}
            version={version}
            baseSeed={baseSeed}
            disabled={filteredRawRows.length === 0}
          />
          <Button
            size="sm"
            variant="destructive"
            disabled={!results.length}
            onClick={clearResults}
            className="gap-1"
          >
            <Trash size={14} />
            {clearResultsLabel}
          </Button>
        </div>
      }
      className="flex flex-col"
      fullHeight={false}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="gen-results-control-title"
      role="region"
    >
      <form onSubmit={(event) => event.preventDefault()} className="flex flex-col gap-3">
        <fieldset className="space-y-2" aria-labelledby="gf-filters" role="group">
          <div id="gf-filters" className="text-[10px] font-medium tracking-wide uppercase text-muted-foreground">{filtersHeading}</div>
          <div className={cn('grid gap-3 items-start', isStack ? 'grid-cols-1' : 'grid-cols-2')}>
            <div className={cn('grid gap-3 items-start', isStack ? 'grid-cols-3' : 'grid-cols-6')}>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="species-select" className="text-[11px] font-medium text-muted-foreground">{fieldLabels.species}</Label>
                <Select
                  value={filters.speciesIds.length ? String(filters.speciesIds[0]) : 'any'}
                  onValueChange={handleSpeciesSelect}
                  disabled={pokemonOptions.length === 0}
                >
                  <SelectTrigger id="species-select" className="h-9">
                    <SelectValue placeholder={anyLabel} />
                  </SelectTrigger>
                  <SelectContent className="max-h-60">
                    <SelectItem value="any">{anyLabel}</SelectItem>
                    {pokemonOptions.map((option) => (
                      <SelectItem key={option.id} value={String(option.id)}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="ability-select" className="text-[11px] font-medium text-muted-foreground">{fieldLabels.ability}</Label>
                <Select
                  value={filters.abilityIndices.length ? String(filters.abilityIndices[0]) : 'any'}
                  onValueChange={handleAbilitySelect}
                  disabled={abilityDisabled}
                >
                  <SelectTrigger id="ability-select" className="h-9">
                    <SelectValue placeholder={abilityDisabled ? noAbilitiesLabel : anyLabel} />
                  </SelectTrigger>
                  <SelectContent className="max-h-60">
                    <SelectItem value="any" disabled={abilityDisabled}>{anyLabel}</SelectItem>
                    {abilityMeta.options.map((option) => (
                      <SelectItem key={option.index} value={String(option.index)}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="gender-select" className="text-[11px] font-medium text-muted-foreground">{fieldLabels.gender}</Label>
                <Select
                  value={filters.genders.length ? filters.genders[0] : 'any'}
                  onValueChange={handleGenderSelect}
                  disabled={genderDisabled}
                >
                  <SelectTrigger id="gender-select" className="h-9">
                    <SelectValue placeholder={genderDisabled ? noGendersLabel : anyLabel} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="any" disabled={genderDisabled}>{anyLabel}</SelectItem>
                    {genderOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value} disabled={!availableGenders.has(option.value)}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="nature-select" className="text-[11px] font-medium text-muted-foreground">{fieldLabels.nature}</Label>
                <Select value={filters.natureIds.length ? String(filters.natureIds[0]) : 'any'} onValueChange={handleNatureSelect}>
                  <SelectTrigger id="nature-select" className="h-9">
                    <SelectValue placeholder={anyLabel} />
                  </SelectTrigger>
                  <SelectContent className="max-h-60">
                    <SelectItem value="any">{anyLabel}</SelectItem>
                    {natureOptions.map((option) => (
                      <SelectItem key={option.id} value={String(option.id)}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="shiny-mode" className="text-[11px] font-medium text-muted-foreground">{fieldLabels.shiny}</Label>
                <Select value={filters.shinyMode} onValueChange={handleShinyModeChange}>
                  <SelectTrigger id="shiny-mode" className="h-9">
                    <SelectValue placeholder={anyLabel} />
                  </SelectTrigger>
                  <SelectContent>
                    {shinyOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-col gap-1 sm:w-auto">
                <Label htmlFor="level-filter" className="text-[11px] font-medium text-muted-foreground">{fieldLabels.level}</Label>
                <Input
                  id="level-filter"
                  type="text"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={3}
                  value={levelValue}
                  onChange={handleLevelValueChange}
                  className="h-9 w-16 px-2 text-right text-xs font-mono sm:w-16"
                  placeholder={anyLabel}
                  disabled={!statsAvailable}
                  aria-label={levelAriaLabel}
                />
              </div>
            </div>
            <div className={`grid grid-cols-6 gap-1 items-end`}>
              {STAT_KEYS.map((stat) => {
                const range = filters.statRanges[stat];
                const value = range?.min != null ? String(range.min) : '';
                const statLabel = statLabels[stat];
                return (
                  <div key={stat} className="flex w-full flex-col gap-1 sm:w-auto">
                    <Label htmlFor={`stat-${stat}`} className="text-[11px] font-medium text-muted-foreground">{statLabel}</Label>
                    <Input
                      id={`stat-${stat}`}
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      maxLength={3}
                      value={value}
                      onChange={handleStatValueChange(stat)}
                      className="h-9 w-full px-2 text-right text-xs font-mono sm:w-16"
                      placeholder={anyLabel}
                      disabled={!statsAvailable}
                      aria-label={formatGenerationResultsControlStatAria(statLabel, optionLocale)}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </fieldset>
      </form>
    </PanelCard>
  );
};
