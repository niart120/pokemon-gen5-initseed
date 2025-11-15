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
import type { GenerationFilters, ShinyFilterMode, StatRangeFilters } from '@/store/generation-store';
import { selectResolvedResults } from '@/store/generation-store';
import { getGeneratedSpeciesById } from '@/data/species/generated';
import type { StatKey } from '@/lib/utils/pokemon-stats';

type AppStoreState = ReturnType<typeof useAppStore.getState>;

const SHINY_MODE_OPTIONS: Array<{ value: ShinyFilterMode; label: string }> = [
  { value: 'all', label: 'Any' },
  { value: 'shiny', label: 'Shiny Only' },
  { value: 'non-shiny', label: 'Non-shiny Only' },
];

const GENDER_OPTIONS: Array<{ value: 'M' | 'F' | 'N'; label: string }> = [
  { value: 'M', label: 'Male' },
  { value: 'F', label: 'Female' },
  { value: 'N', label: 'Genderless' },
];

const ABILITY_SLOT_LABELS: Record<0 | 1 | 2, string> = {
  0: 'Primary Ability',
  1: 'Secondary Ability',
  2: 'Hidden Ability',
};

const STAT_KEYS: StatKey[] = ['hp', 'attack', 'defense', 'specialAttack', 'specialDefense', 'speed'];
const STAT_LABELS: Record<StatKey, string> = {
  hp: 'HP',
  attack: 'Atk',
  defense: 'Def',
  specialAttack: 'SpA',
  specialDefense: 'SpD',
  speed: 'Spe',
};

interface AbilityMeta {
  options: Array<{ index: 0 | 1 | 2; label: string }>;
  available: Set<0 | 1 | 2>;
}

export const GenerationResultsControlCard: React.FC = () => {
  const filters = useAppStore((state) => state.filters);
  const applyFilters = useAppStore((state) => state.applyFilters);
  const resetGenerationFilters = useAppStore((state) => state.resetGenerationFilters);
  const results = useAppStore((state) => state.results);
  const clearResults = useAppStore((state) => state.clearResults);
  const encounterTable = useAppStore((state) => state.encounterTable);
  const statsAvailable = useAppStore((state) => Boolean(state.params?.baseSeed));

  const resolvedResults = useAppStore((state: AppStoreState) => selectResolvedResults(state));

  const optionLocale = 'en' as const;
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
        const name = species ? species.names.en : `#${id}`;
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
        labelBuckets[0].add(ability1.names.en);
      }
      if (ability2) {
        available.add(1);
        labelBuckets[1].add(ability2.names.en);
      }
      if (hidden) {
        available.add(2);
        labelBuckets[2].add(hidden.names.en);
      }
    }

    const options: Array<{ index: 0 | 1 | 2; label: string }> = [];
    ([0, 1, 2] as const).forEach((slot) => {
      if (!available.has(slot)) return;
      const names = Array.from(labelBuckets[slot]).filter(Boolean);
      const preview = names.slice(0, 3).join(', ');
      const suffix = names.length > 3 ? `${preview}…` : preview;
      const label = suffix ? `${ABILITY_SLOT_LABELS[slot]} (${suffix})` : ABILITY_SLOT_LABELS[slot];
      options.push({ index: slot, label });
    });

    return { options, available };
  }, [optionLocale]);
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
      title={<span id="gen-results-control-title">Results Control</span>}
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
            Reset Filters
          </Button>
          <GenerationExportButton results={results} disabled={!results.length} />
          <Button
            size="sm"
            variant="destructive"
            disabled={!results.length}
            onClick={clearResults}
            className="gap-1"
          >
            <Trash size={14} />
            Clear Results
          </Button>
        </div>
      }
      className="flex flex-col"
      fullHeight={false}
      scrollMode={isStack ? 'parent' : 'content'}
      contentClassName="space-y-3 text-xs"
      aria-labelledby="gen-results-control-title"
      role="region"
    >
      <form onSubmit={(event) => event.preventDefault()} className="flex flex-col gap-3">
        <fieldset className="space-y-2" aria-labelledby="gf-filters" role="group">
          <div id="gf-filters" className="text-[10px] font-medium tracking-wide uppercase text-muted-foreground">Filters</div>
          <div className={cn('grid gap-3 items-start', isStack ? 'grid-cols-1' : 'grid-cols-2')}>
            <div className={cn('flex w-full gap-3', isStack ? 'flex-col' : 'flex-wrap items-end')}>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="species-select" className="text-[11px] font-medium text-muted-foreground">Species</Label>
                <Select
                  value={filters.speciesIds.length ? String(filters.speciesIds[0]) : 'any'}
                  onValueChange={handleSpeciesSelect}
                  disabled={pokemonOptions.length === 0}
                >
                  <SelectTrigger id="species-select" className="h-9">
                    <SelectValue placeholder="Any" />
                  </SelectTrigger>
                  <SelectContent className="max-h-60">
                    <SelectItem value="any">Any</SelectItem>
                    {pokemonOptions.map((option) => (
                      <SelectItem key={option.id} value={String(option.id)}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="ability-select" className="text-[11px] font-medium text-muted-foreground">Ability</Label>
                <Select
                  value={filters.abilityIndices.length ? String(filters.abilityIndices[0]) : 'any'}
                  onValueChange={handleAbilitySelect}
                  disabled={abilityDisabled}
                >
                  <SelectTrigger id="ability-select" className="h-9">
                    <SelectValue placeholder={abilityDisabled ? 'No abilities' : 'Any'} />
                  </SelectTrigger>
                  <SelectContent className="max-h-60">
                    <SelectItem value="any" disabled={abilityDisabled}>Any</SelectItem>
                    {abilityMeta.options.map((option) => (
                      <SelectItem key={option.index} value={String(option.index)}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="gender-select" className="text-[11px] font-medium text-muted-foreground">Gender</Label>
                <Select
                  value={filters.genders.length ? filters.genders[0] : 'any'}
                  onValueChange={handleGenderSelect}
                  disabled={genderDisabled}
                >
                  <SelectTrigger id="gender-select" className="h-9">
                    <SelectValue placeholder={genderDisabled ? 'No genders' : 'Any'} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="any" disabled={genderDisabled}>Any</SelectItem>
                    {GENDER_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value} disabled={!availableGenders.has(option.value)}>
                        {option.label}
                      </SelectItem>
                    ))}
                </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="nature-select" className="text-[11px] font-medium text-muted-foreground">Nature</Label>
                <Select value={filters.natureIds.length ? String(filters.natureIds[0]) : 'any'} onValueChange={handleNatureSelect}>
                  <SelectTrigger id="nature-select" className="h-9">
                    <SelectValue placeholder="Any" />
                  </SelectTrigger>
                  <SelectContent className="max-h-60">
                    <SelectItem value="any">Any</SelectItem>
                    {natureOptions.map((option) => (
                      <SelectItem key={option.id} value={String(option.id)}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="shiny-mode" className="text-[11px] font-medium text-muted-foreground">Shiny</Label>
                <Select value={filters.shinyMode} onValueChange={handleShinyModeChange}>
                  <SelectTrigger id="shiny-mode" className="h-9">
                    <SelectValue placeholder="Any" />
                  </SelectTrigger>
                  <SelectContent>
                    {SHINY_MODE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className={cn('flex w-full gap-3', isStack ? 'flex-col' : 'flex-wrap items-end')}>
              <div className="flex w-full flex-col gap-1 sm:w-auto">
                <Label htmlFor="level-filter" className="text-[11px] font-medium text-muted-foreground">Lv</Label>
                <Input
                  id="level-filter"
                  type="text"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={3}
                  value={levelValue}
                  onChange={handleLevelValueChange}
                  className="h-9 w-full px-2 text-right text-xs font-mono sm:w-16"
                  placeholder="Any"
                  disabled={!statsAvailable}
                  aria-label="Level exact value"
                />
              </div>
              {STAT_KEYS.map((stat) => {
                const range = filters.statRanges[stat];
                const value = range?.min != null ? String(range.min) : '';
                return (
                  <div key={stat} className="flex w-full flex-col gap-1 sm:w-auto">
                    <Label htmlFor={`stat-${stat}`} className="text-[11px] font-medium text-muted-foreground">{STAT_LABELS[stat]}</Label>
                    <Input
                      id={`stat-${stat}`}
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      maxLength={3}
                      value={value}
                      onChange={handleStatValueChange(stat)}
                      className="h-9 w-full px-2 text-right text-xs font-mono sm:w-16"
                      placeholder="Any"
                      disabled={!statsAvailable}
                      aria-label={`${STAT_LABELS[stat]} exact value`}
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
