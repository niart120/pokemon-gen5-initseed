import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { useAppStore } from '@/store/app-store';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { ArrowCounterClockwise, CaretDown, Check, FunnelSimple, Trash } from '@phosphor-icons/react';
import { GenerationExportButton } from './GenerationExportButton';
import { cn } from '@/lib/utils/cn';
import { natureName } from '@/lib/utils/format-display';
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
  generationResultsControlLevelAriaLabel,
  generationResultsControlResetFiltersLabel,
  generationResultsControlStatLabels,
  generationResultsControlTitle,
} from '@/lib/i18n/strings/generation-results-control';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import type { SupportedLocale } from '@/types/i18n';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';

type AppStoreState = ReturnType<typeof useAppStore.getState>;

const STAT_KEYS: StatKey[] = ['hp', 'attack', 'defense', 'specialAttack', 'specialDefense', 'speed'];
const FILTER_FIELD_BASE_CLASS = 'flex flex-col gap-1 grow-0';
const FILTER_FIELD_WIDTHS = {
  species: 'basis-[26ch] min-w-[22ch] max-w-[34ch]',
  ability: 'basis-[20ch] min-w-[14ch] max-w-[24ch]',
  gender: 'basis-[14ch] min-w-[12ch] max-w-[16ch]',
  nature: 'basis-[14ch] min-w-[12ch] max-w-[16ch]',
  shiny: 'basis-[12ch] min-w-[10ch] max-w-[14ch]',
  level: 'basis-[6ch] min-w-[6ch] max-w-[7ch]',
  timer0: 'basis-[6ch] min-w-[6ch] max-w-[7ch]',
  vcount: 'basis-[6ch] min-w-[6ch] max-w-[7ch]',
} as const;
const STAT_FIELD_BASE_CLASS = 'flex flex-col gap-1 grow-0';
const STAT_FIELD_WIDTH = 'basis-[6ch] min-w-[6ch] max-w-[7ch]';

type HexFilterFieldKey = 'timer0Filter' | 'vcountFilter';

const HEX_FILTER_FIELD_CONFIG: Record<HexFilterFieldKey, { id: string; widthClass: string; placeholder: string; maxLength: number; labelKey: 'timer0' | 'vcount'; }> = {
  timer0Filter: {
    id: 'timer0-filter',
    widthClass: FILTER_FIELD_WIDTHS.timer0,
    placeholder: '0000',
    maxLength: 6,
    labelKey: 'timer0',
  },
  vcountFilter: {
    id: 'vcount-filter',
    widthClass: FILTER_FIELD_WIDTHS.vcount,
    placeholder: '00',
    maxLength: 4,
    labelKey: 'vcount',
  },
} as const;

interface AbilityMeta {
  options: Array<{ index: 0 | 1 | 2; label: string }>;
  available: Set<0 | 1 | 2>;
}

interface FilterOptionItem {
  value: string;
  label: string;
  disabled?: boolean;
}

interface FilterPopoverFieldProps {
  id: string;
  label: string;
  placeholder: string;
  selectedValue?: string;
  selectedLabel?: string;
  options: FilterOptionItem[];
  onSelect: (value: string) => void;
  disabled?: boolean;
  searchable?: boolean;
  emptyLabel: string;
  searchPlaceholder?: string;
}

const FilterPopoverField: React.FC<FilterPopoverFieldProps> = ({
  id,
  label,
  placeholder,
  selectedValue,
  selectedLabel,
  options,
  onSelect,
  disabled = false,
  searchable = false,
  emptyLabel,
  searchPlaceholder,
}) => {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    if (disabled && open) {
      setOpen(false);
    }
  }, [disabled, open]);

  const handleSelect = React.useCallback((value: string) => {
    onSelect(value);
    setOpen(false);
  }, [onSelect]);

  const displayText = selectedLabel && selectedValue && selectedValue !== 'any'
    ? selectedLabel
    : placeholder;

  const isActive = Boolean(selectedLabel && selectedValue && selectedValue !== 'any');

  return (
    <div className="flex w-full flex-col gap-1">
      <Label htmlFor={id} className="text-[11px] font-medium text-muted-foreground">{label}</Label>
      <Popover open={open} onOpenChange={(next) => { if (!disabled) setOpen(next); }}>
        <PopoverTrigger asChild>
          <Button
            id={id}
            type="button"
            variant="outline"
            disabled={disabled}
            aria-haspopup="listbox"
            aria-expanded={open}
            className={cn(
              'h-10 w-full justify-between px-3 text-left text-sm font-medium',
              !isActive && 'text-muted-foreground',
              isActive && 'border-primary text-primary',
            )}
          >
            <span className="truncate">{displayText}</span>
            <CaretDown size={14} className="shrink-0 opacity-60" />
          </Button>
        </PopoverTrigger>
        {!disabled && (
          <PopoverContent className="w-[min(320px,90vw)] p-0" align="start">
            <Command>
              {searchable && (
                <CommandInput placeholder={searchPlaceholder ?? ''} />
              )}
              <CommandList>
                <CommandEmpty>{emptyLabel}</CommandEmpty>
                <CommandGroup>
                  {options.map((option) => (
                    <CommandItem
                      key={`${id}-${option.value}`}
                      value={option.label || option.value}
                      disabled={option.disabled}
                      onSelect={() => handleSelect(option.value)}
                    >
                      <span className="truncate">{option.label}</span>
                      {selectedValue === option.value && (
                        <Check size={14} className="ml-2 shrink-0 text-primary" />
                      )}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        )}
      </Popover>
    </div>
  );
};

const HEX_FILTER_INPUT_PATTERN = /^[0-9A-FX\s]*$/;

export const GenerationResultsControlCard: React.FC = () => {
  const locale = useLocale();
  const optionLocale: SupportedLocale = locale;
  const anyLabel = resolveLocaleValue(anyOptionLabel, optionLocale);
  const noAbilitiesLabel = resolveLocaleValue(noAbilitySelectionLabel, optionLocale);
  const noGendersLabel = resolveLocaleValue(noGenderSelectionLabel, optionLocale);
  const cardTitle = resolveLocaleValue(generationResultsControlTitle, optionLocale);
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
  const hexFilterFields = React.useMemo(() => [
    {
      field: 'timer0Filter' as const,
      ...HEX_FILTER_FIELD_CONFIG.timer0Filter,
      label: fieldLabels[HEX_FILTER_FIELD_CONFIG.timer0Filter.labelKey],
      value: filters.timer0Filter ?? '',
    },
    {
      field: 'vcountFilter' as const,
      ...HEX_FILTER_FIELD_CONFIG.vcountFilter,
      label: fieldLabels[HEX_FILTER_FIELD_CONFIG.vcountFilter.labelKey],
      value: filters.vcountFilter ?? '',
    },
  ], [fieldLabels, filters.timer0Filter, filters.vcountFilter]);

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

  const shinyOptions = React.useMemo<FilterOptionItem[]>(() => {
    const labels = resolveLocaleValue(shinyModeOptionLabels, optionLocale);
    return [
      { value: 'all', label: labels.all },
      { value: 'shiny', label: labels.shiny },
      { value: 'star', label: labels.star },
      { value: 'square', label: labels.square },
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

  const commandEmptyLabel = optionLocale === 'ja' ? '候補が見つかりません' : 'No matches found';

  const buildSearchPlaceholder = React.useCallback(
    (targetLabel: string) => {
      if (!targetLabel) {
        return optionLocale === 'ja' ? '検索...' : 'Search...';
      }
      return optionLocale === 'ja'
        ? `${targetLabel}を検索...`
        : `Search ${targetLabel}...`;
    },
    [optionLocale],
  );

  const speciesSearchPlaceholder = buildSearchPlaceholder(fieldLabels.species);
  const natureSearchPlaceholder = buildSearchPlaceholder(fieldLabels.nature);

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

  const handleHexFilterChange = React.useCallback(
    (field: HexFilterFieldKey) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const raw = event.target.value.toUpperCase();
      if (!HEX_FILTER_INPUT_PATTERN.test(raw)) {
        return;
      }
      applyFilters({ [field]: raw });
    },
    [applyFilters],
  );
  const speciesFieldDisabled = pokemonOptions.length === 0;
  const hasPokemonSelection = filters.speciesIds.length > 0;
  const abilityDisabled = !hasPokemonSelection || abilityMeta.options.length === 0;
  const genderDisabled = !hasPokemonSelection || availableGenders.size === 0;
  const abilityPlaceholder = abilityDisabled ? noAbilitiesLabel : anyLabel;
  const genderPlaceholder = genderDisabled ? noGendersLabel : anyLabel;

  const speciesSelectedLabel = React.useMemo(() => {
    if (!filters.speciesIds.length) return undefined;
    const speciesId = filters.speciesIds[0];
    return pokemonOptions.find((option) => option.id === speciesId)?.label;
  }, [filters.speciesIds, pokemonOptions]);

  const abilitySelectedLabel = filters.abilityIndices.length
    ? abilityMeta.options.find((option) => option.index === filters.abilityIndices[0])?.label
    : undefined;

  const genderSelectedLabel = filters.genders.length
    ? genderOptions.find((option) => option.value === filters.genders[0])?.label
    : undefined;

  const natureSelectedLabel = filters.natureIds.length
    ? natureOptions.find((option) => option.id === filters.natureIds[0])?.label
    : undefined;

  const shinySelectedLabel = filters.shinyMode === 'all'
    ? undefined
    : shinyOptions.find((option) => option.value === filters.shinyMode)?.label;

  const speciesFieldOptions = React.useMemo<FilterOptionItem[]>(() => [
    { value: 'any', label: anyLabel, disabled: speciesFieldDisabled },
    ...pokemonOptions.map((option) => ({ value: String(option.id), label: option.label })),
  ], [anyLabel, pokemonOptions, speciesFieldDisabled]);

  const abilityFieldOptions = React.useMemo<FilterOptionItem[]>(() => [
    { value: 'any', label: anyLabel, disabled: abilityDisabled },
    ...abilityMeta.options.map((option) => ({ value: String(option.index), label: option.label })),
  ], [abilityDisabled, abilityMeta.options, anyLabel]);

  const genderFieldOptions = React.useMemo<FilterOptionItem[]>(() => [
    { value: 'any', label: anyLabel, disabled: genderDisabled },
    ...genderOptions.map((option) => ({
      value: option.value,
      label: option.label,
      disabled: !availableGenders.has(option.value),
    })),
  ], [anyLabel, availableGenders, genderDisabled, genderOptions]);

  const natureFieldOptions = React.useMemo<FilterOptionItem[]>(() => [
    { value: 'any', label: anyLabel },
    ...natureOptions.map((option) => ({ value: String(option.id), label: option.label })),
  ], [anyLabel, natureOptions]);

  const statNumericFields = STAT_KEYS.map((stat) => {
    const statLabel = statLabels[stat];
    const range = filters.statRanges[stat];
    const value = range?.min != null ? String(range.min) : '';
    return {
      id: `stat-${stat}`,
      label: statLabel,
      value,
      onChange: handleStatValueChange(stat),
      ariaLabel: formatGenerationResultsControlStatAria(statLabel, optionLocale),
    };
  });

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
      aria-labelledby="gen-results-control-title"
      role="region"
    >
      <form onSubmit={(event) => event.preventDefault()} className="flex flex-col gap-4">
        <fieldset className="space-y-3" aria-labelledby="gf-filters" role="group">
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap gap-3">
              <div className={cn(FILTER_FIELD_BASE_CLASS, FILTER_FIELD_WIDTHS.species)}>
                <FilterPopoverField
                  id="species-select"
                  label={fieldLabels.species}
                  placeholder={anyLabel}
                  selectedValue={filters.speciesIds.length ? String(filters.speciesIds[0]) : 'any'}
                  selectedLabel={speciesSelectedLabel}
                  options={speciesFieldOptions}
                  onSelect={handleSpeciesSelect}
                  disabled={speciesFieldDisabled}
                  searchable
                  emptyLabel={commandEmptyLabel}
                  searchPlaceholder={speciesSearchPlaceholder}
                />
              </div>
              <div className={cn(FILTER_FIELD_BASE_CLASS, FILTER_FIELD_WIDTHS.ability)}>
                <FilterPopoverField
                  id="ability-select"
                  label={fieldLabels.ability}
                  placeholder={abilityPlaceholder}
                  selectedValue={filters.abilityIndices.length ? String(filters.abilityIndices[0]) : 'any'}
                  selectedLabel={abilitySelectedLabel}
                  options={abilityFieldOptions}
                  onSelect={handleAbilitySelect}
                  disabled={abilityDisabled}
                  emptyLabel={commandEmptyLabel}
                />
              </div>
              <div className={cn(FILTER_FIELD_BASE_CLASS, FILTER_FIELD_WIDTHS.gender)}>
                <FilterPopoverField
                  id="gender-select"
                  label={fieldLabels.gender}
                  placeholder={genderPlaceholder}
                  selectedValue={filters.genders.length ? filters.genders[0] : 'any'}
                  selectedLabel={genderSelectedLabel}
                  options={genderFieldOptions}
                  onSelect={handleGenderSelect}
                  disabled={genderDisabled}
                  emptyLabel={commandEmptyLabel}
                />
              </div>
              <div className={cn(FILTER_FIELD_BASE_CLASS, FILTER_FIELD_WIDTHS.nature)}>
                <FilterPopoverField
                  id="nature-select"
                  label={fieldLabels.nature}
                  placeholder={anyLabel}
                  selectedValue={filters.natureIds.length ? String(filters.natureIds[0]) : 'any'}
                  selectedLabel={natureSelectedLabel}
                  options={natureFieldOptions}
                  onSelect={handleNatureSelect}
                  searchable
                  emptyLabel={commandEmptyLabel}
                  searchPlaceholder={natureSearchPlaceholder}
                />
              </div>
              <div className={cn(FILTER_FIELD_BASE_CLASS, FILTER_FIELD_WIDTHS.shiny)}>
                <FilterPopoverField
                  id="shiny-mode"
                  label={fieldLabels.shiny}
                  placeholder={anyLabel}
                  selectedValue={filters.shinyMode}
                  selectedLabel={shinySelectedLabel}
                  options={shinyOptions}
                  onSelect={handleShinyModeChange}
                  emptyLabel={commandEmptyLabel}
                />
              </div>
              <div className={cn(FILTER_FIELD_BASE_CLASS, FILTER_FIELD_WIDTHS.level)}>
                <Label htmlFor="level-filter" className="text-[11px] font-medium text-muted-foreground">
                  {fieldLabels.level}
                </Label>
                <Input
                  id="level-filter"
                  type="text"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={3}
                  value={levelValue}
                  onChange={handleLevelValueChange}
                  className="h-10 w-full px-3 text-right font-mono"
                  placeholder={anyLabel}
                  disabled={!statsAvailable}
                  aria-label={levelAriaLabel}
                />
              </div>
              {hexFilterFields.map((field) => (
                <div key={field.id} className={cn(FILTER_FIELD_BASE_CLASS, field.widthClass)}>
                  <Label htmlFor={field.id} className="text-[11px] font-medium text-muted-foreground">
                    {field.label}
                  </Label>
                  <Input
                    id={field.id}
                    type="text"
                    inputMode="text"
                    value={field.value}
                    onChange={handleHexFilterChange(field.field)}
                    className="h-10 w-full px-3 text-right font-mono uppercase"
                    placeholder={field.placeholder}
                    maxLength={field.maxLength}
                    spellCheck={false}
                    autoComplete="off"
                  />
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-2">
              {statNumericFields.map((field) => (
                <div key={field.id} className={cn(STAT_FIELD_BASE_CLASS, STAT_FIELD_WIDTH)}>
                  <Label htmlFor={field.id} className="text-xs font-medium text-muted-foreground">{field.label}</Label>
                  <Input
                    id={field.id}
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={3}
                    value={field.value}
                    onChange={field.onChange}
                    className="h-10 w-full px-3 text-right font-mono"
                    placeholder={anyLabel}
                    disabled={!statsAvailable}
                    aria-label={field.ariaLabel}
                  />
                </div>
              ))}
            </div>
          </div>
        </fieldset>
      </form>
    </PanelCard>
  );
};
