/**
 * GenerationFilterCard
 * 個体生成結果フィルターカード（EggFilterCardと同様の2×nグリッド + HABCDS縦配置レイアウト）
 */

import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { useAppStore } from '@/store/app-store';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { ArrowCounterClockwise, CaretDown, Check, Funnel } from '@phosphor-icons/react';
import { cn } from '@/lib/utils/cn';
import { natureName } from '@/lib/utils/format-display';
import { Input } from '@/components/ui/input';
import type { ShinyFilterMode, StatRangeFilters } from '@/store/generation-store';
import { selectResolvedResults } from '@/store/generation-store';
import { getGeneratedSpeciesById } from '@/data/species/generated';
import type { StatKey } from '@/lib/utils/pokemon-stats';
import { useLocale } from '@/lib/i18n/locale-context';
import { useResponsiveLayout } from '@/hooks/use-mobile';
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

type HexFilterFieldKey = 'timer0Filter' | 'vcountFilter';

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
      <Label htmlFor={id} className="text-xs">{label}</Label>
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
              'h-8 w-full justify-between px-2 text-left text-xs font-medium',
              !isActive && 'text-muted-foreground',
              isActive && 'border-primary text-primary',
            )}
          >
            <span className="truncate">{displayText}</span>
            <CaretDown size={12} className="shrink-0 opacity-60" />
          </Button>
        </PopoverTrigger>
        {!disabled && (
          <PopoverContent className="w-[min(280px,90vw)] p-0" align="start">
            <Command>
              {searchable && (
                <CommandInput placeholder={searchPlaceholder ?? ''} className="text-xs" />
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
                      className="text-xs"
                    >
                      <span className="truncate">{option.label}</span>
                      {selectedValue === option.value && (
                        <Check size={12} className="ml-2 shrink-0 text-primary" />
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

export const GenerationFilterCard: React.FC = () => {
  const locale = useLocale();
  const { isStack } = useResponsiveLayout();
  const optionLocale: SupportedLocale = locale;
  const anyLabel = resolveLocaleValue(anyOptionLabel, optionLocale);
  const noAbilitiesLabel = resolveLocaleValue(noAbilitySelectionLabel, optionLocale);
  const noGendersLabel = resolveLocaleValue(noGenderSelectionLabel, optionLocale);
  const cardTitle = resolveLocaleValue(generationResultsControlTitle, optionLocale);
  const resetFiltersLabel = resolveLocaleValue(generationResultsControlResetFiltersLabel, optionLocale);
  const fieldLabels = resolveLocaleValue(generationResultsControlFieldLabels, optionLocale);
  const statLabels = resolveLocaleValue(generationResultsControlStatLabels, optionLocale);
  const levelAriaLabel = resolveLocaleValue(generationResultsControlLevelAriaLabel, optionLocale);
  const abilityPreviewEllipsis = resolveLocaleValue(generationResultsControlAbilityPreviewEllipsis, optionLocale);
  const filters = useAppStore((state) => state.filters);
  const applyFilters = useAppStore((state) => state.applyFilters);
  const resetGenerationFilters = useAppStore((state) => state.resetGenerationFilters);
  const encounterTable = useAppStore((state) => state.encounterTable);
  const statsAvailable = useAppStore((state) => Boolean(state.params?.baseSeed));

  const resolvedResults = useAppStore((state: AppStoreState) => selectResolvedResults(state));

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

  // ステータス実数値フィールド（HABCDS縦配置用）
  const statNumericFields = STAT_KEYS.map((stat) => {
    const statLabel = statLabels[stat];
    const range = filters.statRanges[stat];
    const value = range?.min != null ? String(range.min) : '';
    return {
      id: `gen-filter-stat-${stat}`,
      label: statLabel,
      value,
      onChange: handleStatValueChange(stat),
      ariaLabel: formatGenerationResultsControlStatAria(statLabel, optionLocale),
    };
  });

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={<span id="gen-filter-title">{cardTitle}</span>}
      headerActions={
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
      }
      className={isStack ? 'min-h-[480px]' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="gen-filter-title"
      role="form"
    >
      {/* 2×n グリッド: ポケモン/特性/性別/性格/色違い/レベル/Timer0/VCount */}
      <div className="grid grid-cols-2 gap-2">
        {/* ポケモン */}
        <FilterPopoverField
          id="gen-filter-species"
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

        {/* 特性 */}
        <FilterPopoverField
          id="gen-filter-ability"
          label={fieldLabels.ability}
          placeholder={abilityPlaceholder}
          selectedValue={filters.abilityIndices.length ? String(filters.abilityIndices[0]) : 'any'}
          selectedLabel={abilitySelectedLabel}
          options={abilityFieldOptions}
          onSelect={handleAbilitySelect}
          disabled={abilityDisabled}
          emptyLabel={commandEmptyLabel}
        />

        {/* 性別 */}
        <FilterPopoverField
          id="gen-filter-gender"
          label={fieldLabels.gender}
          placeholder={genderPlaceholder}
          selectedValue={filters.genders.length ? filters.genders[0] : 'any'}
          selectedLabel={genderSelectedLabel}
          options={genderFieldOptions}
          onSelect={handleGenderSelect}
          disabled={genderDisabled}
          emptyLabel={commandEmptyLabel}
        />

        {/* 性格 */}
        <FilterPopoverField
          id="gen-filter-nature"
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

        {/* 色違い */}
        <FilterPopoverField
          id="gen-filter-shiny"
          label={fieldLabels.shiny}
          placeholder={anyLabel}
          selectedValue={filters.shinyMode}
          selectedLabel={shinySelectedLabel}
          options={shinyOptions}
          onSelect={handleShinyModeChange}
          emptyLabel={commandEmptyLabel}
        />

        {/* レベル */}
        <div className="flex flex-col gap-1">
          <Label htmlFor="gen-filter-level" className="text-xs">
            {fieldLabels.level}
          </Label>
          <Input
            id="gen-filter-level"
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            maxLength={3}
            value={levelValue}
            onChange={handleLevelValueChange}
            className="h-8 w-full px-2 text-right font-mono text-xs"
            placeholder={anyLabel}
            disabled={!statsAvailable}
            aria-label={levelAriaLabel}
          />
        </div>

        {/* Timer0 */}
        <div className="flex flex-col gap-1">
          <Label htmlFor="gen-filter-timer0" className="text-xs">
            {fieldLabels.timer0}
          </Label>
          <Input
            id="gen-filter-timer0"
            type="text"
            inputMode="text"
            value={filters.timer0Filter ?? ''}
            onChange={handleHexFilterChange('timer0Filter')}
            className="h-8 w-full px-2 text-right font-mono uppercase text-xs"
            placeholder="0000"
            maxLength={6}
            spellCheck={false}
            autoComplete="off"
          />
        </div>

        {/* VCount */}
        <div className="flex flex-col gap-1">
          <Label htmlFor="gen-filter-vcount" className="text-xs">
            {fieldLabels.vcount}
          </Label>
          <Input
            id="gen-filter-vcount"
            type="text"
            inputMode="text"
            value={filters.vcountFilter ?? ''}
            onChange={handleHexFilterChange('vcountFilter')}
            className="h-8 w-full px-2 text-right font-mono uppercase text-xs"
            placeholder="00"
            maxLength={4}
            spellCheck={false}
            autoComplete="off"
          />
        </div>
      </div>

      {/* ステータス実数値フィルター（HABCDS縦配置） */}
      <section className="space-y-2 mt-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">
          {optionLocale === 'ja' ? 'ステータス実数値' : 'Stats'}
        </h4>
        <div className="space-y-2">
          {statNumericFields.map((field) => (
            <div key={field.id} className="flex items-center gap-2">
              <span className="text-xs w-8">{field.label}</span>
              <Input
                id={field.id}
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={3}
                value={field.value}
                onChange={field.onChange}
                className="text-xs h-7 w-20 text-center font-mono"
                placeholder={anyLabel}
                disabled={!statsAvailable}
                aria-label={field.ariaLabel}
              />
            </div>
          ))}
        </div>
      </section>
    </PanelCard>
  );
};
