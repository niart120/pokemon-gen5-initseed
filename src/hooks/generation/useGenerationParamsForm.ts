import React from 'react';
import { useAppStore } from '@/store/app-store';
import { buildResolutionContext, enrichForSpecies } from '@/lib/initialization/build-resolution-context';
import { resolveEncounterLocationName, resolveStaticEncounterName } from '@/data/encounters/i18n/display-name-resolver';
import { isLocationBasedEncounter, listEncounterLocations, listEncounterSpeciesOptions, type EncounterSpeciesOption } from '@/data/encounters/helpers';
import { lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { natureName } from '@/lib/utils/format-display';
import type { GenerationParamsHex, SeedSourceMode } from '@/types/generation';
import {
  DomainEncounterCategoryOptions,
  DomainEncounterType,
  getDomainEncounterCategoryDisplayName,
  getDomainEncounterTypeCategory,
  getDomainEncounterTypeDisplayName,
  getDomainEncounterTypeName,
  listDomainEncounterTypeNamesByCategory,
  type DomainEncounterTypeCategoryKey,
} from '@/types/domain';

interface LocationOptionWithLabel {
  key: string;
  label: string;
}

interface StaticOptionWithLabel {
  id: string;
  level: number;
  speciesId: number;
  label: string;
}

export interface GenerationParamsFormActions {
  updateDraft: (partial: Partial<GenerationParamsHex>) => void;
  handleSeedSourceModeChange: (mode: SeedSourceMode) => void;
  handleAbilityModeChange: (mode: NonNullable<GenerationParamsHex['abilityMode']>) => void;
  handleEncounterCategoryChange: (key: DomainEncounterTypeCategoryKey) => void;
  handleEncounterTypeChange: (value: number) => void;
  handleLocationChange: (locationKey: string) => void;
  handleStaticEncounterChange: (id: string) => void;
}

export interface GenerationParamsFormController {
  draftParams: Partial<GenerationParamsHex>;
  hexDraft: Partial<GenerationParamsHex>;
  disabled: boolean;
  seedSourceMode: SeedSourceMode;
  isBootTimingMode: boolean;
  abilityMode: NonNullable<GenerationParamsHex['abilityMode']>;
  syncActive: boolean;
  baseSeedTooltipEntries: ReturnType<typeof getIvTooltipEntries> | null;
  encounterCategory: DomainEncounterTypeCategoryKey;
  encounterCategoryOptions: Array<{ key: DomainEncounterTypeCategoryKey; label: string; disabled?: boolean }>;
  encounterTypeValue: number;
  encounterTypeOptions: Array<{ name: string; value: number; label: string }>;
  isLocationBased: boolean;
  encounterField: string | undefined;
  resolvedLocationOptions: LocationOptionWithLabel[];
  locationSelectDisabled: boolean;
  staticEncounterId: string | null | undefined;
  staticOptionsWithLabels: StaticOptionWithLabel[];
  hasStaticOptions: boolean;
  syncNatureOptions: Array<{ id: number; label: string }>;
  noTypeOptions: boolean;
  typeSelectDisabled: boolean;
  locationOptionsLength: number;
  actions: GenerationParamsFormActions;
}

const SYNC_NATURE_IDS = Array.from({ length: 25 }, (_, id) => id);

const DEFAULT_ENCOUNTER_CATEGORY: DomainEncounterTypeCategoryKey = (
  DomainEncounterCategoryOptions.find(option => !option.disabled)?.key ?? 'wild'
) as DomainEncounterTypeCategoryKey;

function toDomainEncounterType(value: number): DomainEncounterType | null {
  return getDomainEncounterTypeName(value) ? (value as DomainEncounterType) : null;
}

export function useGenerationParamsForm(locale: 'ja' | 'en'): GenerationParamsFormController {
  const draftParams = useAppStore(s => s.draftParams);
  const status = useAppStore(s => s.status);
  const encounterField = useAppStore(s => s.encounterField);
  const encounterSpeciesId = useAppStore(s => s.encounterSpeciesId);
  const staticEncounterId = useAppStore(s => s.staticEncounterId);
  const setDraftParams = useAppStore(s => s.setDraftParams);
  const setEncounterField = useAppStore(s => s.setEncounterField);
  const setEncounterSpeciesId = useAppStore(s => s.setEncounterSpeciesId);
  const setStaticEncounterId = useAppStore(s => s.setStaticEncounterId);
  const setEncounterTable = useAppStore(s => s.setEncounterTable);
  const setGenderRatios = useAppStore(s => s.setGenderRatios);
  const setAbilityCatalog = useAppStore(s => s.setAbilityCatalog);

  const disabled = status === 'running' || status === 'starting' || status === 'stopping';
  const hexDraft: Partial<GenerationParamsHex> = draftParams;

  const baseSeedTooltipEntries = React.useMemo(() => {
    const raw = (hexDraft.baseSeedHex ?? '').trim();
    if (!raw) return null;
    const hexDigits = raw.startsWith('0x') || raw.startsWith('0X') ? raw.slice(2) : raw;
    if (!hexDigits) return null;
    if (!/^[0-9a-fA-F]+$/.test(hexDigits)) return null;
    const lcgSeed = BigInt(`0x${hexDigits}`);
    if (lcgSeed < 0n) return null;
    const mtSeed = lcgSeedToMtSeed(lcgSeed);
    return getIvTooltipEntries(mtSeed, locale);
  }, [hexDraft.baseSeedHex, locale]);

  const updateDraft = React.useCallback((partial: Partial<GenerationParamsHex>) => {
    const currentDraft = useAppStore.getState().draftParams;
    const next: Partial<GenerationParamsHex> = { ...partial };
    const targetVersion = partial.version ?? currentDraft.version ?? 'B';
    if (partial.version !== undefined && (targetVersion === 'B' || targetVersion === 'W')) {
      next.memoryLink = false;
    }
    if (partial.newGame !== undefined) {
      if (!partial.newGame) {
        next.withSave = true;
      } else if (currentDraft.withSave === undefined && partial.withSave === undefined) {
        next.withSave = true;
      }
    }
    if (partial.withSave !== undefined && !partial.withSave) {
      next.memoryLink = false;
    }
    setDraftParams(next);
  }, [setDraftParams]);

  const seedSourceMode = (hexDraft.seedSourceMode ?? 'lcg') as SeedSourceMode;
  const isBootTimingMode = seedSourceMode === 'boot-timing';

  const abilityMode = (hexDraft.abilityMode ?? 'none') as NonNullable<GenerationParamsHex['abilityMode']>;
  const syncActive = abilityMode === 'sync' && (hexDraft.syncEnabled ?? false);

  const encounterValue = hexDraft.encounterType ?? 0;
  const encounterType = React.useMemo(() => toDomainEncounterType(encounterValue), [encounterValue]);
  const encounterCategory = React.useMemo<DomainEncounterTypeCategoryKey>(() => {
    if (encounterType == null) return DEFAULT_ENCOUNTER_CATEGORY;
    return getDomainEncounterTypeCategory(encounterType);
  }, [encounterType]);

  const version = draftParams.version ?? 'B';
  const isLocationBased = encounterType != null && isLocationBasedEncounter(encounterType);

  const locationOptions = React.useMemo(() => {
    if (encounterType == null || !isLocationBased) return [];
    return listEncounterLocations(version, encounterType);
  }, [version, encounterType, isLocationBased]);

  const resolvedLocationOptions = React.useMemo<LocationOptionWithLabel[]>(() => {
    if (!locationOptions.length) return [];
    return locationOptions.map(option => ({
      ...option,
      label: resolveEncounterLocationName(option.displayNameKey, locale, option.displayNameKey),
    }));
  }, [locationOptions, locale]);

  const speciesOptions = React.useMemo(() => {
    if (encounterType == null) return [];
    if (isLocationBased) {
      if (!encounterField) return [];
      return listEncounterSpeciesOptions(version, encounterType, encounterField);
    }
    return listEncounterSpeciesOptions(version, encounterType);
  }, [version, encounterType, isLocationBased, encounterField]);

  const staticOptions = React.useMemo(() => speciesOptions.filter((opt): opt is Extract<EncounterSpeciesOption, { kind: 'static' }> => opt.kind === 'static'), [speciesOptions]);

  const staticOptionsWithLabels = React.useMemo<StaticOptionWithLabel[]>(() => {
    if (!staticOptions.length) return [];
    return staticOptions.map(option => ({
      ...option,
      label: resolveStaticEncounterName(option.displayNameKey, locale, option.displayNameKey),
    }));
  }, [staticOptions, locale]);

  const selectedStaticEncounter = React.useMemo(() => {
    if (isLocationBased) return null;
    if (!staticOptions.length) return null;
    return staticOptions.find(opt => opt.id === staticEncounterId)
      ?? staticOptions.find(opt => opt.speciesId === encounterSpeciesId)
      ?? staticOptions[0]
      ?? null;
  }, [encounterSpeciesId, isLocationBased, staticOptions, staticEncounterId]);

  const encounterContextInput = React.useMemo(() => {
    if (encounterType == null) {
      return { kind: 'none' } as const;
    }
    if (isLocationBased) {
      if (!encounterField) {
        return { kind: 'none' } as const;
      }
      return {
        kind: 'location' as const,
        version,
        encounterType,
        location: encounterField,
      };
    }
    if (!selectedStaticEncounter) {
      return { kind: 'none' } as const;
    }
    return {
      kind: 'static' as const,
      version,
      encounterType,
      staticEncounter: {
        id: selectedStaticEncounter.id,
        speciesId: selectedStaticEncounter.speciesId,
        level: selectedStaticEncounter.level,
      },
    };
  }, [encounterType, encounterField, isLocationBased, selectedStaticEncounter, version]);

  const encounterTypeOptions = React.useMemo(() => {
    const names = listDomainEncounterTypeNamesByCategory(encounterCategory);
    return names.map(name => ({
      name,
      value: (DomainEncounterType as Record<string, number>)[name],
      label: getDomainEncounterTypeDisplayName(name, locale),
    }));
  }, [encounterCategory, locale]);

  const encounterCategoryOptions = React.useMemo(() => {
    return DomainEncounterCategoryOptions.map(option => ({
      ...option,
      label: getDomainEncounterCategoryDisplayName(option.key, locale),
    }));
  }, [locale]);

  const noTypeOptions = encounterTypeOptions.length === 0;
  const typeSelectDisabled = disabled || noTypeOptions;

  const locationOptionsLength = resolvedLocationOptions.length;
  const locationSelectDisabled = disabled || locationOptionsLength === 0;
  const hasStaticOptions = staticOptionsWithLabels.length > 0;

  const syncNatureOptions = React.useMemo(() => {
    return SYNC_NATURE_IDS.map(id => ({ id, label: natureName(id, locale) }));
  }, [locale]);

  React.useEffect(() => {
    const state = useAppStore.getState();
    const resetContext = () => {
      if (state.encounterTable) setEncounterTable(undefined);
      if (state.genderRatios) setGenderRatios(undefined);
      if (state.abilityCatalog) setAbilityCatalog(undefined);
    };

    if (encounterContextInput.kind === 'none') {
      resetContext();
      return;
    }

    const context = encounterContextInput.kind === 'location'
      ? buildResolutionContext({
        version: encounterContextInput.version,
        encounterType: encounterContextInput.encounterType,
        location: encounterContextInput.location,
      })
      : buildResolutionContext({
        version: encounterContextInput.version,
        encounterType: encounterContextInput.encounterType,
        staticEncounter: encounterContextInput.staticEncounter,
      });

    const table = context.encounterTable;
    if (!table) {
      resetContext();
      return;
    }

    if (encounterContextInput.kind === 'location') {
      for (const slot of table.slots) {
        enrichForSpecies(context, slot.speciesId);
      }
    } else {
      enrichForSpecies(context, encounterContextInput.staticEncounter.speciesId);
    }

    if (state.encounterTable !== table) {
      setEncounterTable(table);
    }

    if (state.genderRatios !== context.genderRatios) {
      setGenderRatios(context.genderRatios);
    }

    if (state.abilityCatalog !== context.abilityCatalog) {
      setAbilityCatalog(context.abilityCatalog);
    }
  }, [encounterContextInput, setAbilityCatalog, setEncounterTable, setGenderRatios]);

  const handleSeedSourceModeChange = React.useCallback((mode: SeedSourceMode) => {
    setDraftParams({ seedSourceMode: mode });
  }, [setDraftParams]);

  const handleAbilityModeChange = React.useCallback((mode: NonNullable<GenerationParamsHex['abilityMode']>) => {
    updateDraft({ abilityMode: mode, syncEnabled: mode === 'sync' });
  }, [updateDraft]);

  const handleEncounterCategoryChange = React.useCallback((categoryKey: DomainEncounterTypeCategoryKey) => {
    const names = listDomainEncounterTypeNamesByCategory(categoryKey);
    if (!names.length) return;
    const nextValue = (DomainEncounterType as Record<string, number>)[names[0]];
    if (encounterType != null && nextValue === encounterType) return;
    updateDraft({ encounterType: nextValue });
  }, [encounterType, updateDraft]);

  const handleEncounterTypeChange = React.useCallback((value: number) => {
    if (Number.isNaN(value)) return;
    updateDraft({ encounterType: value });
  }, [updateDraft]);

  const handleLocationChange = React.useCallback((locationKey: string) => {
    setEncounterField(locationKey || undefined);
  }, [setEncounterField]);

  const handleStaticEncounterChange = React.useCallback((id: string) => {
    if (!id) {
      setStaticEncounterId(null);
      setEncounterSpeciesId(undefined);
      return;
    }
    setStaticEncounterId(id);
    const selected = staticOptions.find(opt => opt.id === id);
    if (selected) {
      setEncounterSpeciesId(selected.speciesId);
    } else {
      setEncounterSpeciesId(undefined);
    }
  }, [setEncounterSpeciesId, setStaticEncounterId, staticOptions]);

  return {
    draftParams,
    hexDraft,
    disabled,
    seedSourceMode,
    isBootTimingMode,
    abilityMode,
    syncActive,
    baseSeedTooltipEntries,
    encounterCategory,
    encounterCategoryOptions,
    encounterTypeValue: encounterValue,
    encounterTypeOptions,
    isLocationBased,
    encounterField,
    resolvedLocationOptions,
    locationSelectDisabled,
    staticEncounterId,
    staticOptionsWithLabels,
    hasStaticOptions,
    syncNatureOptions,
    noTypeOptions,
    typeSelectDisabled,
    locationOptionsLength,
    actions: {
      updateDraft,
      handleSeedSourceModeChange,
      handleAbilityModeChange,
      handleEncounterCategoryChange,
      handleEncounterTypeChange,
      handleLocationChange,
      handleStaticEncounterChange,
    },
  } satisfies GenerationParamsFormController;
}
