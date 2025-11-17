import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { resolveEncounterLocationName, resolveStaticEncounterName } from '@/data/encounters/i18n/display-name-resolver';
import { isLocationBasedEncounter, listEncounterLocations, listEncounterSpeciesOptions } from '@/data/encounters/helpers';
import { buildResolutionContext, enrichForSpecies } from '@/lib/initialization/build-resolution-context';
import { natureName } from '@/lib/utils/format-display';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  generationParamsAbilityLabel,
  generationParamsAbilityOptionLabels,
  generationParamsBaseSeedLabel,
  generationParamsBaseSeedPlaceholder,
  generationParamsDataUnavailablePlaceholder,
  generationParamsEncounterCategoryLabel,
  generationParamsEncounterFieldLabel,
  generationParamsEncounterSpeciesLabel,
  generationParamsEncounterTypeLabel,
  generationParamsMaxAdvancesLabel,
  generationParamsMinAdvanceLabel,
  generationParamsNoTypesAvailableLabel,
  generationParamsNotApplicablePlaceholder,
  generationParamsPanelTitle,
  generationParamsScreenReaderAnnouncement,
  generationParamsSectionTitles,
  generationParamsSelectOptionPlaceholder,
  generationParamsSelectSpeciesPlaceholder,
  generationParamsStaticDataPendingLabel,
  generationParamsStopFirstShinyLabel,
  generationParamsStopOnCapLabel,
  generationParamsSyncNatureLabel,
  generationParamsTypeUnavailablePlaceholder,
} from '@/lib/i18n/strings/generation-params';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { useAppStore } from '@/store/app-store';
import type { GenerationParamsHex } from '@/types/generation';
import {
  DomainEncounterType,
  DomainEncounterCategoryOptions,
  getDomainEncounterCategoryDisplayName,
  getDomainEncounterTypeCategory,
  getDomainEncounterTypeDisplayName,
  getDomainEncounterTypeName,
  listDomainEncounterTypeNamesByCategory,
  type DomainEncounterTypeCategoryKey,
} from '@/types/domain';
import { Gear } from '@phosphor-icons/react';

// Simple hex normalization guard
function isHexLike(v: string) {
  return /^(0x)?[0-9a-fA-F]*$/.test(v.trim());
}

function toDomainEncounterType(value: number): DomainEncounterType | null {
  return getDomainEncounterTypeName(value) ? (value as DomainEncounterType) : null;
}

// Ability モード選択肢 (Compound は WIP のため disabled)
const ABILITY_OPTION_DEFS: Array<{ value: NonNullable<GenerationParamsHex['abilityMode']>; disabled?: boolean }> = [
  { value: 'none' },
  { value: 'sync' },
  { value: 'compound', disabled: true },
];

const DEFAULT_ENCOUNTER_CATEGORY: DomainEncounterTypeCategoryKey = (
  DomainEncounterCategoryOptions.find(option => !option.disabled)?.key ?? 'wild'
) as DomainEncounterTypeCategoryKey;

const SYNC_NATURE_IDS = Array.from({ length: 25 }, (_, id) => id);

export const GenerationParamsCard: React.FC = () => {
  const locale = useLocale();
  // NOTE(perf): 必要項目のみ個別購読し encounterField 変更時の全体再レンダーを抑制
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

  const localized = React.useMemo(() => {
    const abilityLabels = resolveLocaleValue(generationParamsAbilityOptionLabels, locale);
    return {
      panelTitle: resolveLocaleValue(generationParamsPanelTitle, locale),
      sectionTitles: {
        target: resolveLocaleValue(generationParamsSectionTitles.target, locale),
        encounter: resolveLocaleValue(generationParamsSectionTitles.encounter, locale),
        stop: resolveLocaleValue(generationParamsSectionTitles.stopConditions, locale),
      },
      labels: {
        baseSeed: resolveLocaleValue(generationParamsBaseSeedLabel, locale),
        baseSeedPlaceholder: resolveLocaleValue(generationParamsBaseSeedPlaceholder, locale),
        minAdvance: resolveLocaleValue(generationParamsMinAdvanceLabel, locale),
        maxAdvances: resolveLocaleValue(generationParamsMaxAdvancesLabel, locale),
        encounterCategory: resolveLocaleValue(generationParamsEncounterCategoryLabel, locale),
        encounterType: resolveLocaleValue(generationParamsEncounterTypeLabel, locale),
        encounterField: resolveLocaleValue(generationParamsEncounterFieldLabel, locale),
        encounterSpecies: resolveLocaleValue(generationParamsEncounterSpeciesLabel, locale),
        ability: resolveLocaleValue(generationParamsAbilityLabel, locale),
        syncNature: resolveLocaleValue(generationParamsSyncNatureLabel, locale),
        stopFirstShiny: resolveLocaleValue(generationParamsStopFirstShinyLabel, locale),
        stopOnCap: resolveLocaleValue(generationParamsStopOnCapLabel, locale),
      },
      placeholders: {
        typeUnavailable: resolveLocaleValue(generationParamsTypeUnavailablePlaceholder, locale),
        selectOption: resolveLocaleValue(generationParamsSelectOptionPlaceholder, locale),
        notApplicable: resolveLocaleValue(generationParamsNotApplicablePlaceholder, locale),
        selectSpecies: resolveLocaleValue(generationParamsSelectSpeciesPlaceholder, locale),
        dataUnavailable: resolveLocaleValue(generationParamsDataUnavailablePlaceholder, locale),
      },
      messages: {
        noTypesAvailable: resolveLocaleValue(generationParamsNoTypesAvailableLabel, locale),
        staticDataPending: resolveLocaleValue(generationParamsStaticDataPendingLabel, locale),
        screenReader: resolveLocaleValue(generationParamsScreenReaderAnnouncement, locale),
      },
      abilityLabels,
    };
  }, [locale]);

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

  const update = (partial: Partial<GenerationParamsHex>) => {
    const next: Partial<GenerationParamsHex> = { ...partial };
    const targetVersion = partial.version ?? draftParams.version ?? 'B';
    if (partial.version !== undefined && (targetVersion === 'B' || targetVersion === 'W')) {
      next.memoryLink = false;
    }
    if (partial.newGame !== undefined) {
      if (!partial.newGame) {
        next.withSave = true;
      } else if (draftParams.withSave === undefined && partial.withSave === undefined) {
        next.withSave = true;
      }
    }
    if (partial.withSave !== undefined && !partial.withSave) {
      next.memoryLink = false;
    }
    setDraftParams(next);
  };

  const abilityMode = (hexDraft.abilityMode ?? 'none') as NonNullable<GenerationParamsHex['abilityMode']>;
  const onAbilityChange = (mode: NonNullable<GenerationParamsHex['abilityMode']>) => {
    // syncEnabled 連動: sync 選択時のみ true
    update({ abilityMode: mode, syncEnabled: mode === 'sync' });
  };
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
  const resolvedLocationOptions = React.useMemo(() => {
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
  const staticOptions = React.useMemo(() => speciesOptions.filter(opt => opt.kind === 'static'), [speciesOptions]);
  const staticOptionsWithLabels = React.useMemo(() => {
    if (!staticOptions.length) return [];
    return staticOptions.map(option => ({
      ...option,
      label: resolveStaticEncounterName(option.displayNameKey, locale, option.displayNameKey),
    }));
  }, [staticOptions, locale]);
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
  const onEncounterCategoryChange = (categoryKey: DomainEncounterTypeCategoryKey) => {
    const names = listDomainEncounterTypeNamesByCategory(categoryKey);
    if (!names.length) return;
    const nextValue = (DomainEncounterType as Record<string, number>)[names[0]];
    if (encounterType != null && nextValue === encounterType) return;
    update({ encounterType: nextValue });
  };
  const noTypeOptions = encounterTypeOptions.length === 0;
  const typeSelectDisabled = disabled || noTypeOptions;
  const typeSelectPlaceholder = noTypeOptions ? localized.placeholders.typeUnavailable : undefined;
  const locationSelectPlaceholder = locationOptions.length ? localized.placeholders.selectOption : localized.placeholders.notApplicable;
  const staticSelectPlaceholder = staticOptions.length ? localized.placeholders.selectSpecies : localized.placeholders.dataUnavailable;
  const { isStack } = useResponsiveLayout();
  const syncNatureOptions = React.useMemo(() => {
    return SYNC_NATURE_IDS.map(id => ({ id, label: natureName(id, locale) }));
  }, [locale]);
  const abilityOptions = React.useMemo(() => {
    return ABILITY_OPTION_DEFS.map(opt => ({
      ...opt,
      label: localized.abilityLabels[opt.value],
    }));
  }, [localized.abilityLabels]);

  // フィールド選択に応じてエンカウントテーブルと補助データをストアへ供給
  React.useEffect(() => {
    // Sync derived encounter context into the shared store for downstream selectors and workers.
    const state = useAppStore.getState();
    const resetContext = () => {
      if (state.encounterTable) setEncounterTable(undefined);
      if (state.genderRatios) setGenderRatios(undefined);
      if (state.abilityCatalog) setAbilityCatalog(undefined);
    };

    if (encounterType == null) {
      resetContext();
      return;
    }

    if (isLocationBased) {
      if (!encounterField) {
        resetContext();
        return;
      }

      const context = buildResolutionContext({
        version,
        location: encounterField,
        encounterType,
      });

      const table = context.encounterTable;
      if (!table) {
        resetContext();
        return;
      }

      for (const slot of table.slots) {
        enrichForSpecies(context, slot.speciesId);
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
      return;
    }

    // Static encounter branch
    const staticOption = staticOptions.find(opt => opt.id === staticEncounterId)
      ?? staticOptions.find(opt => opt.speciesId === encounterSpeciesId)
      ?? staticOptions[0];
    if (!staticOption) {
      resetContext();
      return;
    }

    const context = buildResolutionContext({
      version,
      encounterType,
      staticEncounter: { id: staticOption.id, speciesId: staticOption.speciesId, level: staticOption.level },
    });

    const table = context.encounterTable;
    if (!table) {
      resetContext();
      return;
    }

    enrichForSpecies(context, staticOption.speciesId);

    if (state.encounterTable !== table) {
      setEncounterTable(table);
    }

    if (state.genderRatios !== context.genderRatios) {
      setGenderRatios(context.genderRatios);
    }

    if (state.abilityCatalog !== context.abilityCatalog) {
      setAbilityCatalog(context.abilityCatalog);
    }
  }, [version, encounterType, encounterField, encounterSpeciesId, staticEncounterId, isLocationBased, staticOptions, setEncounterTable, setGenderRatios, setAbilityCatalog]);

  return (
    <PanelCard
      icon={<Gear size={20} className="opacity-80" />}
      title={<span id="gen-params-title">{localized.panelTitle}</span>}
      className={isStack ? 'max-h-200' : 'min-h-64'}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="gen-params-title"
      role="form"
    >
      {/* Profile-managed fields (Version, TID, SID, etc.) are configured via Device Profile panel. */}
      {/* Target (Range) */}
      <section aria-labelledby="gen-target" className="space-y-2" role="group">
        <h4 id="gen-target" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{localized.sectionTitles.target}</h4>
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          {/* Base Seed */}
          <div className="flex flex-col gap-1 min-w-0">
            <Label className="text-xs" htmlFor="base-seed">{localized.labels.baseSeed}</Label>
            {baseSeedTooltipEntries && baseSeedTooltipEntries.length > 0 ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Input
                    id="base-seed"
                    className="font-mono h-9"
                    disabled={disabled}
                    value={hexDraft.baseSeedHex ?? '0'}
                    onChange={e => {
                      const v = e.target.value;
                      if (isHexLike(v)) update({ baseSeedHex: v.replace(/^0x/i, '') });
                    }}
                    placeholder={localized.labels.baseSeedPlaceholder}
                  />
                </TooltipTrigger>
                <TooltipContent side="bottom" className="space-y-1 text-left">
                  {baseSeedTooltipEntries.map(entry => (
                    <div key={entry.label} className="space-y-0.5">
                      <div className="font-semibold leading-tight">{entry.label}</div>
                      <div className="font-mono leading-tight">{entry.spread}</div>
                      <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                    </div>
                  ))}
                </TooltipContent>
              </Tooltip>
            ) : (
              <Input
                id="base-seed"
                className="font-mono h-9"
                disabled={disabled}
                value={hexDraft.baseSeedHex ?? '0'}
                onChange={e => {
                  const v = e.target.value;
                  if (isHexLike(v)) update({ baseSeedHex: v.replace(/^0x/i, '') });
                }}
                placeholder={localized.labels.baseSeedPlaceholder}
              />
            )}
          </div>
          {/* Min Advance (offset) */}
          <div className="flex flex-col gap-1 min-w-0">
            <Label className="text-xs" htmlFor="min-advance">{localized.labels.minAdvance}</Label>
            <Input
              id="min-advance"
              type="number"
              inputMode="numeric"
              className="h-9"
              disabled={disabled}
              value={parseInt(hexDraft.offsetHex ?? '0', 16)}
              onChange={e => update({ offsetHex: Number(e.target.value).toString(16) })}
              placeholder="0"
            />
          </div>
          {/* Max Advances */}
          <div className="flex flex-col gap-1">
            <Label className="text-xs" htmlFor="max-adv">{localized.labels.maxAdvances}</Label>
            <Input
              id="max-adv"
              type="number"
              inputMode="numeric"
              className="h-9"
              disabled={disabled}
              value={draftParams.maxAdvances ?? 0}
              onChange={e => update({ maxAdvances: Number(e.target.value) })}
            />
          </div>
        </div>
      </section>
      <Separator />
      {/* Encounter & Ability */}
      <section aria-labelledby="gen-encounter" className="space-y-2" role="group">
        <h4 id="gen-encounter" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{localized.sectionTitles.encounter}</h4>
        <div className="grid gap-3 grid-cols-1 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,2fr)]">
          {/* Encounter Category */}
          <div className="flex flex-col gap-1 min-w-0 w-full">
            <Label className="text-xs" id="lbl-encounter-category" htmlFor="encounter-category">{localized.labels.encounterCategory}</Label>
            <Select value={encounterCategory} onValueChange={v => onEncounterCategoryChange(v as DomainEncounterTypeCategoryKey)} disabled={disabled}>
              <SelectTrigger id="encounter-category" aria-labelledby="lbl-encounter-category encounter-category" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-72">
                {encounterCategoryOptions.map(option => (
                  <SelectItem key={option.key} value={option.key} disabled={option.disabled}>{option.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {/* Encounter Type */}
          <div className="flex flex-col gap-1 min-w-0 w-full">
            <Label className="text-xs" id="lbl-encounter-type" htmlFor="encounter-type">{localized.labels.encounterType}</Label>
            <Select value={encounterValue.toString()} onValueChange={v => update({ encounterType: Number(v) })} disabled={typeSelectDisabled}>
              <SelectTrigger id="encounter-type" aria-labelledby="lbl-encounter-type encounter-type" className="w-full">
                <SelectValue placeholder={typeSelectPlaceholder} />
              </SelectTrigger>
              <SelectContent className="max-h-72">
                {noTypeOptions ? (
                  <SelectItem value="__no-type" disabled>{localized.messages.noTypesAvailable}</SelectItem>
                ) : encounterTypeOptions.map(opt => (
                  <SelectItem key={opt.name} value={opt.value.toString()}>{opt.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {/* Encounter Field (location) */}
          {isLocationBased && (
            <div className="flex flex-col gap-1 min-w-0 w-full">
              <Label className="text-xs" id="lbl-encounter-field" htmlFor="encounter-field">{localized.labels.encounterField}</Label>
              <Select value={encounterField ?? ''} onValueChange={v => setEncounterField(v)} disabled={disabled || locationOptions.length === 0}>
                <SelectTrigger id="encounter-field" aria-labelledby="lbl-encounter-field encounter-field" className="w-full whitespace-normal text-left">
                  <SelectValue placeholder={locationSelectPlaceholder} className="!line-clamp-2" />
                </SelectTrigger>
                <SelectContent className="max-h-72">
                  {resolvedLocationOptions.map(loc => (
                    <SelectItem key={loc.key} value={loc.key} className="whitespace-normal break-words text-left">{loc.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {/* Encounter Species (static encounters only) */}
          {!isLocationBased && (
            <div className="flex flex-col gap-1 min-w-0 w-full">
              <Label className="text-xs" id="lbl-encounter-species" htmlFor="encounter-species">{localized.labels.encounterSpecies}</Label>
              <Select
                value={staticEncounterId ?? ''}
                onValueChange={id => {
                  setStaticEncounterId(id);
                  const selected = staticOptions.find(opt => opt.id === id);
                  if (selected) {
                    setEncounterSpeciesId(selected.speciesId);
                  } else {
                    setEncounterSpeciesId(undefined);
                  }
                }}
                disabled={disabled}
              >
                <SelectTrigger id="encounter-species" aria-labelledby="lbl-encounter-species encounter-species" className="w-full whitespace-normal text-left">
                  <SelectValue placeholder={staticSelectPlaceholder} className="!line-clamp-2" />
                </SelectTrigger>
                <SelectContent className="max-h-72">
                  {staticOptionsWithLabels.length === 0 ? (
                    <SelectItem value="__coming-soon" disabled>{localized.messages.staticDataPending}</SelectItem>
                  ) : staticOptionsWithLabels.map(sp => (
                    <SelectItem key={sp.id} value={sp.id} className="text-left">
                      {`${sp.label} (Lv.${sp.level})`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {/* Ability Mode */}
          <div className="flex flex-col gap-1 min-w-0 w-full">
            <Label className="text-xs" id="lbl-ability-mode" htmlFor="ability-mode">{localized.labels.ability}</Label>
            <Select value={abilityMode} onValueChange={v => onAbilityChange(v as NonNullable<GenerationParamsHex['abilityMode']>)} disabled={disabled}>
              <SelectTrigger id="ability-mode" aria-labelledby="lbl-ability-mode ability-mode" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-64">
                {abilityOptions.map(opt => <SelectItem key={opt.value} value={opt.value} disabled={opt.disabled}>{opt.label}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          {/* Sync Nature */}
          <div className="flex flex-col gap-1 min-w-0 w-full">
            <Label className="text-xs" id="lbl-sync-nature" htmlFor="sync-nature">{localized.labels.syncNature}</Label>
            <Select value={(draftParams.syncNatureId ?? 0).toString()} onValueChange={v => update({ syncNatureId: Number(v) })} disabled={disabled || !syncActive}>
              <SelectTrigger id="sync-nature" aria-labelledby="lbl-sync-nature sync-nature" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-64">
                {syncNatureOptions.map(option => (
                  <SelectItem key={option.id} value={option.id.toString()}>{option.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </section>
      <Separator />
      {/* Stop Conditions */}
      <section aria-labelledby="gen-stop" className="space-y-2" role="group">
        <h4 id="gen-stop" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{localized.sectionTitles.stop}</h4>
        <div className="flex flex-wrap gap-6">
          <div className="flex items-center gap-2">
            <Checkbox id="stop-first-shiny" aria-labelledby="lbl-stop-first-shiny" checked={draftParams.stopAtFirstShiny ?? false} disabled={disabled} onCheckedChange={v => update({ stopAtFirstShiny: Boolean(v) })} />
            <Label id="lbl-stop-first-shiny" htmlFor="stop-first-shiny" className="text-xs">{localized.labels.stopFirstShiny}</Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="stop-on-cap" aria-labelledby="lbl-stop-on-cap" checked={draftParams.stopOnCap ?? true} disabled={disabled} onCheckedChange={v => update({ stopOnCap: Boolean(v) })} />
            <Label id="lbl-stop-on-cap" htmlFor="stop-on-cap" className="text-xs">{localized.labels.stopOnCap}</Label>
          </div>
        </div>
      </section>
      <div className="sr-only" aria-live="polite">
        {localized.messages.screenReader}
      </div>
    </PanelCard>
  );
};
