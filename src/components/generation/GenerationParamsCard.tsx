import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { BootTimingControls, type BootTimingLabels } from '@/components/generation/boot-timing/BootTimingControls';
import { useGenerationParamsForm } from '@/hooks/generation/useGenerationParamsForm';
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
  generationParamsMaxAdvanceLabel,
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
  generationParamsSeedSourceLabel,
  generationParamsSeedSourceOptionLabels,
  generationParamsBootTimingTimestampLabel,
  generationParamsBootTimingTimestampPlaceholder,
  generationParamsBootTimingKeyInputLabel,
  generationParamsBootTimingConfigureLabel,
  generationParamsBootTimingKeyDialogTitle,
  generationParamsBootTimingKeyResetLabel,
  generationParamsBootTimingKeyApplyLabel,
  generationParamsBootTimingProfileLabel,
} from '@/lib/i18n/strings/generation-params';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import type { GenerationParamsHex, SeedSourceMode } from '@/types/generation';
import type { DomainEncounterTypeCategoryKey } from '@/types/domain';
import { Sliders } from '@phosphor-icons/react';

// Simple hex normalization guard
function isHexLike(v: string) {
  return /^(0x)?[0-9a-fA-F]*$/.test(v.trim());
}

// Ability モード選択肢 (Compound は WIP のため disabled)
const ABILITY_OPTION_DEFS: Array<{ value: NonNullable<GenerationParamsHex['abilityMode']>; disabled?: boolean }> = [
  { value: 'none' },
  { value: 'sync' },
  { value: 'compound', disabled: true },
];
const SEED_SOURCE_OPTIONS: ReadonlyArray<SeedSourceMode> = ['lcg', 'boot-timing'];

export const GenerationParamsCard: React.FC = () => {
  const locale = useLocale();
  const form = useGenerationParamsForm(locale);
  const {
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
    encounterTypeValue,
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
  } = form;
  const {
    updateDraft,
    handleSeedSourceModeChange,
    handleAbilityModeChange,
    handleEncounterCategoryChange,
    handleEncounterTypeChange,
    handleLocationChange,
    handleStaticEncounterChange,
  } = form.actions;

  const localized = React.useMemo(() => {
    const abilityLabels = resolveLocaleValue(generationParamsAbilityOptionLabels, locale);
    const seedSourceOptions = resolveLocaleValue(generationParamsSeedSourceOptionLabels, locale);
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
        seedSource: resolveLocaleValue(generationParamsSeedSourceLabel, locale),
        bootTimestamp: resolveLocaleValue(generationParamsBootTimingTimestampLabel, locale),
        bootKeyInput: resolveLocaleValue(generationParamsBootTimingKeyInputLabel, locale),
        bootProfile: resolveLocaleValue(generationParamsBootTimingProfileLabel, locale),
        minAdvance: resolveLocaleValue(generationParamsMinAdvanceLabel, locale),
        maxAdvance: resolveLocaleValue(generationParamsMaxAdvanceLabel, locale),
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
        bootTimestamp: resolveLocaleValue(generationParamsBootTimingTimestampPlaceholder, locale),
      },
      messages: {
        noTypesAvailable: resolveLocaleValue(generationParamsNoTypesAvailableLabel, locale),
        staticDataPending: resolveLocaleValue(generationParamsStaticDataPendingLabel, locale),
        screenReader: resolveLocaleValue(generationParamsScreenReaderAnnouncement, locale),
      },
      abilityLabels,
      seedSourceOptions,
      bootTiming: {
        configure: resolveLocaleValue(generationParamsBootTimingConfigureLabel, locale),
        dialogTitle: resolveLocaleValue(generationParamsBootTimingKeyDialogTitle, locale),
        reset: resolveLocaleValue(generationParamsBootTimingKeyResetLabel, locale),
        apply: resolveLocaleValue(generationParamsBootTimingKeyApplyLabel, locale),
      },
    };
  }, [locale]);

  const bootTimingLabels = React.useMemo<BootTimingLabels>(() => ({
    timestamp: localized.labels.bootTimestamp,
    timestampPlaceholder: localized.placeholders.bootTimestamp,
    keyInput: localized.labels.bootKeyInput,
    profile: localized.labels.bootProfile,
    configure: localized.bootTiming.configure,
    dialogTitle: localized.bootTiming.dialogTitle,
    reset: localized.bootTiming.reset,
    apply: localized.bootTiming.apply,
  }), [localized]);
  const typeSelectPlaceholder = noTypeOptions ? localized.placeholders.typeUnavailable : undefined;
  const locationSelectPlaceholder = locationOptionsLength ? localized.placeholders.selectOption : localized.placeholders.notApplicable;
  const staticSelectPlaceholder = hasStaticOptions ? localized.placeholders.selectSpecies : localized.placeholders.dataUnavailable;
  const { isStack } = useResponsiveLayout();
  const abilityOptions = React.useMemo(() => {
    return ABILITY_OPTION_DEFS.map(opt => ({
      ...opt,
      label: localized.abilityLabels[opt.value],
    }));
  }, [localized.abilityLabels]);

  return (
    <>
      <PanelCard
        icon={<Sliders size={20} className="opacity-80" />}
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
        <div className="space-y-3">
          <div className="flex flex-col gap-1">
            <Label className="text-xs" id="lbl-seed-source" htmlFor="seed-source">{localized.labels.seedSource}</Label>
            <ToggleGroup
              id="seed-source"
              type="single"
              value={seedSourceMode}
              onValueChange={mode => {
                if (mode === 'lcg' || mode === 'boot-timing') {
                  handleSeedSourceModeChange(mode);
                }
              }}
              className="flex flex-wrap gap-2"
              aria-labelledby="lbl-seed-source"
            >
              {SEED_SOURCE_OPTIONS.map(mode => (
                <ToggleGroupItem
                  key={mode}
                  value={mode}
                  className="px-4 py-2 text-xs"
                  disabled={disabled}
                >
                  {localized.seedSourceOptions[mode]}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {isBootTimingMode ? (
              <BootTimingControls
                disabled={disabled}
                isActive={isBootTimingMode}
                labels={bootTimingLabels}
              />
            ) : (
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
                          if (isHexLike(v)) updateDraft({ baseSeedHex: v.replace(/^0x/i, '') });
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
                      if (isHexLike(v)) updateDraft({ baseSeedHex: v.replace(/^0x/i, '') });
                    }}
                    placeholder={localized.labels.baseSeedPlaceholder}
                  />
                )}
              </div>
            )}
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
                onChange={e => {
                  const v = Number(e.target.value);
                  updateDraft({ offsetHex: (Number.isNaN(v) ? 0 : v).toString(16) });
                }}
                onBlur={() => {
                  const current = parseInt(hexDraft.offsetHex ?? '0', 16);
                  const clamped = Math.max(0, current);
                  updateDraft({ offsetHex: clamped.toString(16) });
                }}
                placeholder="0"
              />
            </div>
            {/* Max Advances */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="max-adv">{localized.labels.maxAdvance}</Label>
              <Input
                id="max-adv"
                type="number"
                inputMode="numeric"
                className="h-9"
                disabled={disabled}
                value={draftParams.maxAdvances ?? 0}
                onChange={e => {
                  const v = Number(e.target.value);
                  updateDraft({ maxAdvances: Number.isNaN(v) ? 0 : v });
                }}
                onBlur={() => {
                  const current = draftParams.maxAdvances ?? 0;
                  const clamped = Math.max(0, current);
                  updateDraft({ maxAdvances: clamped });
                }}
              />
            </div>
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
            <Select value={encounterCategory} onValueChange={v => handleEncounterCategoryChange(v as DomainEncounterTypeCategoryKey)} disabled={disabled}>
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
            <Select value={encounterTypeValue.toString()} onValueChange={v => handleEncounterTypeChange(Number(v))} disabled={typeSelectDisabled}>
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
              <Select value={encounterField ?? ''} onValueChange={handleLocationChange} disabled={locationSelectDisabled}>
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
                onValueChange={handleStaticEncounterChange}
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
            <Select value={abilityMode} onValueChange={v => handleAbilityModeChange(v as NonNullable<GenerationParamsHex['abilityMode']>)} disabled={disabled}>
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
            <Select value={(draftParams.syncNatureId ?? 0).toString()} onValueChange={v => updateDraft({ syncNatureId: Number(v) })} disabled={disabled || !syncActive}>
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
            <Checkbox id="stop-first-shiny" aria-labelledby="lbl-stop-first-shiny" checked={draftParams.stopAtFirstShiny ?? false} disabled={disabled} onCheckedChange={v => updateDraft({ stopAtFirstShiny: Boolean(v) })} />
            <Label id="lbl-stop-first-shiny" htmlFor="stop-first-shiny" className="text-xs">{localized.labels.stopFirstShiny}</Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="stop-on-cap" aria-labelledby="lbl-stop-on-cap" checked={draftParams.stopOnCap ?? true} disabled={disabled} onCheckedChange={v => updateDraft({ stopOnCap: Boolean(v) })} />
            <Label id="lbl-stop-on-cap" htmlFor="stop-on-cap" className="text-xs">{localized.labels.stopOnCap}</Label>
          </div>
        </div>
      </section>
      </PanelCard>
    </>
  );
};
