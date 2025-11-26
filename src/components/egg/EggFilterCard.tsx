import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Funnel } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName } from '@/lib/utils/format-display';
import { DOMAIN_NATURE_COUNT } from '@/types/domain';
import { createDefaultEggFilter, type StatRange, type EggIndividualFilter } from '@/types/egg';
import {
  eggFilterPanelTitle,
  eggFilterDisabledLabel,
  eggFilterIvRangeTitle,
  eggFilterNatureLabel,
  eggFilterGenderLabel,
  eggFilterAbilityLabel,
  eggFilterShinyLabel,
  eggFilterHpTypeLabel,
  eggFilterHpPowerLabel,
  eggFilterNoSelection,
  eggFilterGenderOptions,
  eggFilterAbilityOptions,
  eggFilterShinyOptions,
  eggFilterIvUnknownLabel,
  eggFilterTimer0Label,
  eggFilterVcountLabel,
  eggFilterTimer0Placeholder,
  eggFilterVcountPlaceholder,
  eggFilterBootTimingDisabledHint,
  eggFilterStatNames,
} from '@/lib/i18n/strings/egg-filter';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';

/**
 * EggFilterCard
 * タマゴ個体フィルター設定カード
 */
export const EggFilterCard: React.FC = () => {
  const { draftParams, updateDraftParams, bootTimingFilters, updateBootTimingFilters, status } = useEggStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
  const disabled = status === 'running' || status === 'starting';

  const filter = draftParams.filter || createDefaultEggFilter();
  const filterDisabled = draftParams.filterDisabled ?? false;
  const isBootTimingMode = draftParams.seedSourceMode === 'boot-timing';

  // Localized options
  const genderOptions = resolveLocaleValue(eggFilterGenderOptions, locale);
  const abilityOptions = resolveLocaleValue(eggFilterAbilityOptions, locale);
  const shinyOptions = resolveLocaleValue(eggFilterShinyOptions, locale);
  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;
  const statNames = resolveLocaleValue(eggFilterStatNames, locale);

  const updateFilter = (updates: Partial<EggIndividualFilter>) => {
    updateDraftParams({
      filter: { ...filter, ...updates },
    });
  };

  const handleIvRangeChange = (
    statIndex: number,
    minMax: 'min' | 'max',
    value: number
  ) => {
    const newRanges = [...filter.ivRanges] as [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
    newRanges[statIndex] = {
      ...newRanges[statIndex],
      [minMax]: Math.max(0, Math.min(31, value)),
    };
    updateFilter({ ivRanges: newRanges });
  };

  const handleIvRangeUnknownChange = (
    statIndex: number,
    isUnknown: boolean
  ) => {
    const newRanges = [...filter.ivRanges] as [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
    if (isUnknown) {
      // unknown時は min=0, max=32 を設定
      newRanges[statIndex] = { min: 0, max: 32 };
    } else {
      // チェック解除時は min=0, max=31 を設定
      newRanges[statIndex] = { min: 0, max: 31 };
    }
    updateFilter({ ivRanges: newRanges });
  };

  // IV範囲がunknown(0-32)かどうかを判定
  const isIvRangeUnknown = (statIndex: number): boolean => {
    const range = filter.ivRanges[statIndex];
    return range.min === 0 && range.max === 32;
  };

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={<span id="egg-filter-title">{eggFilterPanelTitle[locale]}</span>}
      className={isStack ? 'min-h-[480px]' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="egg-filter-title"
      role="form"
    >
      <>
        {/* 性格・性別・特性・色違い・めざパ・Timer0/VCount: 2列グリッド */}
          <div className="grid grid-cols-2 gap-2">
            {/* 性格フィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs">{eggFilterNatureLabel[locale]}</Label>
              <Select
                value={filter.nature !== undefined ? String(filter.nature) : 'none'}
                onValueChange={(v) => updateFilter({ nature: v !== 'none' ? Number(v) : undefined })}
                disabled={disabled || filterDisabled}
              >
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder={eggFilterNoSelection[locale]} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none" className="text-xs">{eggFilterNoSelection[locale]}</SelectItem>
                  {Array.from({ length: DOMAIN_NATURE_COUNT }, (_, i) => (
                    <SelectItem key={i} value={String(i)} className="text-xs">
                      {natureName(i, locale)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* 性別フィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs">{eggFilterGenderLabel[locale]}</Label>
              <Select
                value={filter.gender || 'none'}
                onValueChange={(v) => updateFilter({ gender: v !== 'none' ? v as 'male' | 'female' | 'genderless' : undefined })}
                disabled={disabled || filterDisabled}
              >
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder={eggFilterNoSelection[locale]} />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(genderOptions).map(([value, label]) => (
                    <SelectItem key={value} value={value} className="text-xs">
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* 特性フィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs">{eggFilterAbilityLabel[locale]}</Label>
              <Select
                value={filter.ability !== undefined ? String(filter.ability) : 'none'}
                onValueChange={(v) => updateFilter({ ability: v !== 'none' ? Number(v) as 0 | 1 | 2 : undefined })}
                disabled={disabled || filterDisabled}
              >
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder={eggFilterNoSelection[locale]} />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(abilityOptions).map(([value, label]) => (
                    <SelectItem key={value} value={value} className="text-xs">
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* 色違いフィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs">{eggFilterShinyLabel[locale]}</Label>
              <Select
                value={filter.shiny !== undefined ? String(filter.shiny) : 'none'}
                onValueChange={(v) => updateFilter({ shiny: v !== 'none' ? Number(v) as 0 | 1 | 2 : undefined })}
                disabled={disabled || filterDisabled}
              >
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder={eggFilterNoSelection[locale]} />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(shinyOptions).map(([value, label]) => (
                    <SelectItem key={value} value={value} className="text-xs">
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* めざパタイプフィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs">{eggFilterHpTypeLabel[locale]}</Label>
              <Select
                value={filter.hiddenPowerType !== undefined ? String(filter.hiddenPowerType) : 'none'}
                onValueChange={(v) => updateFilter({ hiddenPowerType: v !== 'none' ? Number(v) : undefined })}
                disabled={disabled || filterDisabled}
              >
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder={eggFilterNoSelection[locale]} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none" className="text-xs">{eggFilterNoSelection[locale]}</SelectItem>
                  {hpTypeNames.map((name, i) => (
                    <SelectItem key={i} value={String(i)} className="text-xs">
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* めざパ威力フィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs">{eggFilterHpPowerLabel[locale]}</Label>
              <Input
                type="number"
                min={30}
                max={70}
                value={filter.hiddenPowerPower ?? ''}
                onChange={(e) => {
                  const v = e.target.value;
                  updateFilter({ hiddenPowerPower: v ? Math.max(30, Math.min(70, parseInt(v))) : undefined });
                }}
                disabled={disabled || filterDisabled}
                placeholder={eggFilterNoSelection[locale]}
                className="text-xs"
              />
            </div>

            {/* Timer0 フィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="egg-filter-timer0">{eggFilterTimer0Label[locale]}</Label>
              <Input
                id="egg-filter-timer0"
                value={bootTimingFilters.timer0Filter ?? ''}
                onChange={(e) => updateBootTimingFilters({ timer0Filter: e.target.value.replace(/[^0-9a-fA-F]/g, '').toUpperCase() })}
                disabled={disabled || !isBootTimingMode}
                placeholder={isBootTimingMode ? eggFilterTimer0Placeholder[locale] : eggFilterBootTimingDisabledHint[locale]}
                className="font-mono text-xs"
              />
            </div>

            {/* VCount フィルター */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="egg-filter-vcount">{eggFilterVcountLabel[locale]}</Label>
              <Input
                id="egg-filter-vcount"
                value={bootTimingFilters.vcountFilter ?? ''}
                onChange={(e) => updateBootTimingFilters({ vcountFilter: e.target.value.replace(/[^0-9a-fA-F]/g, '').toUpperCase() })}
                disabled={disabled || !isBootTimingMode}
                placeholder={isBootTimingMode ? eggFilterVcountPlaceholder[locale] : eggFilterBootTimingDisabledHint[locale]}
                className="font-mono text-xs"
              />
            </div>
          </div>

          {/* IV範囲フィルター */}
          <section className="space-y-2 mt-3" role="group">
            <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{eggFilterIvRangeTitle[locale]}</h4>
            <div className="space-y-2">
              {statNames.map((stat, i) => {
                const isUnknown = isIvRangeUnknown(i);
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-xs w-8">{stat}</span>
                    <Input
                      type="number"
                      min={0}
                      max={31}
                      value={isUnknown ? '' : filter.ivRanges[i].min}
                      onChange={(e) => handleIvRangeChange(i, 'min', parseInt(e.target.value) || 0)}
                      disabled={disabled || filterDisabled || isUnknown}
                      className="text-xs h-7 w-14 text-center"
                      placeholder={isUnknown ? '?' : 'min'}
                    />
                    <span className="text-xs">~</span>
                    <Input
                      type="number"
                      min={0}
                      max={31}
                      value={isUnknown ? '' : filter.ivRanges[i].max}
                      onChange={(e) => handleIvRangeChange(i, 'max', parseInt(e.target.value) || 31)}
                      disabled={disabled || filterDisabled || isUnknown}
                      className="text-xs h-7 w-14 text-center"
                      placeholder={isUnknown ? '?' : 'max'}
                    />
                    <div className="flex items-center gap-1">
                      <Checkbox
                        id={`egg-filter-iv-unknown-${i}`}
                        checked={isUnknown}
                        onCheckedChange={(checked) => handleIvRangeUnknownChange(i, !!checked)}
                        disabled={disabled || filterDisabled}
                        className="h-3 w-3"
                      />
                      <Label htmlFor={`egg-filter-iv-unknown-${i}`} className="text-[10px] text-muted-foreground cursor-pointer">
                        {eggFilterIvUnknownLabel[locale]}
                      </Label>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
          {/* フィルター無効化チェック */}
          <div className="flex items-center gap-2 mb-3">
            <Checkbox
              id="egg-filter-disabled"
              checked={filterDisabled}
              onCheckedChange={(checked) => {
                updateDraftParams({ filterDisabled: !!checked });
              }}
              disabled={disabled}
            />
            <Label htmlFor="egg-filter-disabled" className="text-xs">{eggFilterDisabledLabel[locale]}</Label>
          </div>
        </>
    </PanelCard>
  );
};
