/**
 * EggSearchFilterCard
 * 結果フィルターカード
 * 
 * 仕様: spec/agent/pr_egg_boot_timing_search/UI_DESIGN.md
 * EggFilterCardと同等の実装（個体値、性格、性別、特性、色違い、めざパ等）
 */

import React from 'react';
import { Funnel } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName } from '@/lib/utils/format-display';
import { DOMAIN_NATURE_COUNT } from '@/types/domain';
import { createDefaultEggFilter, type StatRange, type EggIndividualFilter } from '@/types/egg';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';
import {
  eggSearchFilterCardTitle,
  eggSearchFilterLabels,
  eggSearchStatNames,
  eggSearchGenderOptions,
  eggSearchAbilityOptions,
  eggSearchShinyOptions,
} from '@/lib/i18n/strings/egg-search';

export function EggSearchFilterCard() {
  const locale = useLocale();
  const { isStack } = useResponsiveLayout();
  const { draftParams, updateFilter, status } = useEggBootTimingSearchStore();
  
  const isRunning = status === 'running' || status === 'starting' || status === 'stopping';
  
  const filter = draftParams.filter || createDefaultEggFilter();

  const genderOptions = resolveLocaleValue(eggSearchGenderOptions, locale);
  const abilityOptions = resolveLocaleValue(eggSearchAbilityOptions, locale);
  const shinyOptions = resolveLocaleValue(eggSearchShinyOptions, locale);
  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;
  const statNames = resolveLocaleValue(eggSearchStatNames, locale);

  const handleFilterUpdate = (updates: Partial<EggIndividualFilter>) => {
    updateFilter(updates);
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
    handleFilterUpdate({ ivRanges: newRanges });
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
    handleFilterUpdate({ ivRanges: newRanges });
  };

  // IV範囲がunknown(0-32)かどうかを判定
  const isIvRangeUnknown = (statIndex: number): boolean => {
    const range = filter.ivRanges[statIndex];
    return range.min === 0 && range.max === 32;
  };

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={eggSearchFilterCardTitle[locale]}
      className={isStack ? 'min-h-[480px]' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
    >
      {/* 性格・性別・特性・色違い: 4列グリッド */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {/* 性格フィルター */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">{eggSearchFilterLabels.nature[locale]}</Label>
          <Select
            value={filter.nature !== undefined ? String(filter.nature) : 'none'}
            onValueChange={(v) => handleFilterUpdate({ nature: v !== 'none' ? Number(v) : undefined })}
            disabled={isRunning }
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={eggSearchFilterLabels.noSelection[locale]} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none" className="text-xs">{eggSearchFilterLabels.noSelection[locale]}</SelectItem>
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
          <Label className="text-xs">{eggSearchFilterLabels.gender[locale]}</Label>
          <Select
            value={filter.gender || 'none'}
            onValueChange={(v) => handleFilterUpdate({ gender: v !== 'none' ? v as 'male' | 'female' | 'genderless' : undefined })}
            disabled={isRunning }
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={eggSearchFilterLabels.noSelection[locale]} />
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
          <Label className="text-xs">{eggSearchFilterLabels.ability[locale]}</Label>
          <Select
            value={filter.ability !== undefined ? String(filter.ability) : 'none'}
            onValueChange={(v) => handleFilterUpdate({ ability: v !== 'none' ? Number(v) as 0 | 1 | 2 : undefined })}
            disabled={isRunning }
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={eggSearchFilterLabels.noSelection[locale]} />
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
          <Label className="text-xs">{eggSearchFilterLabels.shiny[locale]}</Label>
          <Select
            value={filter.shiny !== undefined ? String(filter.shiny) : 'none'}
            onValueChange={(v) => handleFilterUpdate({ shiny: v !== 'none' ? Number(v) as 0 | 1 | 2 : undefined })}
            disabled={isRunning }
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={eggSearchFilterLabels.noSelection[locale]} />
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
      </div>

      {/* めざパフィルター: 2列グリッド */}
      <div className="grid grid-cols-2 gap-2 mt-3">
        {/* めざパタイプフィルター */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">{eggSearchFilterLabels.hpType[locale]}</Label>
          <Select
            value={filter.hiddenPowerType !== undefined ? String(filter.hiddenPowerType) : 'none'}
            onValueChange={(v) => handleFilterUpdate({ hiddenPowerType: v !== 'none' ? Number(v) : undefined })}
            disabled={isRunning }
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={eggSearchFilterLabels.noSelection[locale]} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none" className="text-xs">{eggSearchFilterLabels.noSelection[locale]}</SelectItem>
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
          <Label className="text-xs">{eggSearchFilterLabels.hpPower[locale]}</Label>
          <Input
            type="number"
            min={30}
            max={70}
            value={filter.hiddenPowerPower ?? ''}
            onChange={(e) => {
              const v = e.target.value;
              handleFilterUpdate({ hiddenPowerPower: v ? Math.max(30, Math.min(70, parseInt(v))) : undefined });
            }}
            disabled={isRunning }
            placeholder={eggSearchFilterLabels.noSelection[locale]}
            className="text-xs h-8"
          />
        </div>
      </div>

      {/* IV範囲フィルター */}
      <section className="space-y-2 mt-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">
          {eggSearchFilterLabels.ivRange[locale]}
        </h4>
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
                  onFocus={(e) => e.target.select()}
                  disabled={isRunning  || isUnknown}
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
                  onFocus={(e) => e.target.select()}
                  disabled={isRunning  || isUnknown}
                  className="text-xs h-7 w-14 text-center"
                  placeholder={isUnknown ? '?' : 'max'}
                />
                <div className="flex items-center gap-1">
                  <Checkbox
                    id={`egg-search-filter-iv-unknown-${i}`}
                    checked={isUnknown}
                    onCheckedChange={(checked) => handleIvRangeUnknownChange(i, !!checked)}
                    disabled={isRunning }
                    className="h-3 w-3"
                  />
                  <Label htmlFor={`egg-search-filter-iv-unknown-${i}`} className="text-[10px] text-muted-foreground cursor-pointer">
                    {eggSearchFilterLabels.ivUnknown[locale]}
                  </Label>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </PanelCard>
  );
}
