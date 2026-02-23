/**
 * EggSearchFilterCard
 * 結果フィルターカード
 * 
 * 仕様: spec/agent/pr_egg_boot_timing_search/UI_DESIGN.md
 * EggFilterCardと同等の実装（個体値、性格、性別、特性、色違い、めざパ等）
 */

import React from 'react';
import { ArrowCounterClockwise, Funnel } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
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
import type { ShinyFilterMode } from '@/store/generation-store';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';
import {
  eggSearchFilterCardTitle,
  eggSearchFilterLabels,
  eggSearchStatNames,
  eggSearchGenderOptions,
  eggSearchAbilityOptions,
  eggSearchShinyModeOptions,
  eggSearchFilterResetLabel,
} from '@/lib/i18n/strings/egg-search';

export function EggSearchFilterCard() {
  const locale = useLocale();
  const { isStack } = useResponsiveLayout();
  const { draftParams, updateFilter, updateResultFilters, resultFilters, status, resetFilters } = useEggBootTimingSearchStore();
  
  const isRunning = status === 'running' || status === 'starting' || status === 'stopping';
  
  const filter = draftParams.filter || createDefaultEggFilter();

  const genderOptions = resolveLocaleValue(eggSearchGenderOptions, locale);
  const abilityOptions = resolveLocaleValue(eggSearchAbilityOptions, locale);
  const shinyModeOptions = resolveLocaleValue(eggSearchShinyModeOptions, locale);
  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;
  const statNames = resolveLocaleValue(eggSearchStatNames, locale);
  const resetLabel = resolveLocaleValue(eggSearchFilterResetLabel, locale);

  // 現在の色違いフィルターモード (filter.shinyFilterMode を優先)
  const currentShinyMode: ShinyFilterMode = filter.shinyFilterMode ?? resultFilters.shinyFilterMode ?? 'all';

  /**
   * フィルター更新ハンドラ
   * 検索パラメータ (filter) と結果表示フィルター (resultFilters) の両方に設定
   */
  const handleFilterUpdate = (updates: Partial<EggIndividualFilter>) => {
    // 検索パラメータに設定 → WASM側でフィルタリング
    updateFilter(updates);
    
    // 結果表示用フィルターにも設定 → クライアント側フィルタリング
    // EggIndividualFilter から CommonEggResultFilters への変換
    const resultFilterUpdates: Record<string, unknown> = {};
    if (updates.shinyFilterMode !== undefined) {
      resultFilterUpdates.shinyFilterMode = updates.shinyFilterMode;
    }
    if (updates.nature !== undefined) {
      resultFilterUpdates.nature = updates.nature;
    }
    if (updates.gender !== undefined) {
      resultFilterUpdates.gender = updates.gender;
    }
    if (updates.ability !== undefined) {
      resultFilterUpdates.ability = updates.ability;
    }
    if (updates.hiddenPowerType !== undefined) {
      resultFilterUpdates.hiddenPowerType = updates.hiddenPowerType;
    }
    if (updates.hiddenPowerPower !== undefined) {
      resultFilterUpdates.hiddenPowerPower = updates.hiddenPowerPower;
    }
    if (updates.ivRanges !== undefined) {
      resultFilterUpdates.ivRanges = updates.ivRanges;
    }
    if (Object.keys(resultFilterUpdates).length > 0) {
      updateResultFilters(resultFilterUpdates);
    }
  };

  const handleIvRangeChange = (
    statIndex: number,
    minMax: 'min' | 'max',
    value: string
  ) => {
    const newRanges = [...filter.ivRanges] as [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
    // 空入力は 0 として扱い、入力は許容する
    const numValue = value === '' ? 0 : parseInt(value, 10);
    if (!Number.isNaN(numValue)) {
      newRanges[statIndex] = {
        ...newRanges[statIndex],
        [minMax]: numValue,
      };
      handleFilterUpdate({ ivRanges: newRanges });
    }
  };

  // IV範囲フォーカスアウト時のバリデーション
  const handleIvRangeBlur = (
    statIndex: number,
    minMax: 'min' | 'max'
  ) => {
    const newRanges = [...filter.ivRanges] as [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
    const range = newRanges[statIndex];
    // 32は「不明」を表す特殊値
    if (range.max === 32) return;

    const clampedMin = Math.max(0, Math.min(31, range.min));
    const clampedMax = Math.max(0, Math.min(31, range.max));
    
    // min > max の場合は補正
    let finalMin = clampedMin;
    let finalMax = clampedMax;
    if (finalMin > finalMax) {
      if (minMax === 'min') {
        finalMax = finalMin;
      } else {
        finalMin = finalMax;
      }
    }
    
    newRanges[statIndex] = { min: finalMin, max: finalMax };
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
      headerActions={
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={resetFilters}
          className="gap-1"
        >
          <ArrowCounterClockwise size={14} />
          {resetLabel}
        </Button>
      }
      className={isStack ? 'min-h-[480px]' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
    >
      {/* 特性・性別・性格・色違い: 2列グリッド */}
      <div className="grid grid-cols-2 gap-2">
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

        {/* 色違いフィルター */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">{eggSearchFilterLabels.shiny[locale]}</Label>
          <Select
            value={currentShinyMode}
            onValueChange={(v) => handleFilterUpdate({ shinyFilterMode: v as ShinyFilterMode })}
            disabled={isRunning }
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={eggSearchFilterLabels.noSelection[locale]} />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(shinyModeOptions) as ShinyFilterMode[]).map((mode) => (
                <SelectItem key={mode} value={mode} className="text-xs">
                  {shinyModeOptions[mode]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

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
              handleFilterUpdate({ hiddenPowerPower: v ? parseInt(v) : undefined });
            }}
            onBlur={() => {
              if (filter.hiddenPowerPower !== undefined) {
                handleFilterUpdate({ hiddenPowerPower: Math.max(30, Math.min(70, filter.hiddenPowerPower)) });
              }
            }}
            disabled={isRunning }
            placeholder={eggSearchFilterLabels.noSelection[locale]}
            className="text-xs h-8"
          />
        </div>

        {/* Timer0 フィルター */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="egg-search-filter-timer0">{eggSearchFilterLabels.timer0Range[locale]}</Label>
          <Input
            id="egg-search-filter-timer0"
            value={resultFilters.timer0Filter ?? ''}
            onChange={(e) => updateResultFilters({ timer0Filter: e.target.value.replace(/[^0-9a-fA-F]/g, '').toUpperCase() })}
            disabled={isRunning}
            placeholder="例: 10ED"
            className="font-mono text-xs h-8"
          />
        </div>

        {/* VCount フィルター */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="egg-search-filter-vcount">{eggSearchFilterLabels.vcountRange[locale]}</Label>
          <Input
            id="egg-search-filter-vcount"
            value={resultFilters.vcountFilter ?? ''}
            onChange={(e) => updateResultFilters({ vcountFilter: e.target.value.replace(/[^0-9a-fA-F]/g, '').toUpperCase() })}
            disabled={isRunning}
            placeholder="例: 5C"
            className="font-mono text-xs h-8"
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
                  onChange={(e) => handleIvRangeChange(i, 'min', e.target.value)}
                  onBlur={() => handleIvRangeBlur(i, 'min')}
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
                  onChange={(e) => handleIvRangeChange(i, 'max', e.target.value)}
                  onBlur={() => handleIvRangeBlur(i, 'max')}
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
