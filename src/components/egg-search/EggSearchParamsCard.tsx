/**
 * EggSearchParamsCard
 * 検索条件パラメータ入力カード
 * 
 * 仕様: spec/agent/pr_egg_boot_timing_search/UI_DESIGN.md
 * - 日付範囲をSearchPanelと同様に開始日・終了日 + 時分秒レンジ指定
 * - 親個体値・生成条件はEggParamsCardと同等
 * - Timer0/VCountはProfileから自動取得
 */

import React, { useMemo } from 'react';
import { Sliders, GameController } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { KeyInputDialog } from '@/components/keys';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName } from '@/lib/utils/format-display';
import { DOMAIN_NATURE_COUNT } from '@/types/domain';
import { KEY_INPUT_DEFAULT, keyMaskToNames, toggleKeyInMask, type KeyName } from '@/lib/utils/key-input';
import type { IvSet } from '@/types/egg';
import {
  eggSearchParamsCardTitle,
  eggSearchParamsSectionTitles,
  eggSearchParamsLabels,
  eggSearchStatNames,
  eggSearchFemaleAbilityOptions,
  eggSearchGenderRatioPresets,
} from '@/lib/i18n/strings/egg-search';

export function EggSearchParamsCard() {
  const locale = useLocale();
  const {
    draftParams,
    updateDraftParams,
    updateDraftConditions,
    updateDraftParentsMale,
    updateDraftParentsFemale,
    updateDateRange,
    updateTimeRange,
    status,
  } = useEggBootTimingSearchStore();
  
  const isRunning = status === 'running' || status === 'starting' || status === 'stopping';

  const [isKeyDialogOpen, setIsKeyDialogOpen] = React.useState(false);
  const [tempKeyInput, setTempKeyInput] = React.useState(KEY_INPUT_DEFAULT);

  const statNames = resolveLocaleValue(eggSearchStatNames, locale);
  const femaleAbilityOptions = resolveLocaleValue(eggSearchFemaleAbilityOptions, locale);

  // 性別比プリセットの選択値を計算
  const genderRatioValue = useMemo(() => {
    const { threshold, genderless } = draftParams.conditions.genderRatio;
    const preset = eggSearchGenderRatioPresets.find(
      p => p.threshold === threshold && p.genderless === genderless
    );
    return preset ? `${preset.threshold}-${preset.genderless}` : 'custom';
  }, [draftParams.conditions.genderRatio]);

  // 日付フォーマット
  const formatDate = (year: number, month: number, day: number): string => {
    return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
  };

  const parseDate = (dateString: string) => {
    const date = new Date(dateString);
    return {
      year: date.getFullYear(),
      month: date.getMonth() + 1,
      day: date.getDate(),
    };
  };

  const startDate = formatDate(
    draftParams.dateRange.startYear,
    draftParams.dateRange.startMonth,
    draftParams.dateRange.startDay,
  );

  const endDate = formatDate(
    draftParams.dateRange.endYear,
    draftParams.dateRange.endMonth,
    draftParams.dateRange.endDay,
  );

  const handleStartDateChange = (dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    updateDateRange({
      startYear: year,
      startMonth: month,
      startDay: day,
    });
  };

  const handleEndDateChange = (dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    updateDateRange({
      endYear: year,
      endMonth: month,
      endDay: day,
    });
  };

  // 時刻範囲フィールド設定
  const timeFieldConfigs = [
    { key: 'hour' as const, label: eggSearchParamsLabels.hour[locale], min: 0, max: 23 },
    { key: 'minute' as const, label: eggSearchParamsLabels.minute[locale], min: 0, max: 59 },
    { key: 'second' as const, label: eggSearchParamsLabels.second[locale], min: 0, max: 59 },
  ];

  const handleTimeRangeChange = (
    field: 'hour' | 'minute' | 'second',
    edge: 'start' | 'end',
    rawValue: string,
  ) => {
    // 入力中はバリデーションせず、そのまま保存
    const currentRange = draftParams.timeRange[field];
    const nextRange = { ...currentRange, [edge]: rawValue };
    updateTimeRange({ [field]: nextRange });
  };

  const handleTimeRangeBlur = (
    field: 'hour' | 'minute' | 'second',
    edge: 'start' | 'end',
  ) => {
    const range = draftParams.timeRange[field];
    const config = timeFieldConfigs.find(c => c.key === field)!;

    // 空の場合やNaNの場合はminに補正
    const startValue = typeof range.start === 'string' ? parseInt(range.start as string, 10) : range.start;
    const endValue = typeof range.end === 'string' ? parseInt(range.end as string, 10) : range.end;
    
    const clampedStart = Number.isNaN(startValue) ? config.min : Math.min(Math.max(startValue, config.min), config.max);
    const clampedEnd = Number.isNaN(endValue) ? config.min : Math.min(Math.max(endValue, config.min), config.max);

    // start > end の場合は補正
    let finalStart = clampedStart;
    let finalEnd = clampedEnd;
    if (finalStart > finalEnd) {
      if (edge === 'start') {
        finalEnd = finalStart;
      } else {
        finalStart = finalEnd;
      }
    }

    updateTimeRange({ [field]: { start: finalStart, end: finalEnd } });
  };

  // キー入力
  const availableKeys = useMemo(() => keyMaskToNames(draftParams.keyInputMask), [draftParams.keyInputMask]);
  const tempAvailableKeys = useMemo(() => keyMaskToNames(tempKeyInput), [tempKeyInput]);

  const handleToggleKey = (key: KeyName) => {
    setTempKeyInput(toggleKeyInMask(tempKeyInput, key));
  };

  const handleResetKeys = () => {
    setTempKeyInput(KEY_INPUT_DEFAULT);
  };

  const handleApplyKeys = () => {
    updateDraftParams({ keyInputMask: tempKeyInput });
    setIsKeyDialogOpen(false);
  };

  const openKeyDialog = () => {
    setTempKeyInput(draftParams.keyInputMask);
    setIsKeyDialogOpen(true);
  };

  const keyJoiner = locale === 'ja' ? '、' : ', ';

  // 親IV変更ハンドラ（入力中はバリデーションなし）
  const handleIvChange = (
    parent: 'male' | 'female',
    index: number,
    value: string
  ) => {
    // 入力中はそのまま保存（空入力は0として扱う）
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const newIvs = [...currentIvs] as IvSet;
    const numValue = parseInt(value, 10);
    newIvs[index] = Number.isNaN(numValue) ? 0 : numValue;

    if (parent === 'male') {
      updateDraftParentsMale(newIvs);
    } else {
      updateDraftParentsFemale(newIvs);
    }
  };

  // 親IVフォーカスアウト時のバリデーション
  const handleIvBlur = (
    parent: 'male' | 'female',
    index: number
  ) => {
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const currentValue = currentIvs[index];
    
    // 0-31にクランプ
    const clampedValue = Math.min(31, Math.max(0, currentValue));
    if (clampedValue !== currentValue) {
      const newIvs = [...currentIvs] as IvSet;
      newIvs[index] = clampedValue;
      if (parent === 'male') {
        updateDraftParentsMale(newIvs);
      } else {
        updateDraftParentsFemale(newIvs);
      }
    }
  };

  const handleIvUnknownChange = (
    parent: 'male' | 'female',
    index: number,
    isUnknown: boolean
  ) => {
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const newIvs = [...currentIvs] as IvSet;
    newIvs[index] = isUnknown ? 32 : 0;

    if (parent === 'male') {
      updateDraftParentsMale(newIvs);
    } else {
      updateDraftParentsFemale(newIvs);
    }
  };

  // 性別比変更ハンドラ
  const handleGenderRatioChange = (value: string) => {
    const preset = eggSearchGenderRatioPresets.find(
      p => `${p.threshold}-${p.genderless}` === value
    );
    if (preset) {
      updateDraftConditions({
        genderRatio: {
          threshold: preset.threshold,
          genderless: preset.genderless,
        },
      });
    }
  };

  const timeInputClassName = 'h-8 w-11 px-0 text-center text-sm';

  return (
    <>
      <PanelCard
        icon={<Sliders size={20} className="opacity-80" />}
        title={eggSearchParamsCardTitle[locale]}
        scrollMode="content"
        fullHeight
      >
        <div className="space-y-4">
          {/* 範囲セクション（日時範囲 + 消費範囲を統合） */}
          <section className="space-y-3" role="group">
            <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">
              {eggSearchParamsSectionTitles.range[locale]}
            </h4>
            
            {/* 日付範囲 */}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label htmlFor="start-date" className="text-xs">{eggSearchParamsLabels.startDate[locale]}</Label>
                <Input
                  id="start-date"
                  type="date"
                  min="2000-01-01"
                  max="2099-12-31"
                  className="h-8 text-xs"
                  value={startDate}
                  onChange={(e) => handleStartDateChange(e.target.value)}
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="end-date" className="text-xs">{eggSearchParamsLabels.endDate[locale]}</Label>
                <Input
                  id="end-date"
                  type="date"
                  min="2000-01-01"
                  max="2099-12-31"
                  className="h-8 text-xs"
                  value={endDate}
                  onChange={(e) => handleEndDateChange(e.target.value)}
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* 時刻範囲 */}
            <div className="space-y-1">
              <Label className="text-xs">{eggSearchParamsLabels.timeRange[locale]}</Label>
              <div className="flex items-center gap-0 overflow-x-auto">
                {timeFieldConfigs.map((config) => {
                  const range = draftParams.timeRange[config.key];
                  return (
                    <div key={config.key} className="flex items-center gap-0 whitespace-nowrap">
                      <span className="text-xs text-muted-foreground w-8 text-right">
                        {config.label}
                      </span>
                      <Input
                        type="number"
                        inputMode="numeric"
                        min={config.min}
                        max={config.max}
                        value={range.start}
                        aria-label={`${config.label} min`}
                        className={timeInputClassName}
                        onChange={(e) => handleTimeRangeChange(config.key, 'start', e.target.value)}
                        onBlur={() => handleTimeRangeBlur(config.key, 'start')}
                        disabled={isRunning}
                      />
                      <span className="text-xs text-muted-foreground">~</span>
                      <Input
                        type="number"
                        inputMode="numeric"
                        min={config.min}
                        max={config.max}
                        value={range.end}
                        aria-label={`${config.label} max`}
                        className={timeInputClassName}
                        onChange={(e) => handleTimeRangeChange(config.key, 'end', e.target.value)}
                        onBlur={() => handleTimeRangeBlur(config.key, 'end')}
                        disabled={isRunning}
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 消費範囲 */}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label htmlFor="user-offset" className="text-xs">{eggSearchParamsLabels.userOffset[locale]}</Label>
                <Input
                  id="user-offset"
                  type="number"
                  min={0}
                  value={draftParams.userOffset}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    updateDraftParams({ userOffset: Number.isNaN(v) ? 0 : v });
                  }}
                  onBlur={() => {
                    const num = Math.max(0, draftParams.userOffset || 0);
                    updateDraftParams({ userOffset: num });
                  }}
                  disabled={isRunning}
                  className="h-8 text-xs"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="advance-count" className="text-xs">{eggSearchParamsLabels.advanceCount[locale]}</Label>
                <Input
                  id="advance-count"
                  type="number"
                  min={1}
                  max={100000}
                  value={draftParams.advanceCount}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    updateDraftParams({ advanceCount: Number.isNaN(v) ? 0 : v });
                  }}
                  onBlur={() => {
                    const num = Math.max(1, Math.min(100000, draftParams.advanceCount || 50));
                    updateDraftParams({ advanceCount: num });
                  }}
                  disabled={isRunning}
                  className="h-8 text-xs"
                />
              </div>
            </div>

            {/* キー入力 */}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <Label className="text-xs">{eggSearchParamsLabels.keyInput[locale]}</Label>
                <Button variant="outline" size="sm" onClick={openKeyDialog} className="gap-1 h-7 text-xs" disabled={isRunning}>
                  <GameController size={14} />
                  {eggSearchParamsLabels.keyInputConfigure[locale]}
                </Button>
              </div>
              {availableKeys.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  {availableKeys.join(keyJoiner)}
                </div>
              )}
            </div>
          </section>

          <Separator />

          {/* 親個体情報セクション */}
          <section className="space-y-3" role="group">
            <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">
              {eggSearchParamsSectionTitles.parents[locale]}
            </h4>

            {/* ♂親IV */}
            <div className="space-y-1">
              <Label className="text-xs">{eggSearchParamsLabels.maleParentIv[locale]}</Label>
              <div className="grid grid-cols-6 gap-1">
                {statNames.map((stat, i) => {
                  const isUnknown = draftParams.parents.male[i] === 32;
                  return (
                    <div key={i} className="flex flex-col items-center">
                      <span className="text-[10px] text-muted-foreground">{stat}</span>
                      <Input
                        type="number"
                        min={0}
                        max={31}
                        value={isUnknown ? '' : draftParams.parents.male[i]}
                        onChange={(e) => handleIvChange('male', i, e.target.value)}
                        onBlur={() => handleIvBlur('male', i)}
                        disabled={isRunning || isUnknown}
                        className="text-xs text-center h-7 px-1"
                        placeholder={isUnknown ? '?' : undefined}
                      />
                      <div className="flex items-center gap-1 mt-1">
                        <Checkbox
                          id={`egg-search-male-iv-unknown-${i}`}
                          checked={isUnknown}
                          onCheckedChange={(checked) => handleIvUnknownChange('male', i, !!checked)}
                          disabled={isRunning}
                          className="h-3 w-3"
                        />
                        <Label htmlFor={`egg-search-male-iv-unknown-${i}`} className="text-[9px] text-muted-foreground cursor-pointer">
                          {eggSearchParamsLabels.ivUnknown[locale]}
                        </Label>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* ♀親IV */}
            <div className="space-y-1">
              <Label className="text-xs">{eggSearchParamsLabels.femaleParentIv[locale]}</Label>
              <div className="grid grid-cols-6 gap-1">
                {statNames.map((stat, i) => {
                  const isUnknown = draftParams.parents.female[i] === 32;
                  return (
                    <div key={i} className="flex flex-col items-center">
                      <span className="text-[10px] text-muted-foreground">{stat}</span>
                      <Input
                        type="number"
                        min={0}
                        max={31}
                        value={isUnknown ? '' : draftParams.parents.female[i]}
                        onChange={(e) => handleIvChange('female', i, e.target.value)}
                        onBlur={() => handleIvBlur('female', i)}
                        disabled={isRunning || isUnknown}
                        className="text-xs text-center h-7 px-1"
                        placeholder={isUnknown ? '?' : undefined}
                      />
                      <div className="flex items-center gap-1 mt-1">
                        <Checkbox
                          id={`egg-search-female-iv-unknown-${i}`}
                          checked={isUnknown}
                          onCheckedChange={(checked) => handleIvUnknownChange('female', i, !!checked)}
                          disabled={isRunning}
                          className="h-3 w-3"
                        />
                        <Label htmlFor={`egg-search-female-iv-unknown-${i}`} className="text-[9px] text-muted-foreground cursor-pointer">
                          {eggSearchParamsLabels.ivUnknown[locale]}
                        </Label>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          <Separator />

          {/* 生成条件セクション */}
          <section className="space-y-3" role="group">
            <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">
              {eggSearchParamsSectionTitles.conditions[locale]}
            </h4>

            {/* セレクト系 */}
            <div className="grid grid-cols-3 gap-2">
              {/* 性別比 */}
              <div className="flex flex-col gap-1">
                <Label className="text-xs">{eggSearchParamsLabels.genderRatio[locale]}</Label>
                <Select
                  value={genderRatioValue}
                  onValueChange={handleGenderRatioChange}
                  disabled={isRunning}
                >
                  <SelectTrigger className="text-xs h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {eggSearchGenderRatioPresets.map((preset) => (
                      <SelectItem
                        key={`${preset.threshold}-${preset.genderless}`}
                        value={`${preset.threshold}-${preset.genderless}`}
                        className="text-xs"
                      >
                        {preset.label[locale]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* ♀親の特性 */}
              <div className="flex flex-col gap-1">
                <Label className="text-xs">{eggSearchParamsLabels.femaleAbility[locale]}</Label>
                <Select
                  value={String(draftParams.conditions.femaleParentAbility)}
                  onValueChange={(v) => updateDraftConditions({ femaleParentAbility: Number(v) as 0 | 1 | 2 })}
                  disabled={isRunning}
                >
                  <SelectTrigger className="text-xs h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {([0, 1, 2] as const).map((ability) => (
                      <SelectItem key={ability} value={String(ability)} className="text-xs">
                        {femaleAbilityOptions[ability]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* かわらずのいし */}
              <div className="flex flex-col gap-1">
                <Label className="text-xs">{eggSearchParamsLabels.everstone[locale]}</Label>
                <Select
                  value={draftParams.conditions.everstone.type === 'none' ? 'none' : `fixed-${(draftParams.conditions.everstone as { type: 'fixed'; nature: number }).nature}`}
                  onValueChange={(v) => {
                    if (v === 'none') {
                      updateDraftConditions({ everstone: { type: 'none' } });
                    } else {
                      const nature = parseInt(v.replace('fixed-', ''));
                      updateDraftConditions({ everstone: { type: 'fixed', nature } });
                    }
                  }}
                  disabled={isRunning}
                >
                  <SelectTrigger className="text-xs h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none" className="text-xs">{eggSearchParamsLabels.everstoneNone[locale]}</SelectItem>
                    {Array.from({ length: DOMAIN_NATURE_COUNT }, (_, i) => (
                      <SelectItem key={i} value={`fixed-${i}`} className="text-xs">
                        {natureName(i, locale)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* チェックボックス群 */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              {/* メタモン利用 */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id="egg-search-uses-ditto"
                  checked={draftParams.conditions.usesDitto}
                  onCheckedChange={(checked) => updateDraftConditions({ usesDitto: !!checked })}
                  disabled={isRunning}
                />
                <Label htmlFor="egg-search-uses-ditto" className="text-xs">{eggSearchParamsLabels.usesDitto[locale]}</Label>
              </div>

              {/* 国際孵化 */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id="egg-search-masuda"
                  checked={draftParams.conditions.masudaMethod}
                  onCheckedChange={(checked) => updateDraftConditions({ masudaMethod: !!checked })}
                  disabled={isRunning}
                />
                <Label htmlFor="egg-search-masuda" className="text-xs">{eggSearchParamsLabels.masudaMethod[locale]}</Label>
              </div>

              {/* ニドラン系 */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id="egg-search-nidoran"
                  checked={draftParams.conditions.hasNidoranFlag}
                  onCheckedChange={(checked) => updateDraftConditions({ hasNidoranFlag: !!checked })}
                  disabled={isRunning}
                />
                <Label htmlFor="egg-search-nidoran" className="text-xs">{eggSearchParamsLabels.nidoranFlag[locale]}</Label>
              </div>

              {/* NPC消費考慮 */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id="egg-search-npc-consumption"
                  checked={draftParams.considerNpcConsumption}
                  onCheckedChange={(checked) => updateDraftParams({ considerNpcConsumption: !!checked })}
                  disabled={isRunning}
                />
                <Label htmlFor="egg-search-npc-consumption" className="text-xs">{eggSearchParamsLabels.npcConsumption[locale]}</Label>
              </div>
            </div>
          </section>
        </div>
      </PanelCard>

      {/* キー入力ダイアログ */}
      <KeyInputDialog
        isOpen={isKeyDialogOpen}
        onOpenChange={setIsKeyDialogOpen}
        availableKeys={tempAvailableKeys}
        onToggleKey={handleToggleKey}
        onReset={handleResetKeys}
        onApply={handleApplyKeys}
        labels={{
          dialogTitle: eggSearchParamsLabels.keyInput[locale],
          reset: 'Reset',
          apply: 'Apply',
        }}
      />
    </>
  );
}
