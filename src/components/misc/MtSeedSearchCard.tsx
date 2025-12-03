/**
 * MtSeedSearchCard.tsx
 * MT Seed 32bit全探索カード
 * Card内で閉じる形でパラメータ入力・検索起動・結果表示を提供
 */

import React, { useCallback, useMemo } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from '@/components/ui/select';
import {
  MagnifyingGlass,
  Play,
  Stop,
  Pause,
  ArrowCounterClockwise,
  Copy,
} from '@phosphor-icons/react';
import { useMtSeedSearchStore } from '@/store/mt-seed-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';
import {
  formatRunProgressPercent,
  formatRunProgressCount,
} from '@/lib/i18n/strings/run-progress';
import {
  mtSeedSearchCardTitle,
  mtSeedSearchStatusPrefix,
  mtSeedSearchButtonLabels,
  mtSeedSearchParamLabels,
  mtSeedSearchStatNames,
  getMtSeedSearchStatusLabel,
  getMtSeedSearchModeLabel,
  type MtSeedSearchMode,
} from '@/lib/i18n/strings/mt-seed-search';

export const MtSeedSearchCard: React.FC = () => {
  const locale = useLocale();
  const {
    draftParams,
    updateDraftParams,
    updateIvRange,
    updateFilter,
    validationErrors,
    status,
    progress,
    results,
    errorMessage,
    startSearch,
    pauseSearch,
    resumeSearch,
    stopSearch,
    reset,
  } = useMtSeedSearchStore();

  const isDisabled = status === 'running' || status === 'starting' || status === 'stopping';
  const canStart = status === 'idle' || status === 'completed' || status === 'error';
  const canPause = status === 'running';
  const canResume = status === 'paused';
  const canStop = status === 'running' || status === 'paused';

  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;
  const statNameList = mtSeedSearchStatNames[locale] ?? mtSeedSearchStatNames.en;

  // IV範囲変更ハンドラ
  const handleIvRangeChange = useCallback(
    (statIndex: number, minMax: 'min' | 'max', value: string) => {
      const numValue = value === '' ? 0 : parseInt(value, 10);
      if (!Number.isNaN(numValue)) {
        updateIvRange(statIndex, minMax, Math.max(0, Math.min(31, numValue)));
      }
    },
    [updateIvRange]
  );

  // 結果テキスト生成（0xなしMT Seed改行区切り）
  const resultText = useMemo(() => {
    return results.map((m) => m.mtSeed.toString(16).toUpperCase().padStart(8, '0')).join('\n');
  }, [results]);

  // コピーハンドラ
  const handleCopy = useCallback(() => {
    if (resultText) {
      navigator.clipboard.writeText(resultText);
    }
  }, [resultText]);

  // 検索開始ハンドラ
  const handleStart = useCallback(async () => {
    await startSearch();
  }, [startSearch]);

  // 進捗表示（他のRunCardと同様の形式）
  const progressPercent = progress?.progressPercent ?? 0;
  const matchesFound = progress?.matchesFound ?? results.length;
  const maxResults = 2 ** 32; // MT Seedは32bit全探索
  const percentDisplay = formatRunProgressPercent(progressPercent, locale);
  const countDisplay = formatRunProgressCount(matchesFound, maxResults, locale);

  return (
    <PanelCard
      icon={<MagnifyingGlass size={20} className="opacity-80" />}
      title={<span>{mtSeedSearchCardTitle[locale]}</span>}
      headerActions={
        <div className="flex items-center gap-1">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={handleCopy}
            disabled={results.length === 0}
            className="gap-1 h-7 px-2"
            title={mtSeedSearchButtonLabels.copy[locale]}
          >
            <Copy size={14} />
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={reset}
            disabled={isDisabled}
            className="gap-1 h-7 px-2"
          >
            <ArrowCounterClockwise size={14} />
            {mtSeedSearchButtonLabels.reset[locale]}
          </Button>
        </div>
      }
      fullHeight={false}
      scrollMode="parent"
      spacing="compact"
    >
      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="text-destructive text-xs space-y-0.5" role="alert">
          {validationErrors.map((e, i) => (
            <div key={i}>{e}</div>
          ))}
        </div>
      )}

      {/* Error Message */}
      {errorMessage && (
        <div className="text-destructive text-xs" role="alert">
          {errorMessage}
        </div>
      )}

      {/* 検索ボタン・ステータス */}
      <div className="flex items-center gap-2 flex-wrap">
        {canStart && (
          <Button
            size="sm"
            onClick={handleStart}
            className="flex-1"
          >
            <Play size={16} className="mr-2" />
            {mtSeedSearchButtonLabels.start[locale]}
          </Button>
        )}
        {status === 'starting' && (
          <Button size="sm" disabled className="flex-1">
            <Play size={16} className="mr-2" />
            {mtSeedSearchButtonLabels.starting[locale]}
          </Button>
        )}
        {canPause && (
          <Button size="sm" variant="outline" onClick={pauseSearch} className="flex-1">
            <Pause size={16} className="mr-2" />
            {mtSeedSearchButtonLabels.pause[locale]}
          </Button>
        )}
        {canResume && (
          <Button size="sm" onClick={resumeSearch} className="flex-1">
            <Play size={16} className="mr-2" />
            {mtSeedSearchButtonLabels.resume[locale]}
          </Button>
        )}
        {canStop && (
          <Button size="sm" variant="destructive" onClick={stopSearch}>
            <Stop size={16} className="mr-2" />
            {mtSeedSearchButtonLabels.stop[locale]}
          </Button>
        )}
        <div className="text-xs text-muted-foreground ml-auto">
          {mtSeedSearchStatusPrefix[locale]}: {getMtSeedSearchStatusLabel(status, locale)}
          {progress && ` (${getMtSeedSearchModeLabel(progress.mode as MtSeedSearchMode, locale)})`}
        </div>
      </div>

      {/* 進捗バー - running/paused/stopping時のみ表示 */}
      {(status === 'running' || status === 'paused' || status === 'stopping') && (
        <Progress value={progressPercent} className="h-2" />
      )}

      {/* Result summary - 常に表示 */}
      <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono flex-wrap gap-x-2">
        <span>{percentDisplay}</span>
        <span>{countDisplay}</span>
      </div>

      {/* MT消費数 + 徘徊チェックボックス */}
      <div className="space-y-2">
        <Label className="text-xs font-medium">{mtSeedSearchParamLabels.mtAdvances[locale]}</Label>
        <div className="flex items-center gap-4">
          <Input
            type="number"
            min={0}
            value={draftParams.mtAdvances}
            onChange={(e) =>
              updateDraftParams({
                mtAdvances: Math.max(0, parseInt(e.target.value) || 0),
              })
            }
            disabled={isDisabled}
            className="w-20 h-7 text-xs"
          />
          <div className="flex items-center gap-1.5">
            <Checkbox
              id="mt-seed-roamer"
              checked={draftParams.isRoamer}
              onCheckedChange={(checked) =>
                updateDraftParams({ isRoamer: !!checked })
              }
              disabled={isDisabled}
            />
            <Label htmlFor="mt-seed-roamer" className="text-xs cursor-pointer">
              {mtSeedSearchParamLabels.roamer[locale]}
            </Label>
          </div>
        </div>
      </div>

      {/* IV範囲 */}
      <div className="space-y-2">
        <Label className="text-xs font-medium">{mtSeedSearchParamLabels.ivRanges[locale]}</Label>
        <div className="space-y-1.5">
          {statNameList.map((stat, i) => {
            const range = draftParams.filter.ivRanges[i];
            return (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs w-12 text-muted-foreground">{stat}</span>
                <Input
                  type="number"
                  min={0}
                  max={31}
                  value={range.min}
                  onChange={(e) => handleIvRangeChange(i, 'min', e.target.value)}
                  disabled={isDisabled}
                  className="text-xs h-7 w-14 text-center"
                />
                <span className="text-xs text-muted-foreground">~</span>
                <Input
                  type="number"
                  min={0}
                  max={31}
                  value={range.max}
                  onChange={(e) => handleIvRangeChange(i, 'max', e.target.value)}
                  disabled={isDisabled}
                  className="text-xs h-7 w-14 text-center"
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* めざパ設定 */}
      <div className="grid grid-cols-2 gap-2">
        {/* めざパタイプ */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">{mtSeedSearchParamLabels.hpType[locale]}</Label>
          <Select
            value={
              draftParams.filter.hiddenPowerType !== undefined
                ? String(draftParams.filter.hiddenPowerType)
                : 'none'
            }
            onValueChange={(v) =>
              updateFilter({
                hiddenPowerType: v !== 'none' ? Number(v) : undefined,
              })
            }
            disabled={isDisabled}
          >
            <SelectTrigger className="text-xs h-8">
              <SelectValue placeholder={mtSeedSearchParamLabels.noSelection[locale]} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none" className="text-xs">
                {mtSeedSearchParamLabels.noSelection[locale]}
              </SelectItem>
              {hpTypeNames.map((name, i) => (
                <SelectItem key={i} value={String(i)} className="text-xs">
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* めざパ威力 */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">{mtSeedSearchParamLabels.hpPower[locale]}</Label>
          <Input
            type="number"
            min={30}
            max={70}
            value={draftParams.filter.hiddenPowerPower ?? ''}
            onChange={(e) => {
              const v = e.target.value;
              updateFilter({
                hiddenPowerPower: v
                  ? Math.max(30, Math.min(70, parseInt(v)))
                  : undefined,
              });
            }}
            disabled={isDisabled}
            placeholder={mtSeedSearchParamLabels.noSelection[locale]}
            className="text-xs h-8"
          />
        </div>
      </div>

      {/* 結果表示 */}
      <div className="space-y-1">
        <Label className="text-xs font-medium">
          {mtSeedSearchParamLabels.results[locale]} ({results.length})
        </Label>
        <Textarea
          readOnly
          value={resultText}
          placeholder="--"
          className="font-mono text-xs h-32 resize-none"
        />
      </div>
    </PanelCard>
  );
};
