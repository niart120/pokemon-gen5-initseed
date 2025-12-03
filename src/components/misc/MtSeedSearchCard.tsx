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
import { useMtSeedSearchStore} from '@/store/mt-seed-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';

// === ローカライズ定数 ===
const statNames = {
  ja: ['HP', 'こうげき', 'ぼうぎょ', 'とくこう', 'とくぼう', 'すばやさ'],
  en: ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'],
};

const labels = {
  title: { ja: 'MT Seed 検索', en: 'MT Seed Search' },
  mtAdvances: { ja: 'MT消費数', en: 'MT Advances' },
  roamer: { ja: '徘徊', en: 'Roamer' },
  ivRanges: { ja: 'IV範囲', en: 'IV Ranges' },
  hpType: { ja: 'めざパタイプ', en: 'HP Type' },
  hpPower: { ja: 'めざパ威力', en: 'HP Power' },
  noSelection: { ja: '指定なし', en: 'None' },
  start: { ja: '検索開始', en: 'Start' },
  stop: { ja: '停止', en: 'Stop' },
  pause: { ja: '一時停止', en: 'Pause' },
  resume: { ja: '再開', en: 'Resume' },
  reset: { ja: 'リセット', en: 'Reset' },
  copy: { ja: 'コピー', en: 'Copy' },
  results: { ja: '検索結果', en: 'Results' },
  progress: { ja: '進捗', en: 'Progress' },
  status: {
    idle: { ja: '待機中', en: 'Idle' },
    starting: { ja: '開始中...', en: 'Starting...' },
    running: { ja: '検索中', en: 'Running' },
    stopping: { ja: '停止中...', en: 'Stopping...' },
    paused: { ja: '一時停止中', en: 'Paused' },
    completed: { ja: '完了', en: 'Completed' },
    error: { ja: 'エラー', en: 'Error' },
  },
  mode: {
    gpu: { ja: 'GPU', en: 'GPU' },
    cpu: { ja: 'CPU', en: 'CPU' },
  },
};

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
  const statNameList = statNames[locale] ?? statNames.en;

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

  // 進捗表示
  const progressPercent = progress?.progressPercent ?? 0;
  const progressText = progress
    ? `${progressPercent.toFixed(1)}% (${progress.matchesFound} found)`
    : '';

  return (
    <PanelCard
      icon={<MagnifyingGlass size={20} className="opacity-80" />}
      title={<span>{labels.title[locale]}</span>}
      headerActions={
        <div className="flex items-center gap-1">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={handleCopy}
            disabled={results.length === 0}
            className="gap-1 h-7 px-2"
            title={labels.copy[locale]}
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
            {labels.reset[locale]}
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
              <Play size={16} className="mr-1" />
              {labels.start[locale]}
            </Button>
          )}
          {status === 'starting' && (
            <Button size="sm" disabled className="flex-1">
              <Play size={16} className="mr-1" />
              {labels.status.starting[locale]}
            </Button>
          )}
          {canPause && (
            <Button size="sm" variant="outline" onClick={pauseSearch}>
              <Pause size={16} className="mr-1" />
              {labels.pause[locale]}
            </Button>
          )}
          {canResume && (
            <Button size="sm" variant="outline" onClick={resumeSearch}>
              <Play size={16} className="mr-1" />
              {labels.resume[locale]}
            </Button>
          )}
          {canStop && (
            <Button size="sm" variant="destructive" onClick={stopSearch}>
              <Stop size={16} className="mr-1" />
              {labels.stop[locale]}
            </Button>
          )}
        <div className="text-xs text-muted-foreground ml-auto">
          {labels.status[status][locale]}
          {progress && ` (${labels.mode[progress.mode][locale]})`}
        </div>
      </div>

      {/* 進捗バー */}
      {(status === 'running' || status === 'paused' || status === 'stopping') && (
        <div className="space-y-1">
          <Progress value={progressPercent} className="h-2" />
          <div className="text-xs text-muted-foreground text-right">{progressText}</div>
        </div>
      )}

      {/* MT消費数 + 徘徊チェックボックス */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">{labels.mtAdvances[locale]}</Label>
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
                {labels.roamer[locale]}
              </Label>
            </div>
          </div>
        </div>

        {/* IV範囲 */}
        <div className="space-y-2">
          <Label className="text-xs font-medium">{labels.ivRanges[locale]}</Label>
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
            <Label className="text-xs">{labels.hpType[locale]}</Label>
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
                <SelectValue placeholder={labels.noSelection[locale]} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none" className="text-xs">
                  {labels.noSelection[locale]}
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
            <Label className="text-xs">{labels.hpPower[locale]}</Label>
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
              placeholder={labels.noSelection[locale]}
              className="text-xs h-8"
            />
          </div>
        </div>

        {/* 結果表示 */}
        <div className="space-y-1">
          <Label className="text-xs font-medium">
            {labels.results[locale]} ({results.length})
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
