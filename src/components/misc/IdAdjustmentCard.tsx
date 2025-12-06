/**
 * IdAdjustmentCard.tsx
 * ID調整機能のメインカードコンポーネント
 * 検索パラメータ入力、検索制御（開始/停止/一時停止/再開）、進捗表示、結果テーブルを統合
 * レイアウト: boot-timing系カード（EggSearchRunCard/GenerationRunCard）と同様のデザイン
 */

import React, { useEffect } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { RangeKeySection } from '@/components/shared/RangeKeySection';
import type { TimeFieldKey } from '@/components/shared/RangeKeySection';
import {
  IdentificationCard,
  Play,
  Square,
  Pause,
  Warning,
} from '@phosphor-icons/react';
import { useIdAdjustmentSearchStore } from '@/store/id-adjustment-search-store';
import { useAppStore } from '@/store/app-store';
import { useLocale } from '@/lib/i18n/locale-context';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { IdAdjustmentResultsTable } from './IdAdjustmentResultsTable';
import {
  formatRunProgressPercent,
  formatRunProgressCount,
} from '@/lib/i18n/strings/run-progress';
import {
  idAdjustmentCardTitle,
  idAdjustmentStatusPrefix,
  idAdjustmentButtonLabels,
  idAdjustmentControlsLabel,
  idAdjustmentResultsLabel,
  idAdjustmentBasicSettingLabel,
  idAdjustmentIdSettingLabel,
  idAdjustmentParamLabels,
  getIdAdjustmentStatusLabel,
} from '@/lib/i18n/strings/id-adjustment-search';
import {
  rangeKeyDialogLabels,
  rangeKeySectionLabels,
} from '@/lib/i18n/strings/range-key-section';

export const IdAdjustmentCard: React.FC = () => {
  const locale = useLocale();
  const { isStack } = useResponsiveLayout();
  const {
    draftParams,
    validationErrors,
    updateDraftParams,
    updateDateRange,
    updateTimeRange,
    applyProfile,
    status,
    progress,
    results,
    errorMessage,
    startSearch,
    pauseSearch,
    resumeSearch,
    stopSearch,
    validateDraft,
  } = useIdAdjustmentSearchStore();

  // Profile同期: アクティブプロファイル変更時にstore反映
  const profiles = useAppStore((s) => s.profiles);
  const activeProfileId = useAppStore((s) => s.activeProfileId);

  useEffect(() => {
    const profile = profiles.find((p) => p.id === activeProfileId) ?? profiles[0];
    if (profile) {
      applyProfile(profile);
    }
  }, [profiles, activeProfileId, applyProfile]);
  const handleDateChange = (edge: 'start' | 'end', value: { year: number; month: number; day: number }) => {
    if (edge === 'start') {
      updateDateRange({
        startYear: value.year,
        startMonth: value.month,
        startDay: value.day,
      });
    } else {
      updateDateRange({
        endYear: value.year,
        endMonth: value.month,
        endDay: value.day,
      });
    }
  };

  const handleTimeRangeChange = (
    field: TimeFieldKey,
    edge: 'start' | 'end',
    rawValue: string
  ) => {
    const currentRange = draftParams.timeRange[field];
    updateTimeRange({
      [field]: {
        ...currentRange,
        [edge]: rawValue,
      },
    });
  };

  const handleTimeRangeCommit = (field: TimeFieldKey, range: { start: number; end: number }) => {
    updateTimeRange({ [field]: range });
  };

  const handleKeyMaskChange = (mask: number) => {
    updateDraftParams({ keyInputMask: mask });
  };

  const labels = {
    startDate: rangeKeySectionLabels.startDate[locale],
    endDate: rangeKeySectionLabels.endDate[locale],
    timeRange: rangeKeySectionLabels.timeRange[locale],
    hour: rangeKeySectionLabels.hour[locale],
    minute: rangeKeySectionLabels.minute[locale],
    second: rangeKeySectionLabels.second[locale],
    keyInput: rangeKeySectionLabels.keyInput[locale],
    configure: rangeKeySectionLabels.configure[locale],
    dialogTitle: rangeKeyDialogLabels.title[locale],
    reset: rangeKeyDialogLabels.reset[locale],
    apply: rangeKeyDialogLabels.apply[locale],
  };

  const handleTidChange = (value: string) => {
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 65535) {
      updateDraftParams({ targetTid: numValue });
    } else if (value === '') {
      updateDraftParams({ targetTid: 0 });
    }
  };

  const handleSidChange = (value: string) => {
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 65535) {
      updateDraftParams({ targetSid: numValue });
    } else if (value === '') {
      updateDraftParams({ targetSid: 0 });
    }
  };

  const handleShinyPidChange = (value: string) => {
    if (value === '') {
      updateDraftParams({ shinyPid: null });
    } else {
      const numValue = parseInt(value, 16);
      if (!isNaN(numValue)) {
        updateDraftParams({ shinyPid: numValue });
      }
    }
  };

  // Search control state
  const isIdle = status === 'idle' || status === 'completed' || status === 'error';
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isStarting = status === 'starting';
  const isStopping = status === 'stopping';
  const isSearchActive = isRunning || isPaused || isStarting || isStopping;

  const handleStartSearch = async () => {
    if (validateDraft()) {
      await startSearch();
    }
  };

  // 進捗計算（boot-timing系と同様）
  const foundCount = progress?.foundCount ?? results.length;
  const maxResults = draftParams.maxResults ?? 1000;
  const pct = progress?.progressPercent ?? (maxResults > 0 ? (foundCount / maxResults) * 100 : 0);
  const statusDisplay = getIdAdjustmentStatusLabel(status, locale);
  const percentDisplay = formatRunProgressPercent(pct, locale);
  const countDisplay = formatRunProgressCount(foundCount, maxResults, locale);

  const statusPrefix = idAdjustmentStatusPrefix[locale];
  const controlsLabel = idAdjustmentControlsLabel[locale];
  const resultsLabel = idAdjustmentResultsLabel[locale];

  return (
    <PanelCard
      icon={<IdentificationCard size={20} className="opacity-80" />}
      title={<span id="id-adjustment-title">{idAdjustmentCardTitle[locale]}</span>}
      className={isStack ? 'max-h-[80vh]' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      role="region"
      aria-labelledby="id-adjustment-title"
    >
      {/* エラーメッセージ表示 */}
      {status === 'error' && errorMessage && (
        <div className="flex items-start gap-2 p-2 rounded-md bg-destructive/10 border border-destructive/20 text-xs">
          <Warning size={14} className="text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-destructive break-all">{errorMessage}</p>
        </div>
      )}

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="text-destructive text-xs space-y-0.5" role="alert">
          {validationErrors.map((e, i) => (
            <div key={i}>{e}</div>
          ))}
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap" role="group" aria-label={controlsLabel}>
            {isIdle && (
              <Button
                size="sm"
                onClick={handleStartSearch}
                disabled={validationErrors.length > 0}
                className="flex-1"
                data-testid="id-adjustment-start-btn"
              >
                <Play size={16} className="mr-2" />
                {isStarting
                  ? idAdjustmentButtonLabels.starting[locale]
                  : idAdjustmentButtonLabels.startSearch[locale]}
              </Button>
            )}

            {isRunning && (
              <>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={pauseSearch}
                  className="flex-1"
                >
                  <Pause size={16} className="mr-2" />
                  {idAdjustmentButtonLabels.pause[locale]}
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={stopSearch}
                  data-testid="id-adjustment-stop-btn"
                >
                  <Square size={16} className="mr-2" />
                  {idAdjustmentButtonLabels.stop[locale]}
                </Button>
              </>
            )}

            {isPaused && (
              <>
                <Button
                  size="sm"
                  onClick={resumeSearch}
                  className="flex-1"
                >
                  <Play size={16} className="mr-2" />
                  {idAdjustmentButtonLabels.resume[locale]}
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={stopSearch}
                >
                  <Square size={16} className="mr-2" />
                  {idAdjustmentButtonLabels.stop[locale]}
                </Button>
              </>
            )}

            {(isStarting || isStopping) && (
              <Button disabled className="flex-1" size="sm">
                <span className="animate-spin mr-2">⟳</span>
                {isStopping
                  ? idAdjustmentButtonLabels.stopping[locale]
                  : idAdjustmentButtonLabels.starting[locale]}
              </Button>
            )}

            <div className="text-xs text-muted-foreground ml-auto">
              {statusPrefix}: {statusDisplay}
            </div>
          </div>

          {/* Result summary - 1行表示: 12.3%  xxx / yyy results */}
          <div className="px-4 pb-2" aria-label={resultsLabel}>
            <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono flex-wrap gap-x-2">
              <span>{percentDisplay}</span>
              <span>{countDisplay}</span>
            </div>
          </div>

      {/* パラメータ入力セクション */}
      <Label className="text-xs text-muted-foreground">{idAdjustmentBasicSettingLabel[locale]}</Label>
          <div className="space-y-4">
            <RangeKeySection
              locale={locale}
              dateRange={draftParams.dateRange}
              timeRange={draftParams.timeRange}
              keyMask={draftParams.keyInputMask}
              labels={labels}
              isDisabled={isSearchActive}
              onDateChange={handleDateChange}
              onTimeChange={handleTimeRangeChange}
              onTimeCommit={handleTimeRangeCommit}
              onKeyMaskChange={handleKeyMaskChange}
            />
          </div>
        <Label className="text-xs text-muted-foreground">{idAdjustmentIdSettingLabel[locale]}</Label>
        {/* Target TID/SID + Shiny PID Filter (4-column grid) */}
        <div className="grid grid-cols-4 gap-2">
        <div className="space-y-1">
            <Label htmlFor="target-tid" className="text-xs">
            {idAdjustmentParamLabels.tid[locale]}
            </Label>
            <Input
            id="target-tid"
            type="number"
            inputMode="numeric"
            min={0}
            max={65535}
            value={draftParams.targetTid}
            onChange={(e) => handleTidChange(e.target.value)}
            className="h-9"
            disabled={isSearchActive}
            />
        </div>
        <div className="space-y-1">
            <Label htmlFor="target-sid" className="text-xs">
            {idAdjustmentParamLabels.sid[locale]}
            </Label>
            <Input
            id="target-sid"
            type="number"
            inputMode="numeric"
            min={0}
            max={65535}
            value={draftParams.targetSid ?? ''}
            onChange={(e) => handleSidChange(e.target.value)}
            className="h-9"
            disabled={isSearchActive}
            />
        </div>
        <div className="space-y-1">
            <div className="flex items-center h-9 gap-1.5">
            <Checkbox
                id="use-shiny-pid"
                checked={draftParams.shinyPid !== null}
                onCheckedChange={(checked) => {
                if (checked) {
                    updateDraftParams({ shinyPid: 0 });
                } else {
                    updateDraftParams({ shinyPid: null });
                }
                }}
                disabled={isSearchActive}
            />
            <Label htmlFor="use-shiny-pid" className="text-xs">
                {idAdjustmentParamLabels.shinyPid[locale]}
            </Label>
            </div>
        </div>
        <div className="space-y-1">
            <Label htmlFor="shiny-pid" className="text-xs text-muted-foreground">
            PID
            </Label>
            <Input
            id="shiny-pid"
            type="text"
            placeholder="XXXXXXXX"
            value={draftParams.shinyPid?.toString(16).toUpperCase().padStart(8, '0') ?? ''}
            onChange={(e) => handleShinyPidChange(e.target.value)}
            className="h-9 font-mono"
            maxLength={8}
            disabled={draftParams.shinyPid === null || isSearchActive}
            />
        </div>
      </div>

      {/* 結果テーブル（常に表示） */}
      <IdAdjustmentResultsTable />
    </PanelCard>
  );
};
