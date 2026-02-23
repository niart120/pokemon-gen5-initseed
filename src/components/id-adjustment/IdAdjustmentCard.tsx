/**
 * ID調整検索カードコンポーネント
 * 検索パラメータ入力、検索制御（開始/停止/一時停止/再開）、進捗表示を統合
 */
import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { KeyInputDialog } from '@/components/keys';
import {
  GameController,
  IdentificationCard,
  Play,
  Stop,
  Pause,
} from '@phosphor-icons/react';
import {
  KEY_INPUT_DEFAULT,
  keyMaskToNames,
  toggleKeyInMask,
  type KeyName,
} from '@/lib/utils/key-input';
import { useIdAdjustmentSearchStore } from '@/store/id-adjustment-search-store';
import { useLocale } from '@/lib/i18n/locale-context';

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function formatNumber(num: number): string {
  return num.toLocaleString();
}

export function IdAdjustmentCard() {
  const locale = useLocale();
  const {
    draftParams,
    validationErrors,
    updateDraftParams,
    updateDateRange,
    updateTimeRange,
    status,
    progress,
    startSearch,
    pauseSearch,
    resumeSearch,
    stopSearch,
    validateDraft,
  } = useIdAdjustmentSearchStore();

  const [isKeyDialogOpen, setIsKeyDialogOpen] = React.useState(false);
  const [tempKeyInput, setTempKeyInput] = React.useState(KEY_INPUT_DEFAULT);

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
    draftParams.dateRange.startDay
  );

  const endDate = formatDate(
    draftParams.dateRange.endYear,
    draftParams.dateRange.endMonth,
    draftParams.dateRange.endDay
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

  const availableKeys = React.useMemo(
    () => keyMaskToNames(draftParams.keyInputMask),
    [draftParams.keyInputMask]
  );
  const tempAvailableKeys = React.useMemo(
    () => keyMaskToNames(tempKeyInput),
    [tempKeyInput]
  );

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

  const timeFieldConfigs = [
    {
      key: 'hour' as const,
      label: locale === 'ja' ? '時' : 'Hour',
      min: 0,
      max: 23,
    },
    {
      key: 'minute' as const,
      label: locale === 'ja' ? '分' : 'Min',
      min: 0,
      max: 59,
    },
    {
      key: 'second' as const,
      label: locale === 'ja' ? '秒' : 'Sec',
      min: 0,
      max: 59,
    },
  ];

  const handleTimeRangeChange = (
    field: (typeof timeFieldConfigs)[number]['key'],
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

  const handleTimeRangeBlur = (
    field: (typeof timeFieldConfigs)[number]['key'],
    edge: 'start' | 'end'
  ) => {
    const range = draftParams.timeRange[field];
    const { min, max } = timeFieldConfigs.find(
      (config) => config.key === field
    )!;

    const startValue =
      typeof range.start === 'string' ? parseInt(range.start, 10) : range.start;
    const endValue =
      typeof range.end === 'string' ? parseInt(range.end, 10) : range.end;

    const clampedStart = Number.isNaN(startValue)
      ? min
      : Math.min(Math.max(startValue, min), max);
    const clampedEnd = Number.isNaN(endValue)
      ? min
      : Math.min(Math.max(endValue, min), max);

    let finalStart = clampedStart;
    let finalEnd = clampedEnd;
    if (finalStart > finalEnd) {
      if (edge === 'start') {
        finalEnd = finalStart;
      } else {
        finalStart = finalEnd;
      }
    }

    updateTimeRange({
      [field]: { start: finalStart, end: finalEnd },
    });
  };

  const timeInputClassName = 'h-8 w-11 px-0 text-center text-sm';

  const handleTidChange = (value: string) => {
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 65535) {
      updateDraftParams({ targetTid: numValue });
    } else if (value === '') {
      updateDraftParams({ targetTid: null });
    }
  };

  const handleSidChange = (value: string) => {
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 65535) {
      updateDraftParams({ targetSid: numValue });
    } else if (value === '') {
      updateDraftParams({ targetSid: null });
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

  const handleStartSearch = async () => {
    if (validateDraft()) {
      await startSearch();
    }
  };

  const progressPercent = progress?.progressPercent ?? 0;
  const processedCombinations = progress?.processedCombinations ?? 0;
  const totalCombinations = progress?.totalCombinations ?? 0;
  const foundCount = progress?.foundCount ?? 0;
  const elapsedMs = progress?.elapsedMs ?? 0;
  const estimatedRemainingMs = progress?.estimatedRemainingMs ?? 0;
  const activeWorkers = progress?.activeWorkers ?? 0;

  return (
    <PanelCard
      icon={<IdentificationCard size={20} className="opacity-80" />}
      title={locale === 'ja' ? 'ID調整検索' : 'ID Adjustment Search'}
      headerActions={
        activeWorkers > 0 && isRunning ? (
          <Badge variant="outline" className="text-xs">
            {activeWorkers} {locale === 'ja' ? 'ワーカー' : 'workers'}
          </Badge>
        ) : undefined
      }
    >
      <div className="space-y-4">
        {/* Target TID/SID */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {locale === 'ja' ? '目標ID' : 'Target IDs'}
          </Label>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <Label htmlFor="target-tid" className="text-xs text-muted-foreground">
                {locale === 'ja' ? '表ID (TID)' : 'Trainer ID (TID)'}
              </Label>
              <Input
                id="target-tid"
                type="number"
                inputMode="numeric"
                min={0}
                max={65535}
                value={draftParams.targetTid ?? ''}
                onChange={(e) => handleTidChange(e.target.value)}
                className="h-9"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="target-sid" className="text-xs text-muted-foreground">
                {locale === 'ja' ? '裏ID (SID)' : 'Secret ID (SID)'}
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
              />
            </div>
          </div>
        </div>

          {/* Shiny PID (optional) */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
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
              />
              <Label htmlFor="use-shiny-pid" className="text-sm font-medium">
                {locale === 'ja' ? '色違いPIDでフィルタ' : 'Filter by Shiny PID'}
              </Label>
            </div>
            {draftParams.shinyPid !== null && (
              <div className="space-y-1">
                <Label htmlFor="shiny-pid" className="text-xs text-muted-foreground">
                  {locale === 'ja' ? 'PID (16進数)' : 'PID (hex)'}
                </Label>
                <Input
                  id="shiny-pid"
                  type="text"
                  placeholder="XXXXXXXX"
                  value={draftParams.shinyPid?.toString(16).toUpperCase().padStart(8, '0') ?? ''}
                  onChange={(e) => handleShinyPidChange(e.target.value)}
                  className="h-9 font-mono"
                  maxLength={8}
                />
                <p className="text-xs text-muted-foreground">
                  {locale === 'ja'
                    ? '色違い (Square/Star) になる結果のみ表示'
                    : 'Only show results where PID is Shiny (Square/Star)'}
                </p>
              </div>
            )}
          </div>

          <Separator />

          {/* Date Range */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start-date" className="text-sm font-medium">
                {locale === 'ja' ? '開始日' : 'Start Date'}
              </Label>
              <Input
                id="start-date"
                type="date"
                min="2000-01-01"
                max="2099-12-31"
                className="h-9"
                value={startDate}
                onChange={(event) => handleStartDateChange(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="end-date" className="text-sm font-medium">
                {locale === 'ja' ? '終了日' : 'End Date'}
              </Label>
              <Input
                id="end-date"
                type="date"
                min="2000-01-01"
                max="2099-12-31"
                className="h-9"
                value={endDate}
                onChange={(event) => handleEndDateChange(event.target.value)}
              />
            </div>
          </div>

          {/* Time Range */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              {locale === 'ja' ? '時刻範囲' : 'Time Range'}
            </Label>
            <div className="flex items-center gap-0 overflow-x-auto">
              {timeFieldConfigs.map((config) => {
                const range = draftParams.timeRange[config.key];
                return (
                  <div
                    key={config.key}
                    className="flex items-center gap-0 whitespace-nowrap"
                  >
                    <span className="text-xs text-muted-foreground w-8 text-right">
                      {config.label}
                    </span>
                    <Input
                      type="number"
                      inputMode="numeric"
                      min={config.min}
                      max={config.max}
                      value={range.start}
                      aria-label={`${config.label} ${locale === 'ja' ? '最小' : 'min'}`}
                      className={timeInputClassName}
                      onChange={(event) =>
                        handleTimeRangeChange(config.key, 'start', event.target.value)
                      }
                      onBlur={() => handleTimeRangeBlur(config.key, 'start')}
                    />
                    <span className="text-xs text-muted-foreground">~</span>
                    <Input
                      type="number"
                      inputMode="numeric"
                      min={config.min}
                      max={config.max}
                      value={range.end}
                      aria-label={`${config.label} ${locale === 'ja' ? '最大' : 'max'}`}
                      className={timeInputClassName}
                      onChange={(event) =>
                        handleTimeRangeChange(config.key, 'end', event.target.value)
                      }
                      onBlur={() => handleTimeRangeBlur(config.key, 'end')}
                    />
                  </div>
                );
              })}
            </div>
          </div>

          <Separator />

          {/* Key Input */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground">
              {locale === 'ja' ? 'キー入力' : 'Key Input'}
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="flex-1 min-h-[2.25rem] rounded-md border bg-muted/40 px-3 py-2 text-xs font-mono">
                {availableKeys.length > 0 ? availableKeys.join(keyJoiner) : '—'}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={openKeyDialog}
                className="gap-2"
              >
                <GameController size={16} />
                {locale === 'ja' ? '設定' : 'Configure'}
              </Button>
            </div>
          </div>

          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              <ul className="list-disc list-inside space-y-1">
                {validationErrors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          )}

          <Separator />

          {/* Control Buttons */}
          <div className="flex gap-2">
            {isIdle && (
              <Button
                onClick={handleStartSearch}
                disabled={validationErrors.length > 0}
                className="flex-1 gap-2"
              >
                <Play size={16} weight="fill" />
                {locale === 'ja' ? '検索開始' : 'Start Search'}
              </Button>
            )}

            {isStarting && (
              <Button disabled className="flex-1 gap-2">
                <span className="animate-spin">⟳</span>
                {locale === 'ja' ? '開始中...' : 'Starting...'}
              </Button>
            )}

            {isRunning && (
              <>
                <Button
                  variant="outline"
                  onClick={pauseSearch}
                  className="flex-1 gap-2"
                >
                  <Pause size={16} weight="fill" />
                  {locale === 'ja' ? '一時停止' : 'Pause'}
                </Button>
                <Button
                  variant="destructive"
                  onClick={stopSearch}
                  className="flex-1 gap-2"
                >
                  <Stop size={16} weight="fill" />
                  {locale === 'ja' ? '停止' : 'Stop'}
                </Button>
              </>
            )}

            {isPaused && (
              <>
                <Button onClick={resumeSearch} className="flex-1 gap-2">
                  <Play size={16} weight="fill" />
                  {locale === 'ja' ? '再開' : 'Resume'}
                </Button>
                <Button
                  variant="destructive"
                  onClick={stopSearch}
                  className="flex-1 gap-2"
                >
                  <Stop size={16} weight="fill" />
                  {locale === 'ja' ? '停止' : 'Stop'}
                </Button>
              </>
            )}

            {isStopping && (
              <Button disabled className="flex-1 gap-2">
                <span className="animate-spin">⟳</span>
                {locale === 'ja' ? '停止中...' : 'Stopping...'}
              </Button>
            )}
          </div>

          {/* Progress Display */}
          {(isRunning || isPaused || isStopping) && progress && (
            <div className="space-y-3">
              <Progress value={progressPercent} className="h-2" />

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-xs text-muted-foreground">
                    {locale === 'ja' ? '進捗' : 'Progress'}
                  </div>
                  <div className="font-mono">
                    {formatNumber(processedCombinations)} /{' '}
                    {formatNumber(totalCombinations)}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {progressPercent.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">
                    {locale === 'ja' ? '発見数' : 'Found'}
                  </div>
                  <Badge
                    variant={foundCount > 0 ? 'default' : 'secondary'}
                    className="text-sm"
                  >
                    {formatNumber(foundCount)}
                  </Badge>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground">
                <div>
                  <span>{locale === 'ja' ? '経過: ' : 'Elapsed: '}</span>
                  <span className="font-mono">{formatTime(elapsedMs)}</span>
                </div>
                <div>
                  <span>{locale === 'ja' ? '残り: ' : 'Remaining: '}</span>
                  <span className="font-mono">
                    {estimatedRemainingMs > 0
                      ? formatTime(estimatedRemainingMs)
                      : '—'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Status Messages */}
          {status === 'completed' && (
            <div className="text-center py-2 text-green-600 dark:text-green-400 text-sm">
              {locale === 'ja' ? '検索完了' : 'Search completed'}
            </div>
          )}

          {status === 'error' && (
            <div className="text-center py-2 text-destructive text-sm">
              {locale === 'ja' ? 'エラーが発生しました' : 'An error occurred'}
            </div>
          )}
        </div>

      <KeyInputDialog
        isOpen={isKeyDialogOpen}
        onOpenChange={setIsKeyDialogOpen}
        availableKeys={tempAvailableKeys}
        onToggleKey={handleToggleKey}
        onReset={handleResetKeys}
        onApply={handleApplyKeys}
        labels={{
          dialogTitle: locale === 'ja' ? 'キー入力設定' : 'Key Input Settings',
          reset: locale === 'ja' ? 'リセット' : 'Reset',
          apply: locale === 'ja' ? '適用' : 'Apply',
        }}
      />
    </PanelCard>
  );
}
