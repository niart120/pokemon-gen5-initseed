import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { KeyInputDialog } from '@/components/keys';
import { useAppStore } from '@/store/app-store';
import { Sliders, GameController } from '@phosphor-icons/react';
import { KEY_INPUT_DEFAULT, keyMaskToNames, toggleKeyInMask, type KeyName } from '@/lib/utils/key-input';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatSearchParamsCurrentRange,
  searchParamsApplyButtonLabel,
  searchParamsConfigureButtonLabel,
  searchParamsDialogTitle,
  searchParamsEndDateLabel,
  searchParamsHourLabel,
  searchParamsKeyInputLabel,
  searchParamsMinuteLabel,
  searchParamsPanelTitle,
  searchParamsResetButtonLabel,
  searchParamsSecondLabel,
  searchParamsStartDateLabel,
  searchParamsTimeRangeLabel,
} from '@/lib/i18n/strings/search-params';

export function SearchParamsCard() {
  const { searchConditions, setSearchConditions } = useAppStore();
  const [isDialogOpen, setIsDialogOpen] = React.useState(false);
  const [tempKeyInput, setTempKeyInput] = React.useState(KEY_INPUT_DEFAULT);
  const locale = useLocale();

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
    searchConditions.dateRange.startYear,
    searchConditions.dateRange.startMonth,
    searchConditions.dateRange.startDay,
  );

  const endDate = formatDate(
    searchConditions.dateRange.endYear,
    searchConditions.dateRange.endMonth,
    searchConditions.dateRange.endDay,
  );

  const handleStartDateChange = (dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    setSearchConditions({
      dateRange: {
        ...searchConditions.dateRange,
        startYear: year,
        startMonth: month,
        startDay: day,
        startHour: 0,
        startMinute: 0,
        startSecond: 0,
      },
    });
  };

  const handleEndDateChange = (dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    setSearchConditions({
      dateRange: {
        ...searchConditions.dateRange,
        endYear: year,
        endMonth: month,
        endDay: day,
        endHour: 23,
        endMinute: 59,
        endSecond: 59,
      },
    });
  };

  const availableKeys = React.useMemo(() => keyMaskToNames(searchConditions.keyInput), [searchConditions.keyInput]);
  const tempAvailableKeys = React.useMemo(() => keyMaskToNames(tempKeyInput), [tempKeyInput]);

  const handleToggleKey = (key: KeyName) => {
    setTempKeyInput(toggleKeyInMask(tempKeyInput, key));
  };

  const handleResetKeys = () => {
    setTempKeyInput(KEY_INPUT_DEFAULT);
  };

  const handleApplyKeys = () => {
    setSearchConditions({ keyInput: tempKeyInput });
    setIsDialogOpen(false);
  };

  const openKeyDialog = () => {
    setTempKeyInput(searchConditions.keyInput);
    setIsDialogOpen(true);
  };

  const keyJoiner = locale === 'ja' ? '、' : ', ';

  const timeFieldConfigs = [
    {
      key: 'hour' as const,
      label: resolveLocaleValue(searchParamsHourLabel, locale),
      min: 0,
      max: 23,
    },
    {
      key: 'minute' as const,
      label: resolveLocaleValue(searchParamsMinuteLabel, locale),
      min: 0,
      max: 59,
    },
    {
      key: 'second' as const,
      label: resolveLocaleValue(searchParamsSecondLabel, locale),
      min: 0,
      max: 59,
    },
  ];

  const handleTimeRangeChange = (
    field: (typeof timeFieldConfigs)[number]['key'],
    edge: 'start' | 'end',
    rawValue: string,
  ) => {
    // 入力中はバリデーションせず、そのまま保存
    // フォーカスアウト時にバリデーションを行う
    const currentRange = searchConditions.timeRange[field];
    const nextRange = {
      ...currentRange,
      [edge]: rawValue,
    };

    setSearchConditions({
      timeRange: {
        ...searchConditions.timeRange,
        [field]: nextRange,
      },
    });
  };

  const handleTimeRangeBlur = (
    field: (typeof timeFieldConfigs)[number]['key'],
    edge: 'start' | 'end',
  ) => {
    const range = searchConditions.timeRange[field];
    const { min, max } = timeFieldConfigs.find((config) => config.key === field)!;

    // 空の場合やNaNの場合はminに補正
    const startValue = typeof range.start === 'string' ? parseInt(range.start, 10) : range.start;
    const endValue = typeof range.end === 'string' ? parseInt(range.end, 10) : range.end;
    
    const clampedStart = Number.isNaN(startValue) ? min : Math.min(Math.max(startValue, min), max);
    const clampedEnd = Number.isNaN(endValue) ? min : Math.min(Math.max(endValue, min), max);

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

    setSearchConditions({
      timeRange: {
        ...searchConditions.timeRange,
        [field]: { start: finalStart, end: finalEnd },
      },
    });
  };

  const timeInputClassName = 'h-8 w-11 px-0 text-center text-sm';

  return (
    <>
      <PanelCard
        icon={<Sliders size={20} className="opacity-80" />}
        title={resolveLocaleValue(searchParamsPanelTitle, locale)}
      >
        <div className="space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start-date" className="text-sm font-medium">
                {resolveLocaleValue(searchParamsStartDateLabel, locale)}
              </Label>
              <Input
                id="start-date"
                type="date"
                min="2000-01-01"
                max="2099-12-31"
                className="h-9 w-1/2"
                value={startDate}
                onChange={(event) => handleStartDateChange(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="end-date" className="text-sm font-medium">
                {resolveLocaleValue(searchParamsEndDateLabel, locale)}
              </Label>
              <Input
                id="end-date"
                type="date"
                min="2000-01-01"
                max="2099-12-31"
                className="h-9 w-1/2"
                value={endDate}
                onChange={(event) => handleEndDateChange(event.target.value)}
              />
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            {formatSearchParamsCurrentRange(startDate, endDate, locale)}
          </div>
          <div className="space-y-2">
            <Label className="text-sm font-medium">
              {resolveLocaleValue(searchParamsTimeRangeLabel, locale)}
            </Label>
            <div className="flex items-center gap-0 overflow-x-auto">
              {timeFieldConfigs.map((config) => {
                const range = searchConditions.timeRange[config.key];
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
                      aria-label={`${config.label} ${locale === 'ja' ? '最小' : 'min'}`}
                      className={timeInputClassName}
                      onChange={(event) => handleTimeRangeChange(config.key, 'start', event.target.value)}
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
                      onChange={(event) => handleTimeRangeChange(config.key, 'end', event.target.value)}
                      onBlur={() => handleTimeRangeBlur(config.key, 'end')}
                    />
                  </div>
                );
              })}
            </div>
          </div>
          <Separator />
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">
                {resolveLocaleValue(searchParamsKeyInputLabel, locale)}
              </div>
              <Button variant="outline" size="sm" onClick={openKeyDialog} className="gap-2">
                <GameController size={16} />
                {resolveLocaleValue(searchParamsConfigureButtonLabel, locale)}
              </Button>
            </div>
            {availableKeys.length > 0 && (
              <div className="text-xs text-muted-foreground">
                {availableKeys.join(keyJoiner)}
              </div>
            )}
          </div>
        </div>
      </PanelCard>
      <KeyInputDialog
        isOpen={isDialogOpen}
        onOpenChange={setIsDialogOpen}
        availableKeys={tempAvailableKeys}
        onToggleKey={handleToggleKey}
        onReset={handleResetKeys}
        onApply={handleApplyKeys}
        labels={{
          dialogTitle: resolveLocaleValue(searchParamsDialogTitle, locale),
          reset: resolveLocaleValue(searchParamsResetButtonLabel, locale),
          apply: resolveLocaleValue(searchParamsApplyButtonLabel, locale),
        }}
      />
    </>
  );
}
