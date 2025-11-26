import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Separator } from '@/components/ui/separator';
import { Gear } from '@phosphor-icons/react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Toggle } from '@/components/ui/toggle';
import { useAppStore } from '@/store/app-store';
import { GameController } from '@phosphor-icons/react';
import { KEY_INPUT_DEFAULT, keyMaskToNames, keyNamesToMask, type KeyName } from '@/lib/utils/key-input';
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
    const current = keyMaskToNames(tempKeyInput);
    const next = current.includes(key)
      ? current.filter((item) => item !== key)
      : [...current, key];
    setTempKeyInput(keyNamesToMask(next));
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
        icon={<Gear size={20} className="opacity-80" />}
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
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{resolveLocaleValue(searchParamsDialogTitle, locale)}</DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <div className="flex justify-between px-8">
              <Toggle
                value="L"
                aria-label="L"
                pressed={tempAvailableKeys.includes('L')}
                onPressedChange={() => handleToggleKey('L')}
                className="px-6 py-2"
              >
                L
              </Toggle>
              <Toggle
                value="R"
                aria-label="R"
                pressed={tempAvailableKeys.includes('R')}
                onPressedChange={() => handleToggleKey('R')}
                className="px-6 py-2"
              >
                R
              </Toggle>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="grid grid-cols-3 gap-1 font-arrows">
                  <div />
                  <Toggle
                    value="[↑]"
                    aria-label="Up"
                    pressed={tempAvailableKeys.includes('[↑]')}
                    onPressedChange={() => handleToggleKey('[↑]')}
                    className="w-12 h-12"
                  >
                    [↑]
                  </Toggle>
                  <div />
                  <Toggle
                    value="[←]"
                    aria-label="Left"
                    pressed={tempAvailableKeys.includes('[←]')}
                    onPressedChange={() => handleToggleKey('[←]')}
                    className="w-12 h-12"
                  >
                    [←]
                  </Toggle>
                  <div className="w-12 h-12" />
                  <Toggle
                    value="[→]"
                    aria-label="Right"
                    pressed={tempAvailableKeys.includes('[→]')}
                    onPressedChange={() => handleToggleKey('[→]')}
                    className="w-12 h-12"
                  >
                    [→]
                  </Toggle>
                  <div />
                  <Toggle
                    value="[↓]"
                    aria-label="Down"
                    pressed={tempAvailableKeys.includes('[↓]')}
                    onPressedChange={() => handleToggleKey('[↓]')}
                    className="w-12 h-12"
                  >
                    [↓]
                  </Toggle>
                  <div />
                </div>
              </div>
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="flex gap-2">
                  <Toggle
                    value="Select"
                    aria-label="Select"
                    pressed={tempAvailableKeys.includes('Select')}
                    onPressedChange={() => handleToggleKey('Select')}
                    className="px-3 py-2"
                  >
                    Select
                  </Toggle>
                  <Toggle
                    value="Start"
                    aria-label="Start"
                    pressed={tempAvailableKeys.includes('Start')}
                    onPressedChange={() => handleToggleKey('Start')}
                    className="px-3 py-2"
                  >
                    Start
                  </Toggle>
                </div>
              </div>
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="grid grid-cols-3 gap-1">
                  <div />
                  <Toggle
                    value="X"
                    aria-label="X"
                    pressed={tempAvailableKeys.includes('X')}
                    onPressedChange={() => handleToggleKey('X')}
                    className="w-12 h-12"
                  >
                    X
                  </Toggle>
                  <div />
                  <Toggle
                    value="Y"
                    aria-label="Y"
                    pressed={tempAvailableKeys.includes('Y')}
                    onPressedChange={() => handleToggleKey('Y')}
                    className="w-12 h-12"
                  >
                    Y
                  </Toggle>
                  <div className="w-12 h-12" />
                  <Toggle
                    value="A"
                    aria-label="A"
                    pressed={tempAvailableKeys.includes('A')}
                    onPressedChange={() => handleToggleKey('A')}
                    className="w-12 h-12"
                  >
                    A
                  </Toggle>
                  <div />
                  <Toggle
                    value="B"
                    aria-label="B"
                    pressed={tempAvailableKeys.includes('B')}
                    onPressedChange={() => handleToggleKey('B')}
                    className="w-12 h-12"
                  >
                    B
                  </Toggle>
                  <div />
                </div>
              </div>
            </div>
            <div className="flex justify-between items-center pt-4 border-t">
              <Button variant="outline" size="sm" onClick={handleResetKeys}>
                {resolveLocaleValue(searchParamsResetButtonLabel, locale)}
              </Button>
              <Button size="sm" onClick={handleApplyKeys}>
                {resolveLocaleValue(searchParamsApplyButtonLabel, locale)}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
