import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { KeyInputDialog } from '@/components/keys';
import { GameController } from '@phosphor-icons/react';
import { DATE_INPUT_MIN, DATE_INPUT_MAX } from '@/components/ui/date-input-constraints';
import { KEY_INPUT_DEFAULT, keyMaskToNames, toggleKeyInMask, type KeyName } from '@/lib/utils/key-input';
import type { SupportedLocale } from '@/types/i18n';

export type TimeFieldKey = 'hour' | 'minute' | 'second';
export type TimeFieldValue = number | string;
export type TimeFieldRange = { start: TimeFieldValue; end: TimeFieldValue };
export type TimeRangeValue = Record<TimeFieldKey, TimeFieldRange>;

export type DateRangeValue = {
  startYear: number;
  startMonth: number;
  startDay: number;
  endYear: number;
  endMonth: number;
  endDay: number;
};

export type RangeKeySectionLabels = {
  startDate: string;
  endDate: string;
  timeRange: string;
  hour: string;
  minute: string;
  second: string;
  keyInput: string;
  configure: string;
  dialogTitle: string;
  reset: string;
  apply: string;
};

export type DateLimits = {
  min?: string;
  max?: string;
};

export type TimeLimits = Partial<Record<TimeFieldKey, { min: number; max: number }>>;

export interface RangeKeySectionProps {
  locale: SupportedLocale;
  dateRange: DateRangeValue;
  timeRange: TimeRangeValue;
  keyMask: number;
  labels: RangeKeySectionLabels;
  dateLimits?: DateLimits;
  timeLimits?: TimeLimits;
  keyJoiner?: string;
  isDisabled?: boolean;
  onDateChange: (edge: 'start' | 'end', value: { year: number; month: number; day: number }) => void;
  onResetTimeOnDateChange?: (edge: 'start' | 'end') => void;
  onTimeChange: (field: TimeFieldKey, edge: 'start' | 'end', rawValue: string) => void;
  onTimeCommit: (field: TimeFieldKey, range: { start: number; end: number }) => void;
  onKeyMaskChange: (mask: number) => void;
}

const timeInputClassName = 'h-8 w-11 px-0 text-center text-sm';

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

export function RangeKeySection({
  locale,
  dateRange,
  timeRange,
  keyMask,
  labels,
  dateLimits,
  timeLimits,
  keyJoiner,
  isDisabled = false,
  onDateChange,
  onResetTimeOnDateChange,
  onTimeChange,
  onTimeCommit,
  onKeyMaskChange,
}: RangeKeySectionProps) {
  const [isDialogOpen, setIsDialogOpen] = React.useState(false);
  const [tempKeyMask, setTempKeyMask] = React.useState(keyMask);

  React.useEffect(() => {
    setTempKeyMask(keyMask);
  }, [keyMask]);

  const availableKeys = React.useMemo(() => keyMaskToNames(keyMask), [keyMask]);
  const tempAvailableKeys = React.useMemo(() => keyMaskToNames(tempKeyMask), [tempKeyMask]);
  const joiner = keyJoiner ?? (locale === 'ja' ? '、' : ', ');

  const startDate = formatDate(dateRange.startYear, dateRange.startMonth, dateRange.startDay);
  const endDate = formatDate(dateRange.endYear, dateRange.endMonth, dateRange.endDay);

  const handleDateChange = (edge: 'start' | 'end', dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    onDateChange(edge, { year, month, day });
    onResetTimeOnDateChange?.(edge);
  };

  const resolvedDateMin = dateLimits?.min ?? DATE_INPUT_MIN;
  const resolvedDateMax = dateLimits?.max ?? DATE_INPUT_MAX;

  const timeFieldConfigs: Array<{ key: TimeFieldKey; label: string; min: number; max: number }> = [
    {
      key: 'hour',
      label: labels.hour,
      min: timeLimits?.hour?.min ?? 0,
      max: timeLimits?.hour?.max ?? 23,
    },
    {
      key: 'minute',
      label: labels.minute,
      min: timeLimits?.minute?.min ?? 0,
      max: timeLimits?.minute?.max ?? 59,
    },
    {
      key: 'second',
      label: labels.second,
      min: timeLimits?.second?.min ?? 0,
      max: timeLimits?.second?.max ?? 59,
    },
  ];

  const handleTimeBlur = (field: TimeFieldKey, edge: 'start' | 'end') => {
    const { min, max } = timeFieldConfigs.find((config) => config.key === field)!;
    const range = timeRange[field];

    const rawStart = range.start;
    const rawEnd = range.end;
    const startValue = typeof rawStart === 'string' ? parseInt(rawStart, 10) : rawStart;
    const endValue = typeof rawEnd === 'string' ? parseInt(rawEnd, 10) : rawEnd;

    const clampedStart = Number.isNaN(startValue) ? min : Math.min(Math.max(startValue, min), max);
    const clampedEnd = Number.isNaN(endValue) ? min : Math.min(Math.max(endValue, min), max);

    let finalStart = clampedStart;
    let finalEnd = clampedEnd;
    if (finalStart > finalEnd) {
      if (edge === 'start') {
        finalEnd = finalStart;
      } else {
        finalStart = finalEnd;
      }
    }

    onTimeCommit(field, { start: finalStart, end: finalEnd });
  };

  const handleToggleKey = (key: KeyName) => {
    setTempKeyMask(toggleKeyInMask(tempKeyMask, key));
  };

  const handleResetKeys = () => {
    setTempKeyMask(KEY_INPUT_DEFAULT);
  };

  const handleApplyKeys = () => {
    onKeyMaskChange(tempKeyMask);
    setIsDialogOpen(false);
  };

  const openKeyDialog = () => {
    setTempKeyMask(keyMask);
    setIsDialogOpen(true);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="range-start-date" className="text-sm font-medium">
            {labels.startDate}
          </Label>
          <Input
            id="range-start-date"
            type="date"
            min={resolvedDateMin}
            max={resolvedDateMax}
            className="h-9"
            value={startDate}
            onChange={(event) => handleDateChange('start', event.target.value)}
            disabled={isDisabled}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="range-end-date" className="text-sm font-medium">
            {labels.endDate}
          </Label>
          <Input
            id="range-end-date"
            type="date"
            min={resolvedDateMin}
            max={resolvedDateMax}
            className="h-9"
            value={endDate}
            onChange={(event) => handleDateChange('end', event.target.value)}
            disabled={isDisabled}
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label className="text-sm font-medium">{labels.timeRange}</Label>
        <div className="flex items-center gap-0 overflow-x-auto">
          {timeFieldConfigs.map((config) => {
            const range = timeRange[config.key];
            return (
              <div key={config.key} className="flex items-center gap-0 whitespace-nowrap">
                <span className="text-xs text-muted-foreground w-8 text-right">{config.label}</span>
                <Input
                  type="number"
                  inputMode="numeric"
                  min={config.min}
                  max={config.max}
                  value={range.start}
                  aria-label={`${config.label} ${locale === 'ja' ? '最小' : 'min'}`}
                  className={timeInputClassName}
                  onChange={(event) => onTimeChange(config.key, 'start', event.target.value)}
                  onBlur={() => handleTimeBlur(config.key, 'start')}
                  disabled={isDisabled}
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
                  onChange={(event) => onTimeChange(config.key, 'end', event.target.value)}
                  onBlur={() => handleTimeBlur(config.key, 'end')}
                  disabled={isDisabled}
                />
              </div>
            );
          })}
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground">{labels.keyInput}</div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="flex-1 min-h-[2.25rem] rounded-md border bg-muted/40 px-3 py-2 text-xs font-mono">
            {availableKeys.length > 0 ? availableKeys.join(joiner) : '—'}
          </div>
          <Button variant="outline" size="sm" onClick={openKeyDialog} className="gap-2" disabled={isDisabled}>
            <GameController size={16} />
            {labels.configure}
          </Button>
        </div>
      </div>

      <KeyInputDialog
        isOpen={isDialogOpen}
        onOpenChange={setIsDialogOpen}
        availableKeys={tempAvailableKeys}
        onToggleKey={handleToggleKey}
        onReset={handleResetKeys}
        onApply={handleApplyKeys}
        labels={{
          dialogTitle: labels.dialogTitle,
          reset: labels.reset,
          apply: labels.apply,
        }}
      />
    </div>
  );
}
