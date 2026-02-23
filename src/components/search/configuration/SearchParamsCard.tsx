import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { useAppStore } from '@/store/app-store';
import { Sliders } from '@phosphor-icons/react';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  rangeKeyDialogLabels,
  rangeKeySectionLabels,
} from '@/lib/i18n/strings/range-key-section';
import {
  formatSearchParamsCurrentRange,
  searchParamsPanelTitle,
} from '@/lib/i18n/strings/search-params';
import { RangeKeySection } from '@/components/shared/RangeKeySection';
import type { TimeFieldKey } from '@/components/shared/RangeKeySection';

export function SearchParamsCard() {
  const { searchConditions, setSearchConditions, searchProgress } = useAppStore();
  const locale = useLocale();
  const formatDate = (year: number, month: number, day: number): string => {
    return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
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

  const handleDateChange = (edge: 'start' | 'end', value: { year: number; month: number; day: number }) => {
    const nextDateRange = { ...searchConditions.dateRange };
    if (edge === 'start') {
      nextDateRange.startYear = value.year;
      nextDateRange.startMonth = value.month;
      nextDateRange.startDay = value.day;
    } else {
      nextDateRange.endYear = value.year;
      nextDateRange.endMonth = value.month;
      nextDateRange.endDay = value.day;
    }
    setSearchConditions({ dateRange: nextDateRange });
  };

  const handleTimeRangeChange = (field: TimeFieldKey, edge: 'start' | 'end', rawValue: string) => {
    const currentRange = searchConditions.timeRange[field];
    setSearchConditions({
      timeRange: {
        ...searchConditions.timeRange,
        [field]: {
          ...currentRange,
          [edge]: rawValue,
        },
      },
    });
  };

  const handleTimeRangeCommit = (field: TimeFieldKey, range: { start: number; end: number }) => {
    setSearchConditions({
      timeRange: {
        ...searchConditions.timeRange,
        [field]: range,
      },
    });
  };

  const handleKeyMaskChange = (mask: number) => {
    setSearchConditions({ keyInput: mask });
  };

  const isRangeDisabled = searchProgress.isRunning || searchProgress.isPaused;

  const labels = {
    startDate: resolveLocaleValue(rangeKeySectionLabels.startDate, locale),
    endDate: resolveLocaleValue(rangeKeySectionLabels.endDate, locale),
    timeRange: resolveLocaleValue(rangeKeySectionLabels.timeRange, locale),
    hour: resolveLocaleValue(rangeKeySectionLabels.hour, locale),
    minute: resolveLocaleValue(rangeKeySectionLabels.minute, locale),
    second: resolveLocaleValue(rangeKeySectionLabels.second, locale),
    keyInput: resolveLocaleValue(rangeKeySectionLabels.keyInput, locale),
    configure: resolveLocaleValue(rangeKeySectionLabels.configure, locale),
    dialogTitle: resolveLocaleValue(rangeKeyDialogLabels.title, locale),
    reset: resolveLocaleValue(rangeKeyDialogLabels.reset, locale),
    apply: resolveLocaleValue(rangeKeyDialogLabels.apply, locale),
  };

  return (
    <>
      <PanelCard
        icon={<Sliders size={20} className="opacity-80" />}
        title={resolveLocaleValue(searchParamsPanelTitle, locale)}
      >
        <div className="space-y-3">
          <RangeKeySection
            locale={locale}
            dateRange={searchConditions.dateRange}
            timeRange={searchConditions.timeRange}
            keyMask={searchConditions.keyInput}
            labels={labels}
            isDisabled={isRangeDisabled}
            onDateChange={handleDateChange}
            onTimeChange={handleTimeRangeChange}
            onTimeCommit={handleTimeRangeCommit}
            onKeyMaskChange={handleKeyMaskChange}
          />
          <div className="text-xs text-muted-foreground">
            {formatSearchParamsCurrentRange(startDate, endDate, locale)}
          </div>
        </div>
      </PanelCard>
    </>
  );
}
