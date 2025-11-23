import { ChevronDown, ChevronUp, Eye } from 'lucide-react';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { PanelCard } from '@/components/ui/panel-card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../../ui/table';
import { LazyTooltip } from '@/components/ui/lazy-tooltip';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { useTableVirtualization } from '@/hooks/use-table-virtualization';
import { lcgSeedToHex, lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatResultCount,
  formatSearchDuration,
  searchResultsFilteredEmptyMessage,
  searchResultsHeaders,
  searchResultsInitialMessage,
  searchResultsTitle,
  viewDetailsAriaLabel,
  viewDetailsLabel,
} from '@/lib/i18n/strings/search-results';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import type { InitialSeedResult } from '../../../types/search';
import type { SortField } from './ResultsControlCard';

interface ResultsCardProps {
  filteredAndSortedResults: InitialSeedResult[];
  searchResultsLength: number;
  sortField: SortField;
  sortOrder: 'asc' | 'desc';
  onSort: (field: SortField) => void;
  onShowDetails: (result: InitialSeedResult) => void;
}

const SEARCH_RESULTS_COLUMN_COUNT = 6;
const SEARCH_RESULTS_ROW_HEIGHT = 36;

export function ResultsCard({
  filteredAndSortedResults,
  searchResultsLength,
  sortField,
  sortOrder,
  onSort,
  onShowDetails,
}: ResultsCardProps) {
  const { lastSearchDuration } = useAppStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
  const virtualization = useTableVirtualization({
    rowCount: filteredAndSortedResults.length,
    defaultRowHeight: SEARCH_RESULTS_ROW_HEIGHT,
    overscan: 8,
  });
  const virtualRows = virtualization.virtualRows;

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) return null;
    return sortOrder === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />;
  };

  const handleSort = (field: SortField) => {
    onSort(field);
  };

  const filteredResultsCount = filteredAndSortedResults.length;

  return (
    <PanelCard
      icon={<Eye size={20} className="flex-shrink-0 opacity-80" />}
      title={resolveLocaleValue(searchResultsTitle, locale)}
      headerActions={
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="secondary" className="flex-shrink-0">
            {formatResultCount(filteredResultsCount, locale)}
          </Badge>
          {lastSearchDuration !== null && (
            <Badge variant="outline" className="flex-shrink-0 text-xs">
              {formatSearchDuration(lastSearchDuration, locale)}
            </Badge>
          )}
        </div>
      }
      className={isStack ? 'max-h-96' : 'min-h-96'}
      fullHeight={!isStack}
      padding="none"
      spacing="none"
    >
        <div
          ref={virtualization.containerRef}
          className="flex-1 min-h-0 overflow-y-auto"
        >
          {filteredAndSortedResults.length === 0 ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-muted-foreground">
              {resolveLocaleValue(
                searchResultsLength === 0 ? searchResultsInitialMessage : searchResultsFilteredEmptyMessage,
                locale,
              )}
            </div>
          ) : (
            <Table className="table-auto min-w-full text-xs leading-tight">
              <TableHeader>
                <TableRow className="h-9">
                  <TableHead className="w-12 px-1 text-center">
                    {resolveLocaleValue(searchResultsHeaders.action, locale)}
                  </TableHead>
                  <TableHead className="px-2 select-none">
                    {resolveLocaleValue(searchResultsHeaders.lcgSeed, locale)}
                  </TableHead>
                  <TableHead className="px-2 cursor-pointer select-none" onClick={() => handleSort('datetime')}>
                    <div className="flex items-center gap-1">
                      {resolveLocaleValue(searchResultsHeaders.dateTime, locale)} {getSortIcon('datetime')}
                    </div>
                  </TableHead>
                  <TableHead className="px-2 cursor-pointer select-none" onClick={() => handleSort('seed')}>
                    <div className="flex items-center gap-1">
                      {resolveLocaleValue(searchResultsHeaders.mtSeed, locale)} {getSortIcon('seed')}
                    </div>
                  </TableHead>
                  <TableHead className="px-2 cursor-pointer select-none" onClick={() => handleSort('timer0')}>
                    <div className="flex items-center gap-1">
                      {resolveLocaleValue(searchResultsHeaders.timer0, locale)} {getSortIcon('timer0')}
                    </div>
                  </TableHead>
                  <TableHead className="px-2 cursor-pointer select-none" onClick={() => handleSort('vcount')}>
                    <div className="flex items-center gap-1">
                      {resolveLocaleValue(searchResultsHeaders.vcount, locale)} {getSortIcon('vcount')}
                    </div>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {virtualization.paddingTop > 0 ? (
                  <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                    <TableCell
                      colSpan={SEARCH_RESULTS_COLUMN_COUNT}
                      className="p-0 border-0"
                      style={{ height: virtualization.paddingTop }}
                    />
                  </TableRow>
                ) : null}
                {virtualRows.map(virtualRow => {
                  const result = filteredAndSortedResults[virtualRow.index];
                  if (!result) {
                    return null;
                  }
                  const entryKey = `${result.lcgSeed}-${result.seed}`;
                  return (
                    <TableRow
                      key={entryKey}
                      ref={virtualization.measureRow}
                      data-index={virtualRow.index}
                      className="h-9"
                    >
                      <TableCell className="px-1 py-1 text-center">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onShowDetails(result)}
                          className="h-7 w-7 p-0"
                          title={resolveLocaleValue(viewDetailsLabel, locale)}
                          aria-label={resolveLocaleValue(viewDetailsAriaLabel, locale)}
                        >
                          <Eye size={14} />
                        </Button>
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-nowrap min-w-[120px]">
                        <LazyTooltip
                          trigger={<span>{lcgSeedToHex(result.lcgSeed)}</span>}
                          renderContent={() => (
                            <>
                              {getIvTooltipEntries(lcgSeedToMtSeed(result.lcgSeed), locale).map(entry => (
                                <div key={entry.label} className="space-y-0.5">
                                  <div className="font-semibold leading-tight">{entry.label}</div>
                                  <div className="font-mono leading-tight">{entry.spread}</div>
                                  <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                                </div>
                              ))}
                            </>
                          )}
                          side="bottom"
                          className="space-y-1 text-left"
                        />
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-nowrap">
                        {formatBootTimestampDisplay(result.datetime, locale)}
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-nowrap">
                        <LazyTooltip
                          trigger={<span>0x{result.seed.toString(16).toUpperCase().padStart(8, '0')}</span>}
                          renderContent={() => (
                            <>
                              {getIvTooltipEntries(result.seed, locale).map(entry => (
                                <div key={entry.label} className="space-y-0.5">
                                  <div className="font-semibold leading-tight">{entry.label}</div>
                                  <div className="font-mono leading-tight">{entry.spread}</div>
                                  <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                                </div>
                              ))}
                            </>
                          )}
                          side="bottom"
                          className="space-y-1 text-left"
                        />
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-nowrap">
                        {formatTimer0Hex(result.timer0)}
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-nowrap">
                        {formatVCountHex(result.vcount)}
                      </TableCell>
                    </TableRow>
                  );
                })}
                {virtualization.paddingBottom > 0 ? (
                  <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                    <TableCell
                      colSpan={SEARCH_RESULTS_COLUMN_COUNT}
                      className="p-0 border-0"
                      style={{ height: virtualization.paddingBottom }}
                    />
                  </TableRow>
                ) : null}
              </TableBody>
            </Table>
          )}
        </div>
    </PanelCard>
  );
}
