import { Eye } from 'lucide-react';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { PanelCard } from '@/components/ui/panel-card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../../ui/table';
import { LazyTooltip } from '@/components/ui/lazy-tooltip';
import { SearchExportButton } from './SearchExportButton';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { useTableVirtualization } from '@/hooks/use-table-virtualization';
import { lcgSeedToHex, lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatResultCount,
  formatProcessingDuration,
  searchResultsEmptyMessage,
  searchResultsHeaders,
  searchResultsTitle,
  viewDetailsAriaLabel,
  viewDetailsLabel,
} from '@/lib/i18n/strings/search-results';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import type { InitialSeedResult, SearchResult } from '../../../types/search';

interface ResultsCardProps {
  sortedResults: InitialSeedResult[];
  convertedResults: SearchResult[];
  onShowDetails: (result: InitialSeedResult) => void;
}

const SEARCH_RESULTS_COLUMN_COUNT = 6;
const SEARCH_RESULTS_ROW_HEIGHT = 34;

export function ResultsCard({
  sortedResults,
  convertedResults,
  onShowDetails,
}: ResultsCardProps) {
  const { lastSearchDuration } = useAppStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
  const virtualization = useTableVirtualization({
    rowCount: sortedResults.length,
    defaultRowHeight: SEARCH_RESULTS_ROW_HEIGHT,
    overscan: 8,
  });
  const virtualRows = virtualization.virtualRows;

  const resultsCount = sortedResults.length;

  return (
    <PanelCard
      icon={<Eye size={20} className="flex-shrink-0 opacity-80" />}
      title={resolveLocaleValue(searchResultsTitle, locale)}
      headerActions={
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="secondary" className="flex-shrink-0">
            {formatResultCount(resultsCount, locale)}
          </Badge>
          {lastSearchDuration !== null && (
            <Badge variant="outline" className="flex-shrink-0 text-xs">
              {formatProcessingDuration(lastSearchDuration)}
            </Badge>
          )}
          <SearchExportButton
            results={convertedResults}
            disabled={resultsCount === 0}
          />
        </div>
      }
      className={isStack ? 'max-h-96' : 'min-h-96'}
      fullHeight={!isStack}
      padding="none"
      spacing="none"
    >
        <div
          ref={virtualization.containerRef}
          className="flex-1 min-h-0 overflow-auto"
        >
          {sortedResults.length === 0 ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-muted-foreground">
              {resolveLocaleValue(searchResultsEmptyMessage, locale)}
            </div>
          ) : (
            <Table className="min-w-max text-xs">
              <TableHeader className="sticky top-0 bg-muted text-xs">
                <TableRow className="text-left border-0">
                  <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-center">
                    {resolveLocaleValue(searchResultsHeaders.action, locale)}
                  </TableHead>
                  <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                    {resolveLocaleValue(searchResultsHeaders.lcgSeed, locale)}
                  </TableHead>
                  <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                    {resolveLocaleValue(searchResultsHeaders.dateTime, locale)}
                  </TableHead>
                  <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                    {resolveLocaleValue(searchResultsHeaders.mtSeed, locale)}
                  </TableHead>
                  <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                    {resolveLocaleValue(searchResultsHeaders.timer0, locale)}
                  </TableHead>
                  <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                    {resolveLocaleValue(searchResultsHeaders.vcount, locale)}
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
                  const result = sortedResults[virtualRow.index];
                  if (!result) {
                    return null;
                  }
                  const entryKey = `${result.lcgSeed}-${result.seed}`;
                  return (
                    <TableRow
                      key={entryKey}
                      ref={virtualization.measureRow}
                      data-index={virtualRow.index}
                      className="odd:bg-background even:bg-muted/30 border-0"
                    >
                      <TableCell className="px-2 py-1 text-center">
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
                      <TableCell className="px-2 py-1 font-mono whitespace-nowrap min-w-[120px]">
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
                      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                        {formatBootTimestampDisplay(result.datetime, locale)}
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
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
                      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                        {formatTimer0Hex(result.timer0)}
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
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
