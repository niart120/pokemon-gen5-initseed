/**
 * ID調整検索結果テーブルコンポーネント
 * 仕様: 列順序 - 日時, LCG Seed, 表ID, 裏ID, 色違い, timer0, VCount, キー入力
 * IdAdjustmentCard に直接組み込むため PanelCard は使用しない
 */
import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { useTableVirtualization } from '@/hooks/use-table-virtualization';
import { useIdAdjustmentSearchStore } from '@/store/id-adjustment-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import { formatKeyInputForDisplay } from '@/lib/utils/key-input';
import { DomainShinyType } from '@/types/domain';
import {
  idAdjustmentResultsTableHeaders,
  idAdjustmentFilterLabels,
  idAdjustmentResultsEmpty,
  idAdjustmentResultsSearching,
  getIdAdjustmentShinyTypeLabel,
} from '@/lib/i18n/strings/id-adjustment-search';

const COLUMN_COUNT = 8;
const ROW_HEIGHT = 34;

function formatShinyType(shinyType: DomainShinyType, locale: 'ja' | 'en'): string {
  switch (shinyType) {
    case DomainShinyType.Square:
      return getIdAdjustmentShinyTypeLabel('square', locale);
    case DomainShinyType.Star:
      return getIdAdjustmentShinyTypeLabel('star', locale);
    default:
      return getIdAdjustmentShinyTypeLabel('normal', locale);
  }
}

/**
 * ID調整検索結果テーブル（フィルターとテーブル部分のみ）
 * IdAdjustmentCardに直接組み込んで使用する
 */
export function IdAdjustmentResultsTable() {
  const locale = useLocale();
  const {
    results,
    _pendingResults,
    status,
    resultFilters,
    getFilteredResults,
    updateResultFilters,
    progress,
  } = useIdAdjustmentSearchStore();

  const filteredResults = React.useMemo(
    () => getFilteredResults(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [results, _pendingResults, status, resultFilters, getFilteredResults]
  );

  const virtualization = useTableVirtualization({
    rowCount: filteredResults.length,
    defaultRowHeight: ROW_HEIGHT,
    overscan: 8,
  });

  const virtualRows = virtualization.virtualRows;

  return (
    <div className="flex flex-col flex-1 min-h-0 border-t">
      {/* Results label */}
      <div className="px-4 py-2 border-b">
        <Label className="text-xs">Results ({filteredResults.length})</Label>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 px-4 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <Label htmlFor="timer0-filter" className="text-xs text-muted-foreground">
            {idAdjustmentFilterLabels.timer0[locale]}:
          </Label>
          <Input
            id="timer0-filter"
            type="text"
            placeholder="0x..."
            className="h-7 w-20 text-xs font-mono"
            value={resultFilters.timer0Filter ?? ''}
            onChange={(e) => updateResultFilters({ timer0Filter: e.target.value })}
          />
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="vcount-filter" className="text-xs text-muted-foreground">
            {idAdjustmentFilterLabels.vcount[locale]}:
          </Label>
          <Input
            id="vcount-filter"
            type="text"
            placeholder="0x..."
            className="h-7 w-20 text-xs font-mono"
            value={resultFilters.vcountFilter ?? ''}
            onChange={(e) => updateResultFilters({ vcountFilter: e.target.value })}
          />
        </div>
        <div className="flex items-center gap-2">
          <Checkbox
            id="shiny-only-filter"
            checked={resultFilters.shinyOnly ?? false}
            onCheckedChange={(checked) =>
              updateResultFilters({ shinyOnly: checked === true })
            }
          />
          <Label htmlFor="shiny-only-filter" className="text-xs">
            {idAdjustmentFilterLabels.shinyOnly[locale]}
          </Label>
        </div>
      </div>

      {/* Table */}
      <div
        ref={virtualization.containerRef}
        className="flex-1 min-h-0 overflow-auto"
      >
        {filteredResults.length === 0 ? (
          <div className="flex h-32 items-center justify-center px-6 text-center text-muted-foreground">
            {progress && progress.processedCombinations > 0
              ? idAdjustmentResultsSearching[locale]
              : idAdjustmentResultsEmpty[locale]}
          </div>
        ) : (
          <Table className="min-w-max text-xs">
            <TableHeader className="sticky top-0 bg-muted text-xs">
              <TableRow className="text-left border-0">
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.dateTime[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.lcgSeed[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.tid[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.sid[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.shiny[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.timer0[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.vcount[locale]}
                </TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                  {idAdjustmentResultsTableHeaders.keyInput[locale]}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {virtualization.paddingTop > 0 ? (
                <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                  <TableCell
                    colSpan={COLUMN_COUNT}
                    className="p-0 border-0"
                    style={{ height: virtualization.paddingTop }}
                  />
                </TableRow>
              ) : null}
              {virtualRows.map((virtualRow) => {
                const result = filteredResults[virtualRow.index];
                if (!result) return null;

                const isShiny = (result.shinyType ?? DomainShinyType.Normal) !== DomainShinyType.Normal;

                return (
                  <TableRow
                    key={virtualRow.index}
                    ref={virtualization.measureRow}
                    data-index={virtualRow.index}
                    className={`border-0 ${
                      isShiny
                        ? 'bg-yellow-50 dark:bg-yellow-900/20'
                        : 'odd:bg-background even:bg-muted/30'
                    }`}
                  >
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {formatBootTimestampDisplay(result.boot.datetime, locale)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {`0x${result.lcgSeedHex.toUpperCase()}`}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {result.tid.toString().padStart(5, '0')}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {result.sid.toString().padStart(5, '0')}
                    </TableCell>
                    <TableCell
                      className={`px-2 py-1 whitespace-nowrap ${
                        isShiny ? 'text-yellow-600 dark:text-yellow-400 font-medium' : ''
                      }`}
                    >
                      {formatShinyType(result.shinyType ?? DomainShinyType.Normal, locale)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {formatTimer0Hex(result.boot.timer0)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {formatVCountHex(result.boot.vcount)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {formatKeyInputForDisplay(result.boot.keyCode, result.boot.keyInputNames)}
                    </TableCell>
                  </TableRow>
                );
              })}
              {virtualization.paddingBottom > 0 ? (
                <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                  <TableCell
                    colSpan={COLUMN_COUNT}
                    className="p-0 border-0"
                    style={{ height: virtualization.paddingBottom }}
                  />
                </TableRow>
              ) : null}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
