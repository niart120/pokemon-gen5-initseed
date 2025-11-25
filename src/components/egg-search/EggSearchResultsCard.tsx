/**
 * EggSearchResultsCard
 * 検索結果表示カード
 */

import React from 'react';
import { Table as TableIcon } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { natureName } from '@/lib/utils/format-display';
import {
  eggSearchResultsCardTitle,
  eggSearchResultsEmpty,
  eggSearchResultsTableHeaders,
  formatEggSearchElapsed,
  formatEggSearchResultsCount,
} from '@/lib/i18n/strings/egg-search';

export function EggSearchResultsCard() {
  const locale = useLocale();
  const { isStack } = useResponsiveLayout();
  const { getFilteredResults, lastElapsedMs } = useEggBootTimingSearchStore();

  const filteredResults = getFilteredResults();

  const formatDatetime = (date: Date): string => {
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  };

  const formatTimer0 = (value: number): string => {
    return `0x${value.toString(16).toUpperCase().padStart(4, '0')}`;
  };

  const formatVCount = (value: number): string => {
    return `0x${value.toString(16).toUpperCase().padStart(2, '0')}`;
  };

  const formatIVs = (ivs: readonly [number, number, number, number, number, number]): string => {
    return ivs.join('-');
  };

  const getShinyDisplay = (shiny: number): { text: string; className: string } => {
    switch (shiny) {
      case 2:
        return { text: '★', className: 'text-yellow-500' };
      case 1:
        return { text: '◇', className: 'text-blue-500' };
      default:
        return { text: '-', className: 'text-muted-foreground' };
    }
  };

  return (
    <PanelCard
      icon={<TableIcon size={20} className="opacity-80" />}
      title={eggSearchResultsCardTitle[locale]}
      headerActions={
        <div className="flex items-center gap-2">
          <Badge variant="secondary">
            {formatEggSearchResultsCount(filteredResults.length, locale)}
          </Badge>
          {lastElapsedMs !== null && (
            <Badge variant="outline" className="text-xs">
              {formatEggSearchElapsed(lastElapsedMs, locale)}
            </Badge>
          )}
        </div>
      }
      className={isStack ? 'max-h-96' : 'min-h-96'}
      fullHeight={!isStack}
      padding="none"
      spacing="none"
    >
      <div className="flex-1 min-h-0 overflow-auto">
        {filteredResults.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 text-center text-muted-foreground py-8">
            {eggSearchResultsEmpty[locale]}
          </div>
        ) : (
          <Table className="table-auto min-w-full text-xs leading-tight">
            <TableHeader>
              <TableRow className="h-9">
                <TableHead className="px-2">{eggSearchResultsTableHeaders.bootTime[locale]}</TableHead>
                <TableHead className="px-2">{eggSearchResultsTableHeaders.timer0[locale]}</TableHead>
                <TableHead className="px-2">{eggSearchResultsTableHeaders.vcount[locale]}</TableHead>
                <TableHead className="px-2">{eggSearchResultsTableHeaders.lcgSeed[locale]}</TableHead>
                <TableHead className="px-2">{eggSearchResultsTableHeaders.advance[locale]}</TableHead>
                <TableHead className="px-2">{eggSearchResultsTableHeaders.nature[locale]}</TableHead>
                <TableHead className="px-2">{eggSearchResultsTableHeaders.ivs[locale]}</TableHead>
                <TableHead className="px-2 text-center">{eggSearchResultsTableHeaders.shiny[locale]}</TableHead>
                <TableHead className="px-2 text-center">{eggSearchResultsTableHeaders.stable[locale]}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredResults.map((result, index) => {
                const shinyDisplay = getShinyDisplay(result.egg.egg.shiny);
                return (
                  <TableRow key={`${result.lcgSeedHex}-${index}`} className="h-8">
                    <TableCell className="px-2 py-1 font-mono text-[11px] whitespace-nowrap">
                      {formatDatetime(result.boot.datetime)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px]">
                      {formatTimer0(result.boot.timer0)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px]">
                      {formatVCount(result.boot.vcount)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px]">
                      {result.lcgSeedHex}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px]">
                      {result.egg.advance}
                    </TableCell>
                    <TableCell className="px-2 py-1 text-[11px]">
                      {natureName(result.egg.egg.nature, locale)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px]">
                      {formatIVs(result.egg.egg.ivs)}
                    </TableCell>
                    <TableCell className={`px-2 py-1 text-center ${shinyDisplay.className}`}>
                      {shinyDisplay.text}
                    </TableCell>
                    <TableCell className="px-2 py-1 text-center">
                      {result.isStable ? '○' : '×'}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </div>
    </PanelCard>
  );
}
