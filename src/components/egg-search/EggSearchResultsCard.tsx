/**
 * EggSearchResultsCard
 * 検索結果表示カード
 */

import React from 'react';
import { Table as TableIcon, MagnifyingGlass } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { natureName } from '@/lib/utils/format-display';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';
import {
  eggSearchResultsCardTitle,
  eggSearchResultsEmpty,
  eggSearchResultsTableHeaders,
  formatEggSearchElapsed,
  formatEggSearchResultsCount,
  eggSearchAbilityOptions,
  eggSearchGenderOptions,
} from '@/lib/i18n/strings/egg-search';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import type { EggBootTimingSearchResult } from '@/types/egg-boot-timing-search';

export function EggSearchResultsCard() {
  const locale = useLocale();
  const { isStack } = useResponsiveLayout();
  const { getFilteredResults, lastElapsedMs } = useEggBootTimingSearchStore();
  const [selectedResult, setSelectedResult] = React.useState<EggBootTimingSearchResult | null>(null);

  const filteredResults = getFilteredResults();

  const abilityOptions = resolveLocaleValue(eggSearchAbilityOptions, locale);
  const genderOptions = resolveLocaleValue(eggSearchGenderOptions, locale);
  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;

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

  const getAbilityDisplay = (ability: 0 | 1 | 2): string => {
    return abilityOptions[String(ability) as '0' | '1' | '2'];
  };

  const getGenderDisplay = (gender: 'male' | 'female' | 'genderless'): string => {
    return genderOptions[gender];
  };

  const getHiddenPowerDisplay = (result: EggBootTimingSearchResult): string => {
    const hpInfo = result.egg.egg.hiddenPower;
    if (!hpInfo.known) return '-';
    return `${hpTypeNames[hpInfo.type]} ${hpInfo.power}`;
  };

  const getKeysDisplay = (keyInputNames: string[]): string => {
    if (keyInputNames.length === 0) return '-';
    return keyInputNames.join('+');
  };

  return (
    <>
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
            <Table className="table-auto min-w-max text-xs leading-tight">
              <TableHeader>
                <TableRow className="h-9">
                  <TableHead className="px-1 w-8"></TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.lcgSeed[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.bootTime[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.timer0[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.vcount[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.advance[locale]}</TableHead>
                  <TableHead className="px-1 text-center">{eggSearchResultsTableHeaders.stable[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.ability[locale]}</TableHead>
                  <TableHead className="px-1">{eggSearchResultsTableHeaders.gender[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.nature[locale]}</TableHead>
                  <TableHead className="px-1 text-center">{eggSearchResultsTableHeaders.shiny[locale]}</TableHead>
                  <TableHead className="px-1 text-center">H</TableHead>
                  <TableHead className="px-1 text-center">A</TableHead>
                  <TableHead className="px-1 text-center">B</TableHead>
                  <TableHead className="px-1 text-center">C</TableHead>
                  <TableHead className="px-1 text-center">D</TableHead>
                  <TableHead className="px-1 text-center">S</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.hiddenPower[locale]}</TableHead>
                  <TableHead className="px-2">{eggSearchResultsTableHeaders.keys[locale]}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredResults.map((result, index) => {
                  const shinyDisplay = getShinyDisplay(result.egg.egg.shiny);
                  const ivs = result.egg.egg.ivs;
                  return (
                    <TableRow key={`${result.lcgSeedHex}-${index}`} className="h-8">
                      <TableCell className="px-1 py-1 text-center">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => setSelectedResult(result)}
                        >
                          <MagnifyingGlass size={14} />
                        </Button>
                      </TableCell>
                      <TableCell className="px-2 py-1 font-mono text-[11px]">
                        {result.lcgSeedHex}
                      </TableCell>
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
                        {result.egg.advance}
                      </TableCell>
                      <TableCell className="px-1 py-1 text-center">
                        {result.isStable ? '○' : '×'}
                      </TableCell>
                      <TableCell className="px-2 py-1 text-[11px]">
                        {getAbilityDisplay(result.egg.egg.ability)}
                      </TableCell>
                      <TableCell className="px-1 py-1 text-[11px]">
                        {getGenderDisplay(result.egg.egg.gender)}
                      </TableCell>
                      <TableCell className="px-2 py-1 text-[11px]">
                        {natureName(result.egg.egg.nature, locale)}
                      </TableCell>
                      <TableCell className={`px-1 py-1 text-center ${shinyDisplay.className}`}>
                        {shinyDisplay.text}
                      </TableCell>
                      <TableCell className="px-1 py-1 font-mono text-[11px] text-center">{ivs[0]}</TableCell>
                      <TableCell className="px-1 py-1 font-mono text-[11px] text-center">{ivs[1]}</TableCell>
                      <TableCell className="px-1 py-1 font-mono text-[11px] text-center">{ivs[2]}</TableCell>
                      <TableCell className="px-1 py-1 font-mono text-[11px] text-center">{ivs[3]}</TableCell>
                      <TableCell className="px-1 py-1 font-mono text-[11px] text-center">{ivs[4]}</TableCell>
                      <TableCell className="px-1 py-1 font-mono text-[11px] text-center">{ivs[5]}</TableCell>
                      <TableCell className="px-2 py-1 text-[11px] whitespace-nowrap">
                        {getHiddenPowerDisplay(result)}
                      </TableCell>
                      <TableCell className="px-2 py-1 text-[11px] whitespace-nowrap">
                        {getKeysDisplay(result.boot.keyInputNames)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </div>
      </PanelCard>

      {/* 詳細ダイアログ */}
      <Dialog open={selectedResult !== null} onOpenChange={(open) => !open && setSelectedResult(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{eggSearchResultsTableHeaders.detail[locale]}</DialogTitle>
          </DialogHeader>
          {selectedResult && (
            <div className="space-y-4 py-2">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.bootTime[locale]}:</span>
                  <span className="ml-2 font-mono">{formatDatetime(selectedResult.boot.datetime)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.timer0[locale]}:</span>
                  <span className="ml-2 font-mono">{formatTimer0(selectedResult.boot.timer0)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.vcount[locale]}:</span>
                  <span className="ml-2 font-mono">{formatVCount(selectedResult.boot.vcount)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.lcgSeed[locale]}:</span>
                  <span className="ml-2 font-mono">{selectedResult.lcgSeedHex}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.advance[locale]}:</span>
                  <span className="ml-2 font-mono">{selectedResult.egg.advance}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.ability[locale]}:</span>
                  <span className="ml-2">{getAbilityDisplay(selectedResult.egg.egg.ability)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.gender[locale]}:</span>
                  <span className="ml-2">{getGenderDisplay(selectedResult.egg.egg.gender)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.nature[locale]}:</span>
                  <span className="ml-2">{natureName(selectedResult.egg.egg.nature, locale)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.shiny[locale]}:</span>
                  <span className={`ml-2 ${getShinyDisplay(selectedResult.egg.egg.shiny).className}`}>
                    {getShinyDisplay(selectedResult.egg.egg.shiny).text}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.ivs[locale]}:</span>
                  <span className="ml-2 font-mono">{formatIVs(selectedResult.egg.egg.ivs)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.hiddenPower[locale]}:</span>
                  <span className="ml-2">{getHiddenPowerDisplay(selectedResult)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.keys[locale]}:</span>
                  <span className="ml-2">{getKeysDisplay(selectedResult.boot.keyInputNames)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">{eggSearchResultsTableHeaders.stable[locale]}:</span>
                  <span className="ml-2">{selectedResult.isStable ? '○' : '×'}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">PID:</span>
                  <span className="ml-2 font-mono">0x{selectedResult.egg.egg.pid.toString(16).toUpperCase().padStart(8, '0')}</span>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
