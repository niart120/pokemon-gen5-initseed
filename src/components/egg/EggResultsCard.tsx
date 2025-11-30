import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Table as TableIcon } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useTableVirtualization } from '@/hooks/use-table-virtualization';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import { IV_UNKNOWN, type EnumeratedEggDataWithBootTiming } from '@/types/egg';
import {
  eggResultsPanelTitle,
  eggResultsEmptyMessage,
  getEggResultHeader,
  eggResultShinyLabels,
  eggResultGenderLabels,
  eggResultAbilityLabels,
  eggResultStableLabels,
  eggResultUnknownHp,
} from '@/lib/i18n/strings/egg-results';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';
import { EggExportButton } from './EggExportButton';

const EGG_RESULTS_TABLE_ROW_HEIGHT = 34;

/**
 * EggResultsCard
 * タマゴ生成結果表示カード
 */
export const EggResultsCard: React.FC = () => {
  const { draftParams, getFilteredResults } = useEggStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();

  const isBootTimingMode = draftParams.seedSourceMode === 'boot-timing';
  const filteredResults = getFilteredResults();

  // advanceでソート
  const sortedResults = React.useMemo(() => {
    return [...filteredResults].sort((a, b) => a.advance - b.advance);
  }, [filteredResults]);

  // Boot-Timing モード時の列数計算
  const baseColSpan = 15; // advance + dir + v + ability + gender + nature + shiny + 6 IVs + pid + hp
  const npcColSpan = draftParams.considerNpcConsumption ? 1 : 0;
  const bootTimingColSpan = isBootTimingMode ? 3 : 0; // Timer0 + VCount + MT Seed
  const totalColSpan = baseColSpan + npcColSpan + bootTimingColSpan;

  const virtualization = useTableVirtualization({
    rowCount: sortedResults.length,
    defaultRowHeight: EGG_RESULTS_TABLE_ROW_HEIGHT,
    overscan: 12,
  });
  const virtualRows = virtualization.virtualRows;

  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;
  const shinyLabels = resolveLocaleValue(eggResultShinyLabels, locale);
  const genderLabels = resolveLocaleValue(eggResultGenderLabels, locale);
  const abilityLabels = resolveLocaleValue(eggResultAbilityLabels, locale);
  const stableLabels = resolveLocaleValue(eggResultStableLabels, locale);

  const formatIv = (iv: number): string => {
    return iv === IV_UNKNOWN ? '?' : String(iv);
  };

  const formatHiddenPower = (hp: { type: 'known'; hpType: number; power: number } | { type: 'unknown' }): string => {
    if (hp.type === 'unknown') {
      return eggResultUnknownHp[locale];
    }
    return `${hpTypeNames[hp.hpType] || '?'}/${hp.power}`;
  };

  const formatTimer0 = (row: EnumeratedEggDataWithBootTiming): string => {
    if (row.timer0 === undefined) return '-';
    return `0x${row.timer0.toString(16).toUpperCase().padStart(4, '0')}`;
  };

  const formatVcount = (row: EnumeratedEggDataWithBootTiming): string => {
    if (row.vcount === undefined) return '-';
    return `0x${row.vcount.toString(16).toUpperCase().padStart(2, '0')}`;
  };

  const formatMtSeed = (row: EnumeratedEggDataWithBootTiming): string => {
    return row.egg.mtSeedHex ?? '-';
  };

  // Seedから方向を計算
  const getDirection = (row: EnumeratedEggDataWithBootTiming): { arrow: string; value: number } | null => {
    // egg.lcgSeedHexを使用（各個体生成時のLCG Seed）
    const seedHex = row.egg.lcgSeedHex;
    if (!seedHex) return null;
    try {
      const seed = BigInt(seedHex);
      const value = calculateNeedleDirection(seed);
      const arrow = needleDirectionArrow(value);
      return { arrow, value };
    } catch {
      return null;
    }
  };

  return (
    <PanelCard
      icon={<TableIcon size={20} className="opacity-80" />}
      title={<span id="egg-results-title">{eggResultsPanelTitle[locale]}</span>}
      headerActions={
        <div className="flex items-center gap-2">
          <Badge variant="secondary">{sortedResults.length}</Badge>
          <EggExportButton
            results={sortedResults}
            isBootTimingMode={isBootTimingMode}
            disabled={sortedResults.length === 0}
          />
        </div>
      }
      className={isStack ? 'max-h-96' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      padding="none"
      spacing="none"
      contentClassName="p-0"
      aria-labelledby="egg-results-title"
      role="region"
    >
      <div
        ref={virtualization.containerRef}
        className="flex-1 min-h-0 overflow-y-auto"
        data-testid="egg-results-table"
      >
        {sortedResults.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 text-center text-muted-foreground py-8">
            {eggResultsEmptyMessage[locale]}
          </div>
        ) : (
          <Table className="min-w-full text-xs">
            <TableHeader className="sticky top-0 bg-muted text-xs">
              <TableRow className="text-left border-0">
                <TableHead scope="col" className="px-2 py-1 font-medium">{getEggResultHeader('advance', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('dir', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('dirValue', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium">{getEggResultHeader('ability', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('gender', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium">{getEggResultHeader('nature', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('shiny', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('hp', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('atk', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('def', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('spa', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('spd', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('spe', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium">{getEggResultHeader('pid', locale)}</TableHead>
                <TableHead scope="col" className="px-2 py-1 font-medium">{getEggResultHeader('hiddenPower', locale)}</TableHead>
                {isBootTimingMode && (
                  <>
                    <TableHead scope="col" className="px-2 py-1 font-medium">Timer0</TableHead>
                    <TableHead scope="col" className="px-2 py-1 font-medium">VCount</TableHead>
                    <TableHead scope="col" className="px-2 py-1 font-medium">MT Seed</TableHead>
                  </>
                )}
                {draftParams.considerNpcConsumption && (
                  <TableHead scope="col" className="px-2 py-1 font-medium text-center">{getEggResultHeader('stable', locale)}</TableHead>
                )}
              </TableRow>
            </TableHeader>
            <TableBody>
              {virtualization.paddingTop > 0 ? (
                <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                  <TableCell
                    colSpan={totalColSpan}
                    className="p-0 border-0"
                    style={{ height: virtualization.paddingTop }}
                  />
                </TableRow>
              ) : null}
              {virtualRows.map(virtualRow => {
                const row = sortedResults[virtualRow.index];
                if (!row) {
                  return null;
                }
                const direction = getDirection(row);
                return (
                  <TableRow
                    key={virtualRow.index}
                    ref={virtualization.measureRow}
                    data-index={virtualRow.index}
                    className="odd:bg-background even:bg-muted/30 border-0"
                    data-testid="egg-result-row"
                  >
                    <TableCell className="px-2 py-1 font-mono tabular-nums">{row.advance}</TableCell>
                    <TableCell className="px-2 py-1 text-center font-arrows">{direction?.arrow ?? '-'}</TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{direction?.value ?? '-'}</TableCell>
                    <TableCell className="px-2 py-1">{abilityLabels[row.egg.ability as 0 | 1 | 2] || row.egg.ability}</TableCell>
                    <TableCell className="px-2 py-1 text-center">{genderLabels[row.egg.gender] || row.egg.gender}</TableCell>
                    <TableCell className="px-2 py-1 whitespace-nowrap">{natureName(row.egg.nature, locale)}</TableCell>
                    <TableCell className="px-2 py-1 text-center">
                      {row.egg.shiny > 0 ? (
                        <span className={row.egg.shiny === 2 ? 'text-yellow-500' : 'text-blue-500'}>
                          {shinyLabels[row.egg.shiny as 1 | 2]}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">{shinyLabels[0]}</span>
                      )}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{formatIv(row.egg.ivs[0])}</TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{formatIv(row.egg.ivs[1])}</TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{formatIv(row.egg.ivs[2])}</TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{formatIv(row.egg.ivs[3])}</TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{formatIv(row.egg.ivs[4])}</TableCell>
                    <TableCell className="px-2 py-1 font-mono tabular-nums text-center">{formatIv(row.egg.ivs[5])}</TableCell>
                    <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                      {row.egg.pid.toString(16).toUpperCase().padStart(8, '0')}
                    </TableCell>
                    <TableCell className="px-2 py-1 whitespace-nowrap">{formatHiddenPower(row.egg.hiddenPower)}</TableCell>
                    {isBootTimingMode && (
                      <>
                        <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{formatTimer0(row)}</TableCell>
                        <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{formatVcount(row)}</TableCell>
                        <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{formatMtSeed(row)}</TableCell>
                      </>
                    )}
                    {draftParams.considerNpcConsumption && (
                      <TableCell className="px-2 py-1 text-center">
                        {row.isStable ? stableLabels.yes : stableLabels.no}
                      </TableCell>
                    )}
                  </TableRow>
                );
              })}
              {virtualization.paddingBottom > 0 ? (
                <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                  <TableCell
                    colSpan={totalColSpan}
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
};
