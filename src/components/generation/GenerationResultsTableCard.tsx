import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Badge } from '@/components/ui/badge';
import { Table as TableIcon } from '@phosphor-icons/react';
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { GenerationExportButton } from './GenerationExportButton';
import { useAppStore } from '@/store/app-store';
import { selectFilteredDisplayRows, selectFilteredSortedResults } from '@/store/generation-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useTableVirtualization } from '@/hooks/use-table-virtualization';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatGenerationResultsTableTitle,
  formatGenerationResultsCount,
  formatGenerationProcessingDuration,
  generationResultsTableCaption,
  generationResultsTableUnknownLabel,
  generationResultsTableEmptyMessage,
  generationResultsTableInitialMessage,
  resolveGenerationResultsTableHeaders,
} from '@/lib/i18n/strings/generation-results-table';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { GenerationResultRow } from '@/components/generation/results/GenerationResultRow';
import { buildGenerationResultRowKey } from '@/lib/generation/result-formatters';

type AppStoreState = ReturnType<typeof useAppStore.getState>;
const GENERATION_RESULTS_COLUMN_COUNT = 19;
const GENERATION_TABLE_ROW_HEIGHT = 34;

export const GenerationResultsTableCard: React.FC = () => {
  const locale = useLocale();
  const rows = useAppStore((state: AppStoreState) => selectFilteredDisplayRows(state, locale));
  const filteredRawRows = useAppStore((state: AppStoreState) => selectFilteredSortedResults(state, locale));
  const total = useAppStore(s => s.results.length);
  const lastCompletion = useAppStore(s => s.lastCompletion);
  const encounterTable = useAppStore((state) => state.encounterTable);
  const genderRatios = useAppStore((state) => state.genderRatios);
  const abilityCatalog = useAppStore((state) => state.abilityCatalog);
  const version = useAppStore((state) => (state.params?.version ?? state.draftParams.version ?? 'B') as 'B' | 'W' | 'B2' | 'W2');
  const baseSeed = useAppStore((state) => {
    if (state.params?.baseSeed !== undefined) return state.params.baseSeed;
    const hex = state.draftParams.baseSeedHex;
    if (typeof hex === 'string') {
      const normalized = hex.trim();
      if (normalized !== '') {
        try {
          return BigInt('0x' + normalized.replace(/^0x/i, ''));
        } catch {
          return undefined;
        }
      }
    }
    return undefined;
  });
  const { isStack } = useResponsiveLayout();
  const headers = React.useMemo(() => resolveGenerationResultsTableHeaders(locale), [locale]);
  const panelTitle = formatGenerationResultsTableTitle(rows.length, total, locale);
  const resultsCount = formatGenerationResultsCount(rows.length, locale);
  const caption = resolveLocaleValue(generationResultsTableCaption, locale);
  const unknownLabel = resolveLocaleValue(generationResultsTableUnknownLabel, locale);
  const virtualization = useTableVirtualization({
    rowCount: rows.length,
    defaultRowHeight: GENERATION_TABLE_ROW_HEIGHT,
    overscan: 12,
  });
  const virtualRows = virtualization.virtualRows;
  // スクロール方針
  // - モバイル(isStack): カード内でスクロール(overflow-y-auto)にして、ドキュメント高さの膨張を防ぐ
  // - デスクトップ: 呼び出し元の指定を尊重（既定はfalseでカード内スクロール）
  return (
    <PanelCard
      icon={<TableIcon size={20} className="opacity-80" />}
      title={<span id="gen-results-table-title">{panelTitle}</span>}
      headerActions={
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="secondary" className="flex-shrink-0">
            {resultsCount}
          </Badge>
          {lastCompletion !== null && (
            <Badge variant="outline" className="flex-shrink-0 text-xs">
              {formatGenerationProcessingDuration(lastCompletion.elapsedMs)}
            </Badge>
          )}
          <GenerationExportButton
            rows={filteredRawRows}
            encounterTable={encounterTable}
            genderRatios={genderRatios}
            abilityCatalog={abilityCatalog}
            version={version}
            baseSeed={baseSeed}
            disabled={filteredRawRows.length === 0}
          />
        </div>
      }
      className={isStack ? 'max-h-96' : 'min-h-96'}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      padding="none"
      spacing="none"
      contentClassName="p-0"
      aria-labelledby="gen-results-table-title"
      role="region"
    >
      <div
        ref={virtualization.containerRef}
        className="flex-1 min-h-0 overflow-auto"
      >
        {rows.length === 0 ? (
          <div className="flex h-full items-center justify-center px-6 text-center text-muted-foreground py-8">
            {resolveLocaleValue(
              total === 0 ? generationResultsTableInitialMessage : generationResultsTableEmptyMessage,
              locale,
            )}
          </div>
        ) : (
        <Table className="w-max min-w-full text-xs" aria-describedby="gen-results-table-desc">
          <TableCaption id="gen-results-table-desc">
            {caption}
          </TableCaption>
          <TableHeader className="sticky top-0 bg-muted text-xs">
            <TableRow className="text-left border-0">
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">
                {headers.advance.label}
                {headers.advance.sr ? <span className="sr-only">{headers.advance.sr}</span> : null}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-center">{headers.direction.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-center">{headers.directionValue.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.species.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.ability.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-center">
                {headers.gender.label}
                {headers.gender.sr ? <span className="sr-only">{headers.gender.sr}</span> : null}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.nature.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-center">{headers.shiny.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.level.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.hp.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.attack.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.defense.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.specialAttack.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.specialDefense.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap text-right">{headers.speed.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.seed.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.pid.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.timer0.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium text-xs whitespace-nowrap">{headers.vcount.label}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {virtualization.paddingTop > 0 ? (
              <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                <TableCell
                  colSpan={GENERATION_RESULTS_COLUMN_COUNT}
                  className="p-0 border-0"
                  style={{ height: virtualization.paddingTop }}
                />
              </TableRow>
            ) : null}
          {virtualRows.map(virtualRow => {
            const row = rows[virtualRow.index];
            if (!row) {
              return null;
            }
            const rowKey = buildGenerationResultRowKey(row.advance, row.timer0, row.vcount);
            return (
              <GenerationResultRow
                key={rowKey}
                row={row}
                locale={locale}
                unknownLabel={unknownLabel}
                measureRow={virtualization.measureRow}
                virtualIndex={virtualRow.index}
              />
            );
          })}
            {virtualization.paddingBottom > 0 ? (
              <TableRow aria-hidden="true" className="border-0 pointer-events-none">
                <TableCell
                  colSpan={GENERATION_RESULTS_COLUMN_COUNT}
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
