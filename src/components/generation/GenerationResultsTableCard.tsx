import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Table as TableIcon } from '@phosphor-icons/react';
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useAppStore } from '@/store/app-store';
import { selectFilteredDisplayRows } from '@/store/generation-store';
import { shinyLabel, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useTableVirtualization } from '@/hooks/use-table-virtualization';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatGenerationResultsTableTitle,
  generationResultsTableCaption,
  generationResultsTableUnknownLabel,
  resolveGenerationResultsTableHeaders,
} from '@/lib/i18n/strings/generation-results-table';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';

type AppStoreState = ReturnType<typeof useAppStore.getState>;
const GENERATION_RESULTS_COLUMN_COUNT = 17;
const GENERATION_TABLE_ROW_HEIGHT = 34;

export const GenerationResultsTableCard: React.FC = () => {
  const locale = useLocale();
  const rows = useAppStore((state: AppStoreState) => selectFilteredDisplayRows(state, locale));
  const total = useAppStore(s => s.results.length);
  const { isStack } = useResponsiveLayout();
  const headers = React.useMemo(() => resolveGenerationResultsTableHeaders(locale), [locale]);
  const panelTitle = formatGenerationResultsTableTitle(rows.length, total, locale);
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
        className="flex-1 min-h-0 overflow-y-auto"
      >
        <Table className="min-w-full text-xs" aria-describedby="gen-results-table-desc">
          <TableCaption id="gen-results-table-desc">
            {caption}
          </TableCaption>
          <TableHeader className="sticky top-0 bg-muted text-[11px]">
            <TableRow className="text-left border-0">
              <TableHead scope="col" className="px-2 py-1 font-medium w-14">
                {headers.advance.label}
                {headers.advance.sr ? <span className="sr-only">{headers.advance.sr}</span> : null}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-10">{headers.direction.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-8">{headers.directionValue.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32">{headers.species.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32">{headers.ability.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-8">
                {headers.gender.label}
                {headers.gender.sr ? <span className="sr-only">{headers.gender.sr}</span> : null}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-24">{headers.nature.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-16">{headers.shiny.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-10">{headers.level.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.hp.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.attack.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.defense.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.specialAttack.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.specialDefense.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.speed.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium min-w-[120px] w-36">{headers.seed.label}</TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium w-32">{headers.pid.label}</TableHead>
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
            const needleDir = calculateNeedleDirection(row.seed);
            const stats = row.stats;
            const hpDisplay = stats ? stats.hp : '--';
            const atkDisplay = stats ? stats.attack : '--';
            const defDisplay = stats ? stats.defense : '--';
            const spaDisplay = stats ? stats.specialAttack : '--';
            const spdDisplay = stats ? stats.specialDefense : '--';
            const speDisplay = stats ? stats.speed : '--';
            const natureDisplay = row.natureName;
            return (
              <TableRow
                key={row.advance}
                ref={virtualization.measureRow}
                data-index={virtualRow.index}
                className="odd:bg-background even:bg-muted/30 border-0"
              >
                <TableCell className="px-2 py-1 font-mono tabular-nums">{row.advance}</TableCell>
                <TableCell className="px-2 py-1 text-center font-arrows">{needleDirectionArrow(needleDir)}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums">{needleDir}</TableCell>
                <TableCell className="px-2 py-1 truncate max-w-[120px]" title={row.speciesName || unknownLabel}>
                  {row.speciesName || unknownLabel}
                </TableCell>
                <TableCell className="px-2 py-1 truncate max-w-[120px]" title={row.abilityName || unknownLabel}>
                  {row.abilityName || unknownLabel}
                </TableCell>
                <TableCell className="px-2 py-1">{row.gender || '?'}</TableCell>
                <TableCell className="px-2 py-1 whitespace-nowrap">{natureDisplay}</TableCell>
                <TableCell className="px-2 py-1">{shinyLabel(row.shinyType, locale)}</TableCell>
                <TableCell className="px-2 py-1 tabular-nums">{row.level ?? ''}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums text-right">{hpDisplay}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums text-right">{atkDisplay}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums text-right">{defDisplay}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums text-right">{spaDisplay}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums text-right">{spdDisplay}</TableCell>
                <TableCell className="px-2 py-1 font-mono tabular-nums text-right">{speDisplay}</TableCell>
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{row.seedHex}</TableCell>
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{row.pidHex}</TableCell>
              </TableRow>
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
      </div>
    </PanelCard>
  );
};
