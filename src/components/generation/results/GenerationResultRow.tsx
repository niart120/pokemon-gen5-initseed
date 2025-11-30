import React from 'react';
import { TableRow, TableCell } from '@/components/ui/table';
import type { UiReadyPokemonData } from '@/types/pokemon-resolved';
import { shinyLabel, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import { formatKeyInputForDisplay } from '@/lib/utils/key-input';

interface GenerationResultRowProps {
  row: UiReadyPokemonData;
  locale: 'ja' | 'en';
  unknownLabel: string;
  measureRow: (node: HTMLTableRowElement | null) => void;
  virtualIndex: number;
}

export const GenerationResultRow: React.FC<GenerationResultRowProps> = ({
  row,
  locale,
  unknownLabel,
  measureRow,
  virtualIndex,
}) => {
  const stats = row.stats;
  const hpDisplay = stats ? stats.hp : '--';
  const atkDisplay = stats ? stats.attack : '--';
  const defDisplay = stats ? stats.defense : '--';
  const spaDisplay = stats ? stats.specialAttack : '--';
  const spdDisplay = stats ? stats.specialDefense : '--';
  const speDisplay = stats ? stats.speed : '--';
  const natureDisplay = row.natureName;
  const needleDir = calculateNeedleDirection(row.seed);
  const showBootTimingMeta = row.seedSourceMode === 'boot-timing';
  const timer0Display = showBootTimingMeta ? formatTimer0Hex(row.timer0) : '--';
  const vcountDisplay = showBootTimingMeta ? formatVCountHex(row.vcount) : '--';
  const bootTimestampDisplay = showBootTimingMeta
    ? formatBootTimestampDisplay(row.bootTimestampIso, locale)
    : '';
  const keyInputDisplay = showBootTimingMeta
    ? formatKeyInputForDisplay(null, row.keyInputNames)
    : '';
  return (
    <TableRow
      ref={measureRow}
      data-index={virtualIndex}
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
      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{timer0Display}</TableCell>
      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{vcountDisplay}</TableCell>
      <TableCell className="px-2 py-1 whitespace-nowrap">{bootTimestampDisplay}</TableCell>
      <TableCell className="px-2 py-1 font-mono whitespace-nowrap">{keyInputDisplay}</TableCell>
    </TableRow>
  );
};
