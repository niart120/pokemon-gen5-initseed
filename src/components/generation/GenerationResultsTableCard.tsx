import React from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Table } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { selectFilteredSortedResults } from '@/store/generation-store';
import { pidHex, natureName, shinyLabel } from '@/lib/utils/format-display';

export const GenerationResultsTableCard: React.FC = () => {
  const results = useAppStore(selectFilteredSortedResults);
  const total = useAppStore(s => s.results.length);
  return (
    <Card className="py-2 flex flex-col flex-1 min-h-0" aria-labelledby="gen-results-table-title" role="region">
      <StandardCardHeader icon={<Table size={20} className="opacity-80" />} title={<span id="gen-results-table-title">Results ({results.length}) / Total {total}</span>} />
      <StandardCardContent className="p-0 overflow-auto">
        <table className="min-w-full text-xs" aria-describedby="gen-results-table-desc">
          <caption id="gen-results-table-desc" className="sr-only">Filtered generation results list.</caption>
          <thead className="sticky top-0 bg-muted text-[11px]">
            <tr className="text-left">
              <th scope="col" className="px-2 py-1 font-medium">Advance</th>
              <th scope="col" className="px-2 py-1 font-medium">PID</th>
              <th scope="col" className="px-2 py-1 font-medium">Nature</th>
              <th scope="col" className="px-2 py-1 font-medium">Shiny</th>
            </tr>
          </thead>
          <tbody>
            {results.map(r=> (
              <tr key={r.advance} className="odd:bg-background even:bg-muted/30">
                <td className="px-2 py-1 font-mono tabular-nums">{r.advance}</td>
                <td className="px-2 py-1 font-mono">{pidHex(r.pid)}</td>
                <td className="px-2 py-1">{natureName(r.nature)}</td>
                <td className="px-2 py-1">{shinyLabel(r.shiny_type)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="sr-only" aria-live="polite">{results.length} filtered results shown of {total} total.</div>
      </StandardCardContent>
    </Card>
  );
};
