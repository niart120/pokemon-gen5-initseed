import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/app-store';
import { selectFilteredSortedResults } from '@/store/generation-store';

export const GenerationResultsTableCard: React.FC = () => {
  const results = useAppStore(selectFilteredSortedResults);
  const total = useAppStore(s => s.results.length);
  return (
    <Card className="p-3 flex flex-col gap-2 flex-1 min-h-0">
      <CardHeader className="py-0"><CardTitle className="text-sm">Results ({results.length}) / Total {total}</CardTitle></CardHeader>
      <CardContent className="p-0 flex-1 min-h-0 overflow-auto">
        <table className="min-w-full text-xs">
          <thead className="sticky top-0 bg-muted">
            <tr>
              <th className="text-left px-2 py-1">Advance</th>
              <th className="text-left px-2 py-1">PID</th>
              <th className="text-left px-2 py-1">Nature</th>
              <th className="text-left px-2 py-1">Shiny</th>
            </tr>
          </thead>
          <tbody>
            {results.map(r=> (
              <tr key={r.advance} className="odd:bg-background even:bg-muted/30">
                <td className="px-2 py-1 font-mono">{r.advance}</td>
                <td className="px-2 py-1 font-mono">0x{(r.pid>>>0).toString(16).padStart(8,'0')}</td>
                <td className="px-2 py-1">{r.nature}</td>
                <td className="px-2 py-1">{r.shiny_type===0?'No':(r.shiny_type===1?'Square':'Star')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
};
