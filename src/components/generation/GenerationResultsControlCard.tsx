import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/app-store';
import { exportGenerationResults } from '@/lib/export/generation-exporter';

export const GenerationResultsControlCard: React.FC = () => {
  const { filters, applyFilters, resetGenerationFilters, results, clearResults } = useAppStore();
  const [natureInput, setNatureInput] = useState( filters.natureIds.join(',') );
  const [advMin, setAdvMin] = useState(filters.advanceRange?.min ?? '');
  const [advMax, setAdvMax] = useState(filters.advanceRange?.max ?? '');
  const [shinyTypesInput, setShinyTypesInput] = useState(filters.shinyTypes?.join(',') || '');

  const parseList = (txt: string) => txt.split(',').map(s=>s.trim()).filter(Boolean).map(Number).filter(n=> Number.isFinite(n));

  const onApplyNature = () => {
    const list = parseList(natureInput).filter(n=> n>=0 && n<=24);
    applyFilters({ natureIds: list });
  };
  const onApplyShinyTypes = () => {
    const list = parseList(shinyTypesInput).filter(n=> n>=0 && n<=2);
    applyFilters({ shinyTypes: list.length?list:undefined });
  };
  const onApplyAdvRange = () => {
    const min = advMin === '' ? undefined : Number(advMin);
    const max = advMax === '' ? undefined : Number(advMax);
    applyFilters({ advanceRange: (min==null && max==null)? undefined : { min, max } });
  };
  const onExport = (format: 'csv'|'json'|'txt') => {
    const blob = new Blob([exportGenerationResults(results, { format })], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generation-results.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="p-3 flex flex-col gap-2">
      <CardHeader className="py-0"><CardTitle className="text-sm flex justify-between items-center">Results Control
        <div className="flex gap-2 text-[10px]">
          <button className="px-2 py-1 border rounded" onClick={()=>onExport('csv')} disabled={!results.length}>CSV</button>
          <button className="px-2 py-1 border rounded" onClick={()=>onExport('json')} disabled={!results.length}>JSON</button>
          <button className="px-2 py-1 border rounded" onClick={()=>onExport('txt')} disabled={!results.length}>TXT</button>
          <button className="px-2 py-1 border rounded text-red-600" onClick={clearResults} disabled={!results.length}>Clear</button>
          <button className="px-2 py-1 border rounded" onClick={resetGenerationFilters}>Reset</button>
        </div>
      </CardTitle></CardHeader>
      <CardContent className="p-0 flex flex-col gap-2 text-xs">
        <div className="flex flex-wrap gap-4 items-center">
          <label className="flex items-center gap-1">
            <input type="checkbox" checked={filters.shinyOnly} onChange={e=>applyFilters({ shinyOnly: e.target.checked })} /> ShinyOnly
          </label>
          <div className="flex items-center gap-1">
            <span>Sort:</span>
            <select
              value={filters.sortField}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                const v = e.target.value as 'advance' | 'pid' | 'nature' | 'shiny';
                applyFilters({ sortField: v });
              }}
              className="border rounded px-1 py-0.5"
            >
              <option value="advance">Advance</option>
              <option value="pid">PID</option>
              <option value="nature">Nature</option>
              <option value="shiny">Shiny</option>
            </select>
            <select
              value={filters.sortOrder}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                const v = e.target.value === 'desc' ? 'desc' : 'asc';
                applyFilters({ sortOrder: v });
              }}
              className="border rounded px-1 py-0.5"
            >
              <option value="asc">Asc</option>
              <option value="desc">Desc</option>
            </select>
          </div>
          <div className="flex items-center gap-1">
            <span>Advance:</span>
            <input value={advMin} onChange={e=>setAdvMin(e.target.value)} placeholder="min" className="w-16 border rounded px-1 py-0.5" />
            <input value={advMax} onChange={e=>setAdvMax(e.target.value)} placeholder="max" className="w-16 border rounded px-1 py-0.5" />
            <button className="px-2 py-0.5 border rounded" onClick={onApplyAdvRange}>Set</button>
          </div>
        </div>
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-1">
            <span>Natures:</span>
            <input value={natureInput} onChange={e=>setNatureInput(e.target.value)} placeholder="e.g. 1,2,6" className="w-40 border rounded px-1 py-0.5" />
            <button className="px-2 py-0.5 border rounded" onClick={onApplyNature}>Apply</button>
          </div>
          <div className="flex items-center gap-1">
            <span>ShinyTypes:</span>
            <input value={shinyTypesInput} onChange={e=>setShinyTypesInput(e.target.value)} placeholder="1,2" className="w-24 border rounded px-1 py-0.5" />
            <button className="px-2 py-0.5 border rounded" onClick={onApplyShinyTypes}>Apply</button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
