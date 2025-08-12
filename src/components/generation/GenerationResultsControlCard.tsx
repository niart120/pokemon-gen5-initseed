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
        <div className="flex gap-2 text-[10px]" role="group" aria-label="Export and utility buttons">
          <button className="px-2 py-1 border rounded" onClick={()=>onExport('csv')} disabled={!results.length} aria-disabled={!results.length} aria-label="Export CSV">CSV</button>
          <button className="px-2 py-1 border rounded" onClick={()=>onExport('json')} disabled={!results.length} aria-disabled={!results.length} aria-label="Export JSON">JSON</button>
            <button className="px-2 py-1 border rounded" onClick={()=>onExport('txt')} disabled={!results.length} aria-disabled={!results.length} aria-label="Export text">TXT</button>
          <button className="px-2 py-1 border rounded text-red-600" onClick={clearResults} disabled={!results.length} aria-disabled={!results.length}>Clear</button>
          <button className="px-2 py-1 border rounded" onClick={resetGenerationFilters}>Reset</button>
        </div>
      </CardTitle></CardHeader>
      <CardContent className="p-0 flex flex-col gap-3 text-xs" as-child>
        <form onSubmit={e=> e.preventDefault()} className="flex flex-col gap-3">
          <fieldset className="flex flex-wrap gap-4 items-center">
            <legend className="sr-only">Sorting and primary filters</legend>
            <label className="flex items-center gap-1" aria-label="Show only shiny results">
              <input type="checkbox" checked={filters.shinyOnly} onChange={e=>applyFilters({ shinyOnly: e.target.checked })} aria-checked={filters.shinyOnly} /> ShinyOnly
            </label>
            <div className="flex items-center gap-1" aria-label="Sort controls">
              <span className="sr-only" id="sort-field-label">Sort field</span>
              <select
                aria-labelledby="sort-field-label"
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
              <span className="sr-only" id="sort-order-label">Sort order</span>
              <select
                aria-labelledby="sort-order-label"
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
            <div className="flex items-center gap-1" aria-label="Advance range filter">
              <label className="sr-only" htmlFor="adv-min">Advance minimum</label>
              <input id="adv-min" value={advMin} onChange={e=>setAdvMin(e.target.value)} placeholder="min" className="w-16 border rounded px-1 py-0.5" inputMode="numeric" aria-describedby="adv-range-hint" />
              <label className="sr-only" htmlFor="adv-max">Advance maximum</label>
              <input id="adv-max" value={advMax} onChange={e=>setAdvMax(e.target.value)} placeholder="max" className="w-16 border rounded px-1 py-0.5" inputMode="numeric" aria-describedby="adv-range-hint" />
              <button type="button" className="px-2 py-0.5 border rounded" onClick={onApplyAdvRange} aria-label="Apply advance range">Set</button>
              <span id="adv-range-hint" className="sr-only">Set minimum and/or maximum advance indices</span>
            </div>
          </fieldset>
          <fieldset className="flex flex-wrap gap-4 items-center">
            <legend className="sr-only">Secondary filters</legend>
            <div className="flex items-center gap-1" aria-label="Nature ID list filter">
              <label className="sr-only" htmlFor="nature-input">Nature IDs (comma separated)</label>
              <input id="nature-input" value={natureInput} onChange={e=>setNatureInput(e.target.value)} placeholder="1,2,6" className="w-40 border rounded px-1 py-0.5" aria-describedby="nature-hint" />
              <button type="button" className="px-2 py-0.5 border rounded" onClick={onApplyNature} aria-label="Apply nature IDs">Apply</button>
              <span id="nature-hint" className="sr-only">Enter nature IDs 0 to 24 separated by commas</span>
            </div>
            <div className="flex items-center gap-1" aria-label="Shiny type list filter">
              <label className="sr-only" htmlFor="shiny-types-input">Shiny types (comma separated)</label>
              <input id="shiny-types-input" value={shinyTypesInput} onChange={e=>setShinyTypesInput(e.target.value)} placeholder="1,2" className="w-24 border rounded px-1 py-0.5" aria-describedby="shiny-types-hint" />
              <button type="button" className="px-2 py-0.5 border rounded" onClick={onApplyShinyTypes} aria-label="Apply shiny types">Apply</button>
              <span id="shiny-types-hint" className="sr-only">Enter shiny type codes 0 normal 1 square 2 star separated by commas</span>
            </div>
          </fieldset>
        </form>
      </CardContent>
    </Card>
  );
};
