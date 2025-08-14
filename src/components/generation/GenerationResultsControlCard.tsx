import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { useAppStore } from '@/store/app-store';
import { exportGenerationResults } from '@/lib/export/generation-exporter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { FunnelSimple, DownloadSimple, Trash, ArrowsDownUp } from '@phosphor-icons/react';
import { useResponsiveLayout } from '@/hooks/use-mobile';

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
  const { isStack } = useResponsiveLayout();
  return (
    <Card className={`py-2 flex flex-col ${isStack ? 'max-h-96' : 'h-full min-h-64'}`} aria-labelledby="gen-results-control-title" role="region">
      <StandardCardHeader icon={<FunnelSimple size={20} className="opacity-80" />} title={<span id="gen-results-control-title">Results Control</span>} />
  <StandardCardContent noScroll={isStack}>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Export and utility buttons">
          <Button size="sm" variant="outline" disabled={!results.length} onClick={()=>onExport('csv')}><DownloadSimple size={14}/>CSV</Button>
          <Button size="sm" variant="outline" disabled={!results.length} onClick={()=>onExport('json')}><DownloadSimple size={14}/>JSON</Button>
          <Button size="sm" variant="outline" disabled={!results.length} onClick={()=>onExport('txt')}><DownloadSimple size={14}/>TXT</Button>
          <Button size="sm" variant="destructive" disabled={!results.length} onClick={clearResults}><Trash size={14}/>Clear</Button>
          <Button size="sm" variant="ghost" onClick={resetGenerationFilters}>Reset</Button>
        </div>
        <Separator />
        <form onSubmit={e=> e.preventDefault()} className="flex flex-col gap-4 text-xs" aria-describedby="results-filter-hint">
          {/* Primary filters & sorting */}
          <fieldset className="space-y-3" aria-labelledby="gf-primary-label" role="group">
            <div id="gf-primary-label" className="text-[10px] font-medium tracking-wide uppercase text-muted-foreground">Primary</div>
            <div className="flex flex-wrap gap-4 items-center">
              <div className="flex items-center gap-2">
                <Checkbox id="shiny-only" aria-labelledby="lbl-shiny-only" checked={filters.shinyOnly} onCheckedChange={v=>applyFilters({ shinyOnly: Boolean(v) })} />
                <Label id="lbl-shiny-only" htmlFor="shiny-only" className="text-xs">Shiny Only</Label>
              </div>
              <div className="flex items-center gap-2" aria-label="Sort controls">
                <span id="sort-field-label" className="sr-only">Sort field</span>
                <Select value={filters.sortField} onValueChange={v=>applyFilters({ sortField: v as typeof filters.sortField })}>
                  <SelectTrigger id="sort-field" size="sm" className="w-[110px]" aria-labelledby="sort-field sort-field-label">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="advance">Advance</SelectItem>
                    <SelectItem value="pid">PID</SelectItem>
                    <SelectItem value="nature">Nature</SelectItem>
                    <SelectItem value="shiny">Shiny</SelectItem>
                  </SelectContent>
                </Select>
                <ArrowsDownUp size={14} className="opacity-50" aria-hidden="true" />
                <span id="sort-order-label" className="sr-only">Sort order</span>
                <Select value={filters.sortOrder} onValueChange={v=>applyFilters({ sortOrder: v as typeof filters.sortOrder })}>
                  <SelectTrigger id="sort-order" size="sm" className="w-[80px]" aria-labelledby="sort-order sort-order-label">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="asc">Asc</SelectItem>
                    <SelectItem value="desc">Desc</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-1" aria-label="Advance range filter">
                <Label htmlFor="adv-min" className="sr-only">Advance minimum</Label>
                <Input id="adv-min" value={advMin} onChange={e=>setAdvMin(e.target.value)} placeholder="min" className="h-8 w-20" inputMode="numeric" aria-describedby="adv-range-hint" />
                <Label htmlFor="adv-max" className="sr-only">Advance maximum</Label>
                <Input id="adv-max" value={advMax} onChange={e=>setAdvMax(e.target.value)} placeholder="max" className="h-8 w-20" inputMode="numeric" aria-describedby="adv-range-hint" />
                <Button type="button" size="sm" variant="secondary" onClick={onApplyAdvRange}>Set</Button>
                <span id="adv-range-hint" className="sr-only">Set minimum and/or maximum advance indices</span>
              </div>
            </div>
          </fieldset>
          <Separator />
          {/* Secondary filters */}
            <fieldset className="space-y-3" aria-labelledby="gf-secondary-label" role="group">
              <div id="gf-secondary-label" className="text-[10px] font-medium tracking-wide uppercase text-muted-foreground">Secondary</div>
              <div className="flex flex-wrap gap-6 items-center">
                <div className="flex items-center gap-2" aria-label="Nature ID list filter">
                  <Label htmlFor="nature-input" className="sr-only">Nature IDs (comma separated)</Label>
                  <Input id="nature-input" value={natureInput} onChange={e=>setNatureInput(e.target.value)} placeholder="1,2,6" className="h-8 w-40" aria-describedby="nature-hint" />
                  <Button type="button" size="sm" variant="secondary" onClick={onApplyNature}>Apply</Button>
                  <span id="nature-hint" className="sr-only">Enter nature IDs 0 to 24 separated by commas</span>
                </div>
                <div className="flex items-center gap-2" aria-label="Shiny type list filter">
                  <Label htmlFor="shiny-types-input" className="sr-only">Shiny types (comma separated)</Label>
                  <Input id="shiny-types-input" value={shinyTypesInput} onChange={e=>setShinyTypesInput(e.target.value)} placeholder="1,2" className="h-8 w-28" aria-describedby="shiny-types-hint" />
                  <Button type="button" size="sm" variant="secondary" onClick={onApplyShinyTypes}>Apply</Button>
                  <span id="shiny-types-hint" className="sr-only">Enter shiny type codes 0 normal 1 square 2 star separated by commas</span>
                </div>
              </div>
            </fieldset>
  </form>
  <div id="results-filter-hint" className="sr-only" aria-live="polite">Results filtering controls configured.</div>
      </StandardCardContent>
    </Card>
  );
};
