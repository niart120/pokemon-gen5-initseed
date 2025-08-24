import React, { useState, useMemo } from 'react';
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
import { getGeneratedSpeciesById } from '@/data/species/generated';
import { useResponsiveLayout } from '@/hooks/use-mobile';

// === Precomputed species options (Gen5: 1..649) ===
// 1回だけ構築し再利用。検索で使う正規化済み文字列を保持。
interface SpeciesOptionEntry { id: number; labelJa: string; labelEn: string; normJa: string; normEn: string; }
const ALL_SPECIES_OPTIONS: SpeciesOptionEntry[] = (() => {
  const arr: SpeciesOptionEntry[] = [];
  for (let id = 1; id <= 649; id++) {
    const s = getGeneratedSpeciesById(id);
    if (!s) continue;
    const ja = s.names.ja;
    const en = s.names.en;
    arr.push({ id, labelJa: ja, labelEn: en, normJa: ja.toLowerCase(), normEn: en.toLowerCase() });
  }
  return arr;
})();

export const GenerationResultsControlCard: React.FC = () => {
  // NOTE(perf): 以前は useAppStore() 全体取得で encounterField 変更時など不要再レンダーが発生していたため細粒度購読へ分割
  const filters = useAppStore(s => s.filters);
  const applyFilters = useAppStore(s => s.applyFilters);
  const resetGenerationFilters = useAppStore(s => s.resetGenerationFilters);
  const results = useAppStore(s => s.results); // export 用に実体参照
  const clearResults = useAppStore(s => s.clearResults);
  // Species filter UI: reuse ParamCard style listbox (Radix Select) for adding one at a time
  // filters.speciesIds が未定義のとき毎回新しい [] を生成すると useMemo 依存が常に変化するため安定化
  const selectedSpeciesIds = useMemo(() => filters.speciesIds ?? [], [filters.speciesIds]);
  const addSpecies = (id:number) => {
    if (selectedSpeciesIds.includes(id)) return;
    applyFilters({ speciesIds: [...selectedSpeciesIds, id] });
  };
  const removeSpecies = (id:number) => {
    const next = selectedSpeciesIds.filter(s=>s!==id);
    applyFilters({ speciesIds: next.length? next: undefined, abilityIndices: undefined, genders: undefined });
  };
  const speciesSelectItems = useMemo(()=> ALL_SPECIES_OPTIONS.map(o=>({ value:o.id.toString(), label:o.labelJa })), []);
  // Abilities derived from selected species (union of indices that exist)
  const availableAbilityIndices: (0|1|2)[] = useMemo(() => {
    if (!selectedSpeciesIds.length) return [];
    const set = new Set<number>();
    for (const id of selectedSpeciesIds) {
      const s = getGeneratedSpeciesById(id);
      if (!s) continue;
      if (s.abilities.ability1) set.add(0);
      if (s.abilities.ability2) set.add(1);
      if (s.abilities.hidden) set.add(2);
    }
    return Array.from(set).sort() as (0|1|2)[];
  }, [selectedSpeciesIds]);
  const toggleAbilityIndex = (idx:0|1|2) => {
    const current = filters.abilityIndices || [];
    const exists = current.includes(idx);
    const next = exists ? current.filter(i=>i!==idx) : [...current, idx];
    applyFilters({ abilityIndices: next.length? next : undefined });
  };
  const genderOptions: ('M'|'F'|'N')[] = ['M','F','N'];
  const toggleGender = (g:'M'|'F'|'N') => {
    const current = filters.genders || [];
    const exists = current.includes(g);
    const next = exists ? current.filter(x=>x!==g) : [...current, g];
    applyFilters({ genders: next.length? next : undefined });
  };
  // Level range (independent)
  const [lvlMin, setLvlMin] = useState(filters.levelRange?.min ?? '');
  const [lvlMax, setLvlMax] = useState(filters.levelRange?.max ?? '');
  const applyLevelRange = () => {
    const min = lvlMin === '' ? undefined : Number(lvlMin);
    const max = lvlMax === '' ? undefined : Number(lvlMax);
    applyFilters({ levelRange: (min==null && max==null)? undefined : { min, max } });
  };
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
                    <SelectItem value="species">Species*</SelectItem>
                    <SelectItem value="ability">Ability*</SelectItem>
                    <SelectItem value="level">Level*</SelectItem>
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
          {/* Species / Ability / Gender / Level filters */}
          <fieldset className="space-y-3" aria-labelledby="gf-species-label" role="group">
            <div id="gf-species-label" className="text-[10px] font-medium tracking-wide uppercase text-muted-foreground">Pokemon Filters</div>
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-2" aria-label="Species filter selector">
                <div className="flex items-center gap-2">
                  <Label id="lbl-filter-species" className="text-[11px]" htmlFor="filter-species">Species</Label>
                  <Select value="" onValueChange={v=> { const id = Number(v); if (id>0) addSpecies(id); }}>
                    <SelectTrigger id="filter-species" className="h-8 w-44" aria-labelledby="lbl-filter-species filter-species">
                      <SelectValue placeholder="Add species" />
                    </SelectTrigger>
                    <SelectContent className="max-h-72">
                      {speciesSelectItems.map(item => (
                        <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedSpeciesIds.length>0 && (
                    <Button type="button" size="sm" variant="secondary" onClick={()=> applyFilters({ speciesIds: undefined, abilityIndices: undefined, genders: undefined })}>Clear</Button>
                  )}
                </div>
                <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto border rounded p-1 bg-muted/30" aria-label="Selected species list">
                  {selectedSpeciesIds.length === 0 && <span className="text-[10px] text-muted-foreground">none</span>}
                  {selectedSpeciesIds.map(id => {
                    const s = getGeneratedSpeciesById(id);
                    return (
                      <button key={id} type="button" onClick={()=>removeSpecies(id)} className="text-[10px] px-1 py-[2px] rounded bg-secondary hover:bg-secondary/70" aria-label={`Remove ${s?.names.ja || id}`}>{s?.names.ja || id} ×</button>
                    );
                  })}
                </div>
              </div>
              {/* Ability & Gender shown only when species selected */}
              {selectedSpeciesIds.length>0 && (
                <div className="flex flex-wrap gap-4">
                  <div className="flex items-center gap-2" aria-label="Ability indices filter">
                    {availableAbilityIndices.map(idx => (
                      <label key={idx} className="flex items-center gap-1 text-[11px] cursor-pointer">
                        <Checkbox checked={Boolean(filters.abilityIndices?.includes(idx))} onCheckedChange={()=>toggleAbilityIndex(idx)} />
                        <span>{idx===0?'通常1': idx===1?'通常2':'隠れ'}</span>
                      </label>
                    ))}
                  </div>
                  <div className="flex items-center gap-2" aria-label="Gender filter">
                    {genderOptions.map(g => (
                      <label key={g} className="flex items-center gap-1 text-[11px] cursor-pointer">
                        <Checkbox checked={Boolean(filters.genders?.includes(g))} onCheckedChange={()=>toggleGender(g)} />
                        <span>{g==='N'? '性別不明':'性別'+g}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
              {/* Level range always visible */}
              <div className="flex items-center gap-1" aria-label="Level range filter">
                <Label htmlFor="lvl-min" className="sr-only">Level minimum</Label>
                <Input id="lvl-min" value={lvlMin} onChange={e=>setLvlMin(e.target.value)} placeholder="Lv min" className="h-8 w-20" inputMode="numeric" />
                <Label htmlFor="lvl-max" className="sr-only">Level maximum</Label>
                <Input id="lvl-max" value={lvlMax} onChange={e=>setLvlMax(e.target.value)} placeholder="Lv max" className="h-8 w-20" inputMode="numeric" />
                <Button type="button" size="sm" variant="secondary" onClick={applyLevelRange}>Set</Button>
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
