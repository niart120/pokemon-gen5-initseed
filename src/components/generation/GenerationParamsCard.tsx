import React from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { useAppStore } from '@/store/app-store';
import type { GenerationParamsHex } from '@/types/generation';
import { Gear } from '@phosphor-icons/react';

// Encounter type list (value,label)
const ENCOUNTER_TYPES: [number,string][] = [
  [0,'Wild'],[1,'DoubleGrass'],[2,'Surf'],[3,'SurfDark'],[4,'FishingOld'],[5,'FishingGood'],[6,'FishingSuper'],[7,'Shaking'],
  [10,'Static'],[11,'Gift'],[12,'Roamer'],[13,'HiddenGrotto'],[20,'Egg']
];

// Simple hex normalization guard
function isHexLike(v: string) { return /^(0x)?[0-9a-fA-F]*$/.test(v.trim()); }

export const GenerationParamsCard: React.FC = () => {
  const { draftParams, setDraftParams, status } = useAppStore();
  const disabled = status === 'running' || status === 'paused' || status === 'starting';
  const hexDraft: Partial<GenerationParamsHex> = draftParams;

  const update = (partial: Partial<GenerationParamsHex>) => setDraftParams(partial);

  return (
    <Card className="py-2 flex flex-col gap-2 h-full" aria-labelledby="gen-params-title" role="form">
      <StandardCardHeader icon={<Gear size={20} className="opacity-80" />} title={<span id="gen-params-title">Generation Parameters</span>} />
      <StandardCardContent>
  {/* Basics */}
  <section aria-labelledby="gen-basics" className="space-y-2" role="group">
          <h4 id="gen-basics" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Basics</h4>
          <div className="grid gap-3 grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
            {/* Version */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" id="lbl-version-select" htmlFor="version-select">Version</Label>
              <Select value={draftParams.version ?? 'B'} onValueChange={v=> update({ version: v as GenerationParamsHex['version'] })} disabled={disabled}>
                <SelectTrigger id="version-select" size="sm" aria-labelledby="lbl-version-select version-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {['B','W','B2','W2'].map(v=> <SelectItem key={v} value={v}>{v}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            {/* Base Seed */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" htmlFor="base-seed">Base Seed (hex)</Label>
              <Input id="base-seed" className="font-mono h-8 text-sm" disabled={disabled} value={hexDraft.baseSeedHex ?? '0'}
                onChange={e=> { const v=e.target.value; if (isHexLike(v)) update({ baseSeedHex: v.replace(/^0x/i,'') }); }} placeholder="1a2b3c" />
            </div>
            {/* Offset */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" htmlFor="offset-hex">Offset (hex)</Label>
              <Input id="offset-hex" className="font-mono h-8 text-sm" disabled={disabled} value={hexDraft.offsetHex ?? '0'}
                onChange={e=> { const v=e.target.value; if (isHexLike(v)) update({ offsetHex: v.replace(/^0x/i,'') }); }} placeholder="0" />
            </div>
          </div>
        </section>
        <Separator />
  {/* Limits */}
  <section aria-labelledby="gen-limits" className="space-y-2" role="group">
          <h4 id="gen-limits" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Limits</h4>
          <div className="grid gap-3 grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6">
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="max-adv">Max Advances</Label>
              <Input id="max-adv" type="number" inputMode="numeric" className="h-8 text-sm" disabled={disabled} value={draftParams.maxAdvances ?? 0} onChange={e=> update({ maxAdvances: Number(e.target.value) })} />
            </div>
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="max-results">Max Results</Label>
              <Input id="max-results" type="number" inputMode="numeric" className="h-8 text-sm" disabled={disabled} value={draftParams.maxResults ?? 0} onChange={e=> update({ maxResults: Number(e.target.value) })} />
            </div>
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="batch-size">Batch Size</Label>
              <Input id="batch-size" type="number" inputMode="numeric" className="h-8 text-sm" disabled={disabled} value={draftParams.batchSize ?? 0} onChange={e=> update({ batchSize: Number(e.target.value) })} />
            </div>
          </div>
        </section>
        <Separator />
  {/* Trainer IDs */}
  <section aria-labelledby="gen-trainer" className="space-y-2" role="group">
          <h4 id="gen-trainer" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Trainer IDs</h4>
          <div className="grid gap-3 grid-cols-2 md:grid-cols-4 xl:grid-cols-6 2xl:grid-cols-8">
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="tid">TID</Label>
              <Input id="tid" type="number" inputMode="numeric" className="h-8 text-sm" disabled={disabled} value={draftParams.tid ?? 0} onChange={e=> update({ tid: Number(e.target.value) })} />
            </div>
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="sid">SID</Label>
              <Input id="sid" type="number" inputMode="numeric" className="h-8 text-sm" disabled={disabled} value={draftParams.sid ?? 0} onChange={e=> update({ sid: Number(e.target.value) })} />
            </div>
            <div className="flex items-center gap-2 pt-5">
              <Checkbox id="sync-enabled" aria-labelledby="lbl-sync-enabled" checked={draftParams.syncEnabled ?? false} disabled={disabled} onCheckedChange={v=> update({ syncEnabled: Boolean(v) })} />
              <Label id="lbl-sync-enabled" htmlFor="sync-enabled" className="text-xs">Sync Enabled</Label>
            </div>
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" id="lbl-sync-nature" htmlFor="sync-nature">Sync Nature</Label>
              <Select value={(draftParams.syncNatureId ?? 0).toString()} onValueChange={v=> update({ syncNatureId: Number(v) })} disabled={disabled || !(draftParams.syncEnabled ?? false)}>
                <SelectTrigger id="sync-nature" size="sm" aria-labelledby="lbl-sync-nature sync-nature">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-64">
                  {Array.from({length:25},(_,i)=>i).map(id=> <SelectItem key={id} value={id.toString()}>{id}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>
        <Separator />
  {/* Encounter */}
  <section aria-labelledby="gen-encounter" className="space-y-2" role="group">
          <h4 id="gen-encounter" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Encounter</h4>
          <div className="grid gap-3 grid-cols-2 md:grid-cols-4 xl:grid-cols-6 2xl:grid-cols-8">
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" id="lbl-encounter-type" htmlFor="encounter-type">Type</Label>
              <Select value={(draftParams.encounterType ?? 0).toString()} onValueChange={v=> update({ encounterType: Number(v) })} disabled={disabled}>
                <SelectTrigger id="encounter-type" size="sm" aria-labelledby="lbl-encounter-type encounter-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-72">
                  {ENCOUNTER_TYPES.map(([val,label]) => <SelectItem key={val} value={val.toString()}>{label}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>
        <Separator />
  {/* Stop Conditions */}
  <section aria-labelledby="gen-stop" className="space-y-2" role="group">
          <h4 id="gen-stop" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Stop Conditions</h4>
          <div className="flex flex-wrap gap-6">
            <div className="flex items-center gap-2">
              <Checkbox id="stop-first-shiny" aria-labelledby="lbl-stop-first-shiny" checked={draftParams.stopAtFirstShiny ?? false} disabled={disabled} onCheckedChange={v=> update({ stopAtFirstShiny: Boolean(v) })} />
              <Label id="lbl-stop-first-shiny" htmlFor="stop-first-shiny" className="text-xs">Stop at First Shiny</Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="stop-on-cap" aria-labelledby="lbl-stop-on-cap" checked={draftParams.stopOnCap ?? true} disabled={disabled} onCheckedChange={v=> update({ stopOnCap: Boolean(v) })} />
              <Label id="lbl-stop-on-cap" htmlFor="stop-on-cap" className="text-xs">Stop On Cap</Label>
            </div>
          </div>
        </section>
        <div className="sr-only" aria-live="polite">
          Generation parameters configuration. Editing disabled while generation is active.
        </div>
      </StandardCardContent>
    </Card>
  );
};
