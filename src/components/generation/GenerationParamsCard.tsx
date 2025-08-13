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
import { DomainEncounterTypeNames, DomainEncounterType } from '@/types/domain';

// Simple hex normalization guard
function isHexLike(v: string) { return /^(0x)?[0-9a-fA-F]*$/.test(v.trim()); }

// Ability モード選択肢 (Compound は WIP のため disabled)
const ABILITY_OPTIONS: { value: GenerationParamsHex['abilityMode']; label: string; disabled?: boolean }[] = [
  { value: 'none', label: '-' },
  { value: 'sync', label: 'Sync' },
  { value: 'compound', label: 'Compound (WIP)', disabled: true },
];

export const GenerationParamsCard: React.FC = () => {
  const { draftParams, setDraftParams, status } = useAppStore();
  const disabled = status === 'running' || status === 'paused' || status === 'starting';
  const hexDraft: Partial<GenerationParamsHex> = draftParams;

  const update = (partial: Partial<GenerationParamsHex>) => setDraftParams(partial);

  const abilityMode = hexDraft.abilityMode ?? 'none';
  const onAbilityChange = (mode: NonNullable<GenerationParamsHex['abilityMode']>) => {
    // syncEnabled 連動: sync 選択時のみ true
    update({ abilityMode: mode, syncEnabled: mode === 'sync' });
  };
  const syncActive = abilityMode === 'sync' && (hexDraft.syncEnabled ?? false);
  const encounterValue = hexDraft.encounterType ?? 0;

  return (
    <Card className="py-2 flex flex-col gap-2 h-full" aria-labelledby="gen-params-title" role="form">
      <StandardCardHeader icon={<Gear size={20} className="opacity-80" />} title={<span id="gen-params-title">Generation Parameters</span>} />
      <StandardCardContent>
  {/* Basics */}
  <section aria-labelledby="gen-basics" className="space-y-2" role="group">
          <h4 id="gen-basics" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Basics</h4>
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
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
            {/* TID */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="tid">TID</Label>
              <Input id="tid" type="number" inputMode="numeric" className="h-9" disabled={disabled} value={draftParams.tid ?? 0} onChange={e=> update({ tid: Number(e.target.value) })} />
            </div>
            {/* SID */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="sid">SID</Label>
              <Input id="sid" type="number" inputMode="numeric" className="h-9" disabled={disabled} value={draftParams.sid ?? 0} onChange={e=> update({ sid: Number(e.target.value) })} />
            </div>
          </div>
          {/* Checkboxes - separate row to prevent overlap */}
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 pt-2">
            {/* Shiny Charm */}
            <div className="flex items-center gap-2">
              <Checkbox id="shiny-charm" aria-labelledby="lbl-shiny-charm" checked={draftParams.shinyCharm ?? false} disabled={disabled || (draftParams.version === 'B' || draftParams.version === 'W')} onCheckedChange={v=> update({ shinyCharm: Boolean(v) })} />
              <Label id="lbl-shiny-charm" htmlFor="shiny-charm" className="text-xs">Shiny Charm</Label>
            </div>
            {/* Memory Link */}
            <div className="flex items-center gap-2">
              <Checkbox id="memory-link" aria-labelledby="lbl-memory-link" checked={draftParams.memoryLink ?? false} disabled={disabled || (draftParams.version === 'B' || draftParams.version === 'W')} onCheckedChange={v=> update({ memoryLink: Boolean(v) })} />
              <Label id="lbl-memory-link" htmlFor="memory-link" className="text-xs">Memory Link</Label>
            </div>
          </div>
        </section>
        <Separator />
  {/* Target (Range) */}
  <section aria-labelledby="gen-target" className="space-y-2" role="group">
          <h4 id="gen-target" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Target</h4>
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
            {/* Base Seed */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" htmlFor="base-seed">Base Seed (hex)</Label>
              <Input id="base-seed" className="font-mono h-9 min-w-40" disabled={disabled} value={hexDraft.baseSeedHex ?? '0'}
                onChange={e=> { const v=e.target.value; if (isHexLike(v)) update({ baseSeedHex: v.replace(/^0x/i,'') }); }} placeholder="1a2b3c4d5e6f7890" />
            </div>
            {/* Min Advance (offset) */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" htmlFor="offset-hex">Min Advance (hex)</Label>
              <Input id="offset-hex" className="font-mono h-9 min-w-32" disabled={disabled} value={hexDraft.offsetHex ?? '0'}
                onChange={e=> { const v=e.target.value; if (isHexLike(v)) update({ offsetHex: v.replace(/^0x/i,'') }); }} placeholder="0" />
            </div>
            {/* Max Advances */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="max-adv">Max Advances</Label>
              <Input id="max-adv" type="number" inputMode="numeric" className="h-9" disabled={disabled} value={draftParams.maxAdvances ?? 0} onChange={e=> update({ maxAdvances: Number(e.target.value) })} />
            </div>
            {/* Max Results */}
            <div className="flex flex-col gap-1">
              <Label className="text-xs" htmlFor="max-results">Max Results</Label>
              <Input id="max-results" type="number" inputMode="numeric" className="h-9" disabled={disabled} value={draftParams.maxResults ?? 0} onChange={e=> update({ maxResults: Number(e.target.value) })} />
            </div>
          </div>
        </section>
        <Separator />
  {/* Encounter & Ability */}
  <section aria-labelledby="gen-encounter" className="space-y-2" role="group">
          <h4 id="gen-encounter" className="text-xs font-medium text-muted-foreground tracking-wide uppercase">Encounter</h4>
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {/* Encounter Type */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" id="lbl-encounter-type" htmlFor="encounter-type">Type</Label>
              <Select value={encounterValue.toString()} onValueChange={v=> update({ encounterType: Number(v) })} disabled={disabled}>
                <SelectTrigger id="encounter-type" aria-labelledby="lbl-encounter-type encounter-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-72">
                  {DomainEncounterTypeNames.map(name => {
                    const val = (DomainEncounterType as Record<string, number>)[name];
                    return <SelectItem key={name} value={val.toString()}>{name}</SelectItem>;
                  })}
                </SelectContent>
              </Select>
            </div>
            {/* Ability Mode */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" id="lbl-ability-mode" htmlFor="ability-mode">Ability</Label>
              <Select value={abilityMode} onValueChange={v=> onAbilityChange(v as NonNullable<GenerationParamsHex['abilityMode']>)} disabled={disabled}>
                <SelectTrigger id="ability-mode" aria-labelledby="lbl-ability-mode ability-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-64">
                  {ABILITY_OPTIONS.map(opt => <SelectItem key={opt.value} value={opt.value!} disabled={opt.disabled}>{opt.label}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            {/* Sync Nature */}
            <div className="flex flex-col gap-1 min-w-0">
              <Label className="text-xs" id="lbl-sync-nature" htmlFor="sync-nature">Sync Nature</Label>
              <Select value={(draftParams.syncNatureId ?? 0).toString()} onValueChange={v=> update({ syncNatureId: Number(v) })} disabled={disabled || !syncActive}>
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
