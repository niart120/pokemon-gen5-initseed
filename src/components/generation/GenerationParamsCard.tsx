import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import type { GenerationParamsHex } from '@/types/generation';
import { useAppStore } from '@/store/app-store';

// TODO: 共通化候補 (Search 側と似たスタイルに揃える)
interface NumberInputProps { label: string; value: number; onChange: (v:number)=>void; min?: number; max?: number; }
const NumberInput: React.FC<NumberInputProps> = ({ label, value, onChange, min, max }) => (
  <label className="flex flex-col gap-1 text-xs font-medium w-28">
    <span>{label}</span>
    <input
      type="number"
      className="border rounded px-2 py-1 text-sm bg-background"
      value={value}
      min={min}
      max={max}
      onChange={e => onChange(Number(e.target.value))}
    />
  </label>
);

interface HexInputProps { label: string; value: string; onChange: (v:string)=>void; placeholder?: string }
const HexInput: React.FC<HexInputProps> = ({ label, value, onChange, placeholder }) => (
  <label className="flex flex-col gap-1 text-xs font-medium w-36">
    <span>{label}</span>
    <input
      type="text"
      className="border rounded px-2 py-1 text-sm bg-background font-mono"
      value={value}
      placeholder={placeholder}
      onChange={e => {
        const txt = e.target.value.trim();
        if (/^(0x)?[0-9a-fA-F]*$/.test(txt)) onChange(txt.replace(/^0x/i,''));
      }}
    />
  </label>
);

export const GenerationParamsCard: React.FC = () => {
  const { draftParams, setDraftParams, status } = useAppStore();
  const disabled = status === 'running' || status === 'paused' || status === 'starting';
  const hexDraft: Partial<GenerationParamsHex> = draftParams;

  return (
    <Card className="p-3 flex flex-col gap-3">
      <CardHeader className="py-0">
        <CardTitle className="text-sm flex items-center gap-2">Generation Parameters</CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs font-medium w-28">
          <span>Version</span>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
            value={draftParams.version ?? 'B'}
            disabled={disabled}
            onChange={(e)=>setDraftParams({ version: e.target.value as GenerationParamsHex['version'] })}
          >
            {['B','W','B2','W2'].map(v=> <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <HexInput label="Base Seed (hex)" value={hexDraft.baseSeedHex ?? '0'} onChange={v=>setDraftParams({ baseSeedHex: v })} />
        <HexInput label="Offset" value={hexDraft.offsetHex ?? '0'} onChange={v=>setDraftParams({ offsetHex: v })} />
        <NumberInput label="Max Advances" value={draftParams.maxAdvances ?? 0} onChange={v=>setDraftParams({ maxAdvances: v })} />
        <NumberInput label="Max Results" value={draftParams.maxResults ?? 0} onChange={v=>setDraftParams({ maxResults: v })} />
        <NumberInput label="Batch Size" value={draftParams.batchSize ?? 0} onChange={v=>setDraftParams({ batchSize: v })} />
        <NumberInput label="TID" value={draftParams.tid ?? 0} onChange={v=>setDraftParams({ tid: v })} />
        <NumberInput label="SID" value={draftParams.sid ?? 0} onChange={v=>setDraftParams({ sid: v })} />
        <label className="flex flex-col gap-1 text-xs font-medium w-40">
          <span>Encounter</span>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
            value={draftParams.encounterType ?? 0}
            disabled={disabled}
            onChange={e=>setDraftParams({ encounterType: Number(e.target.value) })}
          >
            {[ [0,'Wild'],[1,'DoubleGrass'],[2,'Surf'],[3,'SurfDark'],[4,'FishingOld'],[5,'FishingGood'],[6,'FishingSuper'],[7,'Shaking'], [10,'Static'],[11,'Gift'],[12,'Roamer'],[13,'HiddenGrotto'],[20,'Egg'] ].map(([val,label]) => <option key={val} value={val}>{label}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs font-medium w-full sm:w-auto">
          <input type="checkbox" checked={draftParams.syncEnabled ?? false} disabled={disabled} onChange={e=>setDraftParams({ syncEnabled: e.target.checked })} />
          Sync Enabled
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium w-36">
          <span>Sync Nature</span>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
            value={draftParams.syncNatureId ?? 0}
            disabled={disabled || !(draftParams.syncEnabled ?? false)}
            onChange={e=>setDraftParams({ syncNatureId: Number(e.target.value) })}
          >
            {Array.from({length:25},(_,i)=>i).map(id=> <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs font-medium">
          <input type="checkbox" checked={draftParams.stopAtFirstShiny ?? false} disabled={disabled} onChange={e=>setDraftParams({ stopAtFirstShiny: e.target.checked })} />
          Stop at First Shiny
        </label>
        <label className="flex items-center gap-2 text-xs font-medium">
          <input type="checkbox" checked={draftParams.stopOnCap ?? true} disabled={disabled} onChange={e=>setDraftParams({ stopOnCap: e.target.checked })} />
          Stop On Cap
        </label>
      </CardContent>
    </Card>
  );
};
