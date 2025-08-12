import React, { useCallback } from 'react';
import type { GenerationParamsHex } from '@/types/generation';
import { useAppStore } from '@/store/app-store';
import { selectThroughputEma, selectEtaFormatted, selectShinyCount } from '@/store/generation-store';

function NumberInput({ label, value, onChange, min, max }: { label: string; value: number; onChange: (v:number)=>void; min?: number; max?: number }) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium">
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
}

function HexInput({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v:string)=>void; placeholder?: string }) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium">
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
}

export const GenerationPanel: React.FC = () => {
  const {
    draftParams, setDraftParams, validateDraft, validationErrors,
    startGeneration, pauseGeneration, resumeGeneration, stopGeneration,
    status, results, progress, filters, applyFilters, lastCompletion
  } = useAppStore();
  const hexDraft: Partial<GenerationParamsHex> = draftParams;

  const throughput = selectThroughputEma(useAppStore.getState());
  const eta = selectEtaFormatted(useAppStore.getState());
  const shinyCount = selectShinyCount(useAppStore.getState());

  const disabledInputs = status === 'running' || status === 'paused' || status === 'starting';

  const onStart = useCallback(async () => {
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  }, [validateDraft, validationErrors, startGeneration]);

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      {/* Params Card */}
      <div className="border rounded-lg p-3 flex flex-wrap gap-4">
        {/* Version */}
        <label className="flex flex-col gap-1 text-xs font-medium w-28">
          <span>Version</span>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
      value={draftParams.version ?? 'B'}
            disabled={disabledInputs}
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
        {/* Encounter Type */}
        <label className="flex flex-col gap-1 text-xs font-medium w-40">
          <span>Encounter</span>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
            value={draftParams.encounterType ?? 0}
            disabled={disabledInputs}
            onChange={e=>setDraftParams({ encounterType: Number(e.target.value) })}
          >
            {[
              [0,'Wild'],[1,'DoubleGrass'],[2,'Surf'],[3,'SurfDark'],[4,'FishingOld'],[5,'FishingGood'],[6,'FishingSuper'],[7,'Shaking'],
              [10,'Static'],[11,'Gift'],[12,'Roamer'],[13,'HiddenGrotto'],[20,'Egg']
            ].map(([val,label]) => <option key={val} value={val}>{label}</option>)}
          </select>
        </label>
        {/* Synchronize */}
        <label className="flex items-center gap-2 text-xs font-medium w-full sm:w-auto">
          <input type="checkbox" checked={draftParams.syncEnabled ?? false} disabled={disabledInputs} onChange={e=>setDraftParams({ syncEnabled: e.target.checked })} />
          Sync Enabled
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium w-36">
          <span>Sync Nature</span>
          <select
            className="border rounded px-2 py-1 text-sm bg-background"
            value={draftParams.syncNatureId ?? 0}
            disabled={disabledInputs || !(draftParams.syncEnabled ?? false)}
            onChange={e=>setDraftParams({ syncNatureId: Number(e.target.value) })}
          >
            {Array.from({length:25},(_,i)=>i).map(id=> <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs font-medium">
          <input type="checkbox" checked={draftParams.stopAtFirstShiny ?? false} onChange={e=>setDraftParams({ stopAtFirstShiny: e.target.checked })} />
          Stop at First Shiny
        </label>
        <label className="flex items-center gap-2 text-xs font-medium">
          <input type="checkbox" checked={draftParams.stopOnCap ?? true} onChange={e=>setDraftParams({ stopOnCap: e.target.checked })} />
          Stop On Cap
        </label>
        <label className="flex items-center gap-2 text-xs font-medium">
          <input type="checkbox" checked={filters.shinyOnly} onChange={e=>applyFilters({ shinyOnly: e.target.checked })} />
          Shiny Only (view)
        </label>
      </div>
      {validationErrors.length > 0 && (
        <div className="text-red-500 text-xs space-y-0.5">
          {validationErrors.map((e,i)=>(<div key={i}>{e}</div>))}
        </div>
      )}

      {/* Action Bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={onStart} disabled={disabledInputs} className="px-3 py-1 text-sm rounded bg-green-600 text-white disabled:opacity-50">Start</button>
        <button onClick={pauseGeneration} disabled={status!=='running'} className="px-3 py-1 text-sm rounded bg-yellow-600 text-white disabled:opacity-50">Pause</button>
        <button onClick={resumeGeneration} disabled={status!=='paused'} className="px-3 py-1 text-sm rounded bg-blue-600 text-white disabled:opacity-50">Resume</button>
        <button onClick={stopGeneration} disabled={(status!=='running' && status!=='paused')} className="px-3 py-1 text-sm rounded bg-red-600 text-white disabled:opacity-50">Stop</button>
        <div className="text-xs text-muted-foreground">Status: {status}{lastCompletion?` (${lastCompletion.reason})`:''}</div>
      </div>

      {/* Progress / Metrics */}
      <div className="text-xs grid grid-cols-2 md:grid-cols-5 gap-2">
        <div>Adv: {progress?.processedAdvances ?? 0}/{progress?.totalAdvances ?? draftParams.maxAdvances}</div>
        <div>Results: {results.length}</div>
        <div>Shiny: {shinyCount}</div>
        <div>Thruput: {throughput?throughput.toFixed(1)+' adv/s':'--'}</div>
        <div>ETA: {eta ?? '--:--'}</div>
      </div>

      {/* Results Table */}
      <div className="flex-1 min-h-0 border rounded-lg overflow-auto">
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
            {results.filter(r=>!filters.shinyOnly || r.shiny_type!==0).map(r=> (
              <tr key={r.advance} className="odd:bg-background even:bg-muted/30">
                <td className="px-2 py-1 font-mono">{r.advance}</td>
                <td className="px-2 py-1 font-mono">0x{(r.pid>>>0).toString(16).padStart(8,'0')}</td>
                <td className="px-2 py-1">{r.nature}</td>
                <td className="px-2 py-1">{r.shiny_type===0?'No':(r.shiny_type===1?'Square':'Star')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
