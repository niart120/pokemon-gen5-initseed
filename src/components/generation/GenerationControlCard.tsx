import React, { useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/app-store';

export const GenerationControlCard: React.FC = () => {
  const {
    validateDraft, validationErrors, startGeneration, pauseGeneration, resumeGeneration, stopGeneration,
    status, lastCompletion
  } = useAppStore();

  const onStart = useCallback(async () => {
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  }, [validateDraft, validationErrors, startGeneration]);

  return (
    <Card className="p-3 flex flex-col gap-2">
      <CardHeader className="py-0">
        <CardTitle className="text-sm">Generation Control</CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex flex-col gap-2">
        {validationErrors.length > 0 && (
          <div className="text-red-500 text-xs space-y-0.5" role="alert" aria-live="polite">
            {validationErrors.map((e,i)=>(<div key={i}>{e}</div>))}
          </div>
        )}
        <div className="flex items-center gap-3 flex-wrap">
          <button onClick={onStart} disabled={status==='running'||status==='paused'||status==='starting'} className="px-3 py-1 text-xs rounded bg-green-600 text-white disabled:opacity-50">Start</button>
          <button onClick={pauseGeneration} disabled={status!=='running'} className="px-3 py-1 text-xs rounded bg-yellow-600 text-white disabled:opacity-50">Pause</button>
            <button onClick={resumeGeneration} disabled={status!=='paused'} className="px-3 py-1 text-xs rounded bg-blue-600 text-white disabled:opacity-50">Resume</button>
          <button onClick={stopGeneration} disabled={(status!=='running' && status!=='paused')} className="px-3 py-1 text-xs rounded bg-red-600 text-white disabled:opacity-50">Stop</button>
          <div className="text-xs text-muted-foreground">Status: {status}{lastCompletion?` (${lastCompletion.reason})`:''}</div>
        </div>
      </CardContent>
    </Card>
  );
};
