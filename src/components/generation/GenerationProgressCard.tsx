import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/app-store';
import { selectThroughputEma, selectEtaFormatted, selectShinyCount } from '@/store/generation-store';

// 状態表示用テキストマッピング
const statusLabel: Record<string, string> = {
  idle: 'Idle',
  starting: 'Starting…',
  running: 'Running',
  paused: 'Paused',
  stopping: 'Stopping…',
  completed: 'Completed',
  error: 'Error',
};

export const GenerationProgressCard: React.FC = () => {
  const progress = useAppStore(s => s.progress);
  const draftParams = useAppStore(s => s.draftParams);
  const status = useAppStore(s => s.status);
  const resultsCount = useAppStore(s => s.results.length);
  const throughput = useAppStore(selectThroughputEma);
  const eta = useAppStore(selectEtaFormatted);
  const shinyCount = useAppStore(selectShinyCount);

  const total = progress?.totalAdvances ?? draftParams.maxAdvances ?? 0;
  const done = progress?.processedAdvances ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const statusText = statusLabel[status] || status;

  return (
    <Card className="p-3 flex flex-col gap-2" aria-label="Generation progress and metrics">
      <CardHeader className="py-0"><CardTitle className="text-sm flex justify-between">
        <span>Progress / Metrics</span>
        <span className="text-[10px] font-normal px-1 py-0.5 rounded border bg-muted/30" aria-label="Current status">{statusText}</span>
      </CardTitle></CardHeader>
      <CardContent className="p-0 text-xs flex flex-col gap-2">
        {/* 進捗バー */}
        <div className="h-2 w-full bg-muted rounded overflow-hidden" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done} aria-label="Advances progress">
          <div className="h-full bg-blue-500 transition-all" style={{ width: pct + '%' }} />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
          <div>Adv: {done}/{total}</div>
          <div>Results: {resultsCount}</div>
          <div>Shiny: {shinyCount}</div>
            <div>Thruput: {throughput ? throughput.toFixed(1) + ' adv/s' : '--'}</div>
          <div>ETA: {eta ?? '--:--'}</div>
          <div>{pct ? pct.toFixed(1) + '%' : '0%'}</div>
        </div>
        {/* アクセシビリティ用 live リージョン */}
        <div className="sr-only" aria-live="polite">
          {statusText}. {done} of {total} advances. {shinyCount} shiny. ETA {eta ?? 'unknown'}.
        </div>
      </CardContent>
    </Card>
  );
};
