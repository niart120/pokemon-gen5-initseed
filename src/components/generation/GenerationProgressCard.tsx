import React from 'react';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { StandardCardHeader, StandardCardContent, MetricsGrid } from '@/components/ui/card-helpers';
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
  const progress = useAppStore((s) => s.progress);
  const draftParams = useAppStore((s) => s.draftParams);
  const status = useAppStore((s) => s.status);
  const resultsCount = useAppStore((s) => s.results.length);
  const throughput = useAppStore(selectThroughputEma);
  const eta = useAppStore(selectEtaFormatted);
  const shinyCount = useAppStore(selectShinyCount);

  const total = progress?.totalAdvances ?? draftParams.maxAdvances ?? 0;
  const done = progress?.processedAdvances ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const statusText = statusLabel[status] || status;

  return (
    <Card className="py-2 flex flex-col gap-2" aria-labelledby="gen-progress-title" role="region">
      <StandardCardHeader title={<span id="gen-progress-title">Generation Progress</span>} />
      <StandardCardContent>
        <div>
          <Badge variant="outline" className="text-xs" aria-label="Current status">{statusText}</Badge>
        </div>
  <Progress value={pct} className="h-2" aria-label="Overall progress" />
        <MetricsGrid
          items={[
            { label: 'Advances', value: `${done}/${total}` },
            { label: 'Results', value: resultsCount },
            { label: 'Shiny', value: shinyCount },
            { label: 'Throughput', value: throughput ? `${throughput.toFixed(1)} adv/s` : '--' },
            { label: 'ETA', value: eta ?? '--:--' },
            { label: 'Progress', value: pct ? `${pct.toFixed(1)}%` : '0%' },
          ]}
        />
        <div className="sr-only" aria-live="polite">
          {statusText}. {done} of {total} advances. {shinyCount} shiny. ETA {eta ?? 'unknown'}.
        </div>
      </StandardCardContent>
    </Card>
  );
};
