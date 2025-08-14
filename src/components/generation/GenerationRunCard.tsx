import React, { useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent, MetricsGrid } from '@/components/ui/card-helpers';
import { Button } from '@/components/ui/button';
import { Play, Pause, Square, ChartBar } from '@phosphor-icons/react';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/store/app-store';
import { selectThroughputEma, selectEtaFormatted, selectShinyCount } from '@/store/generation-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';

// Control + Progress 統合カード (Phase1 experimental)
export const GenerationRunCard: React.FC = () => {
  const {
    validateDraft,
    validationErrors,
    startGeneration,
    pauseGeneration,
    resumeGeneration,
    stopGeneration,
    status,
    lastCompletion,
    draftParams,
    progress,
    results,
  } = useAppStore();
  const throughput = useAppStore(selectThroughputEma);
  const eta = useAppStore(selectEtaFormatted);
  const shinyCount = useAppStore(selectShinyCount);

  const total = progress?.totalAdvances ?? draftParams.maxAdvances ?? 0;
  const done = progress?.processedAdvances ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;

  const handleStart = useCallback(async () => {
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  }, [validateDraft, validationErrors, startGeneration]);

  const { isStack } = useResponsiveLayout();
  const isStarting = status === 'starting';
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const canStart = status === 'idle' || status === 'completed' || status === 'error';

  return (
    <Card className={`py-2 flex flex-col ${isStack ? '' : 'h-full min-h-64'}`} role="region" aria-labelledby="gen-run-title">
      <StandardCardHeader
        icon={<ChartBar size={20} className="opacity-80" />}
        title={<span id="gen-run-title">Generation Run</span>}
      />
  <StandardCardContent className="gap-3" noScroll={isStack}>
        {/* Validation Errors */}
        {validationErrors.length > 0 && (
          <div className="text-destructive text-xs space-y-0.5" role="alert" aria-live="polite">
            {validationErrors.map((e, i) => (
              <div key={i}>{e}</div>
            ))}
          </div>
        )}
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap" role="group" aria-label="Generation execution controls">
          {canStart && (
            <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1 min-w-[120px]" data-testid="gen-start-btn">
              <Play size={16} className="mr-2" />
              {isStarting ? 'Starting…' : 'Start'}
            </Button>
          )}
          {isRunning && (
            <>
              <Button size="sm" variant="secondary" onClick={pauseGeneration} className="flex-1 min-w-[110px]" data-testid="gen-pause-btn">
                <Pause size={16} className="mr-2" />
                Pause
              </Button>
              <Button size="sm" variant="destructive" onClick={stopGeneration} data-testid="gen-stop-btn">
                <Square size={16} className="mr-2" />
                Stop
              </Button>
            </>
          )}
          {isPaused && (
            <>
              <Button size="sm" onClick={resumeGeneration} className="flex-1 min-w-[110px]" data-testid="gen-resume-btn">
                <Play size={16} className="mr-2" />
                Resume
              </Button>
              <Button size="sm" variant="destructive" onClick={stopGeneration} data-testid="gen-stop-btn">
                <Square size={16} className="mr-2" />
                Stop
              </Button>
            </>
          )}
          <div className="text-xs text-muted-foreground ml-auto" aria-live="polite">
            Status: {status}{lastCompletion ? ` (${lastCompletion.reason})` : ''}
          </div>
        </div>
        {/* Progress Row */}
        <div className="flex items-center justify-between gap-2">
          <Badge variant="outline" className="text-xs" aria-label="Progress percentage">
            {pct ? pct.toFixed(1) : '0.0'}%
          </Badge>
          <div className="text-[11px] text-muted-foreground font-mono">{done}/{total} adv</div>
        </div>
        {/* Metrics */}
        <MetricsGrid
          columns="grid-cols-2 md:grid-cols-5"
          items={[
            { label: 'Results', value: results.length },
            { label: 'Shiny', value: shinyCount },
            { label: 'Throughput', value: throughput ? `${throughput.toFixed(1)} adv/s` : '--' },
            { label: 'ETA', value: eta ?? '--:--' },
            { label: 'Status', value: status },
          ]}
        />
        <div className="sr-only" aria-live="polite">
          {status}. {done} of {total} advances. {shinyCount} shiny. ETA {eta ?? 'unknown'}.
        </div>
      </StandardCardContent>
    </Card>
  );
};
