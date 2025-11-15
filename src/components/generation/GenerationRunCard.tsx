import React, { useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Button } from '@/components/ui/button';
import { Play, Pause, Square, ChartBar } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { Progress } from '@/components/ui/progress';

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
  } = useAppStore();

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
    <Card className="flex flex-col" role="region" aria-labelledby="gen-run-title">
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
        {/* Progress */}
        <div className="space-y-1" aria-label="Generation progress">
          <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
            <span>{pct ? `${pct.toFixed(1)}%` : '0.0%'}</span>
            <span>{done}/{total} adv</span>
          </div>
          <Progress value={Math.max(0, Math.min(100, pct))} aria-label="Generation progress bar" />
        </div>
        <div className="sr-only" aria-live="polite">
          {status}. {done} of {total} advances. {pct ? pct.toFixed(1) : '0.0'} percent complete.
        </div>
      </StandardCardContent>
    </Card>
  );
};
