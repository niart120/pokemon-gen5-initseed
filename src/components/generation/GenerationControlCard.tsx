import React, { useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Button } from '@/components/ui/button';
import { Play, Pause, Square } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';

export const GenerationControlCard: React.FC = () => {
  const {
    validateDraft,
    validationErrors,
    startGeneration,
    pauseGeneration,
    resumeGeneration,
    stopGeneration,
    status,
    lastCompletion
  } = useAppStore();

  const handleStart = useCallback(async () => {
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  }, [validateDraft, validationErrors, startGeneration]);

  const isStarting = status === 'starting';
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  // 再開可能状態: 初期(idle) または 完了(completed) / error 終了後
  const canStart = status === 'idle' || status === 'completed' || status === 'error';
  const { isStack } = useResponsiveLayout();

  return (
    <Card className={`py-2 flex flex-col ${isStack ? 'max-h-96' : 'h-full min-h-64'}`} aria-labelledby="gen-control-title">
      <StandardCardHeader icon={<Play size={20} className="opacity-80" />} title={<span id="gen-control-title">Generation Control</span>} />
      <StandardCardContent>
        {validationErrors.length > 0 && (
          <div
            className="text-destructive text-xs space-y-0.5"
            role="alert"
            aria-live="polite"
            data-testid="gen-validation-errors"
          >
            {validationErrors.map((e, i) => (
              <div key={i}>{e}</div>
            ))}
          </div>
        )}
        <div className="flex items-center gap-2 flex-wrap" role="group" aria-label="Generation execution controls">
          {canStart && (
            <Button
              size="sm"
              onClick={handleStart}
              disabled={isStarting}
              className="flex-1"
              data-testid="gen-start-btn"
            >
              <Play size={16} className="mr-2" />
              {isStarting ? 'Starting…' : 'Start'}
            </Button>
          )}
          {isRunning && (
            <>
              <Button
                size="sm"
                variant="secondary"
                onClick={pauseGeneration}
                className="flex-1"
                data-testid="gen-pause-btn"
              >
                <Pause size={16} className="mr-2" />
                Pause
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={stopGeneration}
                data-testid="gen-stop-btn"
              >
                <Square size={16} className="mr-2" />
                Stop
              </Button>
            </>
          )}
          {isPaused && (
            <>
              <Button
                size="sm"
                onClick={resumeGeneration}
                className="flex-1"
                data-testid="gen-resume-btn"
              >
                <Play size={16} className="mr-2" />
                Resume
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={stopGeneration}
                data-testid="gen-stop-btn"
              >
                <Square size={16} className="mr-2" />
                Stop
              </Button>
            </>
          )}
          <div className="text-xs text-muted-foreground ml-auto" aria-live="polite">
            Status: {status}{lastCompletion ? ` (${lastCompletion.reason})` : ''}
          </div>
        </div>
  </StandardCardContent>
    </Card>
  );
};
