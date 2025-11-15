import React, { useCallback } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Play, Pause, Square, ChartBar } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { Progress } from '@/components/ui/progress';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatGenerationRunAdvancesDisplay,
  formatGenerationRunPercentDisplay,
  formatGenerationRunScreenReaderSummary,
  formatGenerationRunStatusDisplay,
  generationRunButtonLabels,
  generationRunControlsLabel,
  generationRunPanelTitle,
  generationRunProgressBarLabel,
  generationRunProgressLabel,
  generationRunStatusPrefix,
} from '@/lib/i18n/strings/generation-run';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';

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

  const locale = useLocale();
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

  const title = resolveLocaleValue(generationRunPanelTitle, locale);
  const controlsLabel = resolveLocaleValue(generationRunControlsLabel, locale);
  const progressLabel = resolveLocaleValue(generationRunProgressLabel, locale);
  const progressBarLabel = resolveLocaleValue(generationRunProgressBarLabel, locale);
  const statusPrefix = resolveLocaleValue(generationRunStatusPrefix, locale);
  const startLabel = resolveLocaleValue(generationRunButtonLabels.start, locale);
  const startingLabel = resolveLocaleValue(generationRunButtonLabels.starting, locale);
  const pauseLabel = resolveLocaleValue(generationRunButtonLabels.pause, locale);
  const resumeLabel = resolveLocaleValue(generationRunButtonLabels.resume, locale);
  const stopLabel = resolveLocaleValue(generationRunButtonLabels.stop, locale);

  const statusDisplay = formatGenerationRunStatusDisplay(status, lastCompletion?.reason ?? null, locale);
  const advancesDisplay = formatGenerationRunAdvancesDisplay(done, total, locale);
  const percentDisplay = formatGenerationRunPercentDisplay(pct, locale);
  const screenReaderSummary = formatGenerationRunScreenReaderSummary(statusDisplay, advancesDisplay, percentDisplay, locale);

  return (
    <PanelCard
      icon={<ChartBar size={20} className="opacity-80" />}
      title={<span id="gen-run-title">{title}</span>}
      fullHeight={false}
      scrollMode={isStack ? 'parent' : 'content'}
      contentClassName="gap-3"
      role="region"
      aria-labelledby="gen-run-title"
    >
        {/* Validation Errors */}
        {validationErrors.length > 0 && (
          <div className="text-destructive text-xs space-y-0.5" role="alert" aria-live="polite">
            {validationErrors.map((e, i) => (
              <div key={i}>{e}</div>
            ))}
          </div>
        )}
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap" role="group" aria-label={controlsLabel}>
          {canStart && (
            <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1 min-w-[120px]" data-testid="gen-start-btn">
              <Play size={16} className="mr-2" />
              {isStarting ? startingLabel : startLabel}
            </Button>
          )}
          {isRunning && (
            <>
              <Button size="sm" variant="secondary" onClick={pauseGeneration} className="flex-1 min-w-[110px]" data-testid="gen-pause-btn">
                <Pause size={16} className="mr-2" />
                {pauseLabel}
              </Button>
              <Button size="sm" variant="destructive" onClick={stopGeneration} data-testid="gen-stop-btn">
                <Square size={16} className="mr-2" />
                {stopLabel}
              </Button>
            </>
          )}
          {isPaused && (
            <>
              <Button size="sm" onClick={resumeGeneration} className="flex-1 min-w-[110px]" data-testid="gen-resume-btn">
                <Play size={16} className="mr-2" />
                {resumeLabel}
              </Button>
              <Button size="sm" variant="destructive" onClick={stopGeneration} data-testid="gen-stop-btn">
                <Square size={16} className="mr-2" />
                {stopLabel}
              </Button>
            </>
          )}
          <div className="text-xs text-muted-foreground ml-auto" aria-live="polite">
            {statusPrefix} {statusDisplay}
          </div>
        </div>
        {/* Progress */}
        <div className="space-y-1" aria-label={progressLabel}>
          <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
            <span>{percentDisplay}</span>
            <span>{advancesDisplay}</span>
          </div>
          <Progress value={Math.max(0, Math.min(100, pct))} aria-label={progressBarLabel} />
        </div>
        <div className="sr-only" aria-live="polite">
          {screenReaderSummary}
        </div>
    </PanelCard>
  );
};
