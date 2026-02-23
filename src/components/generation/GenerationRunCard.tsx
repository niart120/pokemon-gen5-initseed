import React, { useCallback } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Play, Square } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  getGenerationRunStatusLabel,
  generationRunButtonLabels,
  generationRunControlsLabel,
  generationRunPanelTitle,
  generationRunResultsLabel,
  generationRunStatusPrefix,
} from '@/lib/i18n/strings/generation-run';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatRunProgressPercent,
  formatRunProgressCount,
} from '@/lib/i18n/strings/run-progress';

// Control + Progress 統合カード (Phase1 experimental)
export const GenerationRunCard: React.FC = () => {
  const {
    validateDraft,
    validationErrors,
    startGeneration,
    stopGeneration,
    status,
    draftParams,
  } = useAppStore();
  const resultCount = useAppStore(s => s.results.length);
  const params = useAppStore(s => s.params);

  const locale = useLocale();
  const maxResults = params?.maxResults ?? draftParams.maxResults ?? 0;
  const pct = maxResults > 0 ? (resultCount / maxResults) * 100 : 0;

  const { isStack } = useResponsiveLayout();
  const isStarting = status === 'starting';
  const isRunning = status === 'running';
  const isStopping = status === 'stopping';

  const handleStart = useCallback(async () => {
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  }, [startGeneration, validateDraft, validationErrors.length]);

  const canStart = status === 'idle' || status === 'completed' || status === 'error';
  const statusPrefix = resolveLocaleValue(generationRunStatusPrefix, locale);
  const startLabel = resolveLocaleValue(generationRunButtonLabels.start, locale);
  const startingLabel = resolveLocaleValue(generationRunButtonLabels.starting, locale);
  const stopLabel = resolveLocaleValue(generationRunButtonLabels.stop, locale);
  const title = resolveLocaleValue(generationRunPanelTitle, locale);
  const controlsLabel = resolveLocaleValue(generationRunControlsLabel, locale);
  const resultsLabel: string = resolveLocaleValue(generationRunResultsLabel, locale);

  const statusDisplay = getGenerationRunStatusLabel(status, locale);
  const percentDisplay = formatRunProgressPercent(pct, locale);
  const countDisplay = formatRunProgressCount(resultCount, maxResults, locale);

  return (
    <>
      <PanelCard
        icon={<Play size={20} className="opacity-80" />}
        title={<span id="gen-run-title">{title}</span>}
        className={isStack ? 'max-h-96' : undefined}
        fullHeight={!isStack}
        scrollMode={isStack ? 'parent' : 'content'}
        role="region"
        aria-labelledby="gen-run-title"
      >
        {/* Validation Errors */}
        {validationErrors.length > 0 && (
          <div className="text-destructive text-xs space-y-0.5" role="alert">
            {validationErrors.map((e, i) => (
              <div key={i}>{e}</div>
            ))}
          </div>
        )}
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap" role="group" aria-label={controlsLabel}>
          {canStart && (
            <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1" data-testid="gen-start-btn">
              <Play size={16} className="mr-2" />
              {isStarting ? startingLabel : startLabel}
            </Button>
          )}
          {(isRunning || isStopping) && (
            <Button size="sm" variant="destructive" onClick={stopGeneration} disabled={isStopping} data-testid="gen-stop-btn">
              <Square size={16} className="mr-2" />
              {stopLabel}
            </Button>
          )}
          <div className="text-xs text-muted-foreground ml-auto">
            {statusPrefix}: {statusDisplay}
          </div>
        </div>
        {/* Result summary - 1行表示: 12.3%  xxx / yyy results */}
        <div className="space-y-1" aria-label={resultsLabel}>
          <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono flex-wrap gap-x-2">
            <span>{percentDisplay}</span>
            <span>{countDisplay}</span>
          </div>
        </div>
      </PanelCard>

    </>
  );
};
