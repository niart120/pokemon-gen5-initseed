import React, { useCallback } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Play, Square, ChartBar } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggRunPanelTitle,
  eggRunControlsLabel,
  eggRunStatusPrefix,
  eggRunButtonLabels,
  getEggRunStatusLabel,
  formatEggRunProgress,
} from '@/lib/i18n/strings/egg-run';

/**
 * EggRunCard
 * タマゴ生成の実行制御カード (GenerationRunCard と同等のシンプルな表示)
 */
export const EggRunCard: React.FC = () => {
  const {
    validateDraft,
    validationErrors,
    startGeneration,
    stopGeneration,
    status,
    results,
    draftParams,
  } = useEggStore();

  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
  const isStarting = status === 'starting';
  const isRunning = status === 'running';
  const isStopping = status === 'stopping';

  const handleStart = useCallback(async () => {
    const valid = validateDraft();
    if (valid) {
      await startGeneration();
    }
  }, [startGeneration, validateDraft]);

  const canStart = status === 'idle' || status === 'completed' || status === 'error';

  const pct = draftParams.count > 0 ? (results.length / draftParams.count) * 100 : 0;
  const advancesDisplay = formatEggRunProgress(results.length, draftParams.count, locale);
  const percentDisplay = `${pct.toFixed(1)}%`;

  return (
    <PanelCard
      icon={<ChartBar size={20} className="opacity-80" />}
      title={<span id="egg-run-title">{eggRunPanelTitle[locale]}</span>}
      className={isStack ? 'max-h-96' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      role="region"
      aria-labelledby="egg-run-title"
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
      <div className="flex items-center gap-2 flex-wrap" role="group" aria-label={eggRunControlsLabel[locale]}>
        {canStart && (
          <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1" data-testid="egg-start-btn">
            <Play size={16} className="mr-2" />
            {isStarting ? eggRunButtonLabels.starting[locale] : eggRunButtonLabels.start[locale]}
          </Button>
        )}
        {(isRunning || isStopping) && (
          <Button size="sm" variant="destructive" onClick={stopGeneration} disabled={isStopping} data-testid="egg-stop-btn">
            <Square size={16} className="mr-2" />
            {eggRunButtonLabels.stop[locale]}
          </Button>
        )}
        <div className="text-xs text-muted-foreground ml-auto">
          {eggRunStatusPrefix[locale]} {getEggRunStatusLabel(status, locale)}
        </div>
      </div>
      {/* Result summary */}
      <div className="space-y-1" aria-label="Results">
        <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
          <span>{percentDisplay}</span>
          <span>{advancesDisplay}</span>
        </div>
      </div>
    </PanelCard>
  );
};
