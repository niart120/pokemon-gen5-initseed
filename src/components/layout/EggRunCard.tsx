import React, { useCallback } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Play, Square, ChartBar } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';

/**
 * EggRunCard
 * タマゴ生成の実行制御カード
 */
export const EggRunCard: React.FC = () => {
  const {
    validateDraft,
    validationErrors,
    startGeneration,
    stopGeneration,
    status,
    lastCompletion,
    results,
    draftParams,
  } = useEggStore();

  const { isStack } = useResponsiveLayout();
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

  return (
    <PanelCard
      icon={<ChartBar size={20} className="opacity-80" />}
      title={<span id="egg-run-title">実行制御</span>}
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
      <div className="flex items-center gap-2 flex-wrap" role="group" aria-label="制御ボタン">
        {canStart && (
          <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1" data-testid="egg-start-btn">
            <Play size={16} className="mr-2" />
            {isStarting ? '開始中...' : '開始'}
          </Button>
        )}
        {(isRunning || isStopping) && (
          <Button size="sm" variant="destructive" onClick={stopGeneration} disabled={isStopping} className="flex-1" data-testid="egg-stop-btn">
            <Square size={16} className="mr-2" />
            {isStopping ? '停止中...' : '停止'}
          </Button>
        )}
      </div>
      {/* Status */}
      <div className="text-xs text-muted-foreground mt-2 space-y-1">
        <div>ステータス: {status}</div>
        <div>結果: {results.length} / {draftParams.count} ({pct.toFixed(1)}%)</div>
        {lastCompletion && (
          <>
            <div>処理済み: {lastCompletion.processedCount}</div>
            <div>フィルター後: {lastCompletion.filteredCount}</div>
            <div>実行時間: {lastCompletion.elapsedMs.toFixed(0)}ms</div>
          </>
        )}
      </div>
    </PanelCard>
  );
};
