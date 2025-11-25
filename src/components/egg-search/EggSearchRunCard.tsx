/**
 * EggSearchRunCard
 * 検索実行制御カード
 */

import React from 'react';
import { Play, Stop, Clock } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggSearchRunCardTitle,
  eggSearchStatusPrefix,
  eggSearchFoundLabel,
  eggSearchElapsedLabel,
  eggSearchProgressLabel,
  eggSearchButtonLabels,
  getEggSearchStatusLabel,
  formatEggSearchElapsed,
} from '@/lib/i18n/strings/egg-search';

export function EggSearchRunCard() {
  const locale = useLocale();
  const {
    status,
    progress,
    startSearch,
    stopSearch,
    results,
    lastElapsedMs,
  } = useEggBootTimingSearchStore();

  const isRunning = status === 'running' || status === 'starting';
  const isStopping = status === 'stopping';
  const canStart = status === 'idle' || status === 'completed' || status === 'error';

  const handleStart = async () => {
    await startSearch();
  };

  const handleStop = () => {
    stopSearch();
  };

  return (
    <PanelCard
      icon={<Clock size={20} className="opacity-80" />}
      title={eggSearchRunCardTitle[locale]}
    >
      <div className="space-y-4">
        {/* ステータス表示 */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{eggSearchStatusPrefix[locale]}:</span>
          <Badge variant={status === 'error' ? 'destructive' : 'secondary'}>
            {getEggSearchStatusLabel(status, locale)}
          </Badge>
        </div>

        {/* 進捗バー */}
        {progress && isRunning && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>{eggSearchProgressLabel[locale]}</span>
              <span>{progress.progressPercent.toFixed(1)}%</span>
            </div>
            <Progress value={progress.progressPercent} className="h-2" />
          </div>
        )}

        {/* 結果数・経過時間 */}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">{eggSearchFoundLabel[locale]}:</span>
            <span className="font-mono">{progress?.foundCount ?? results.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">{eggSearchElapsedLabel[locale]}:</span>
            <span className="font-mono">
              {progress?.elapsedMs
                ? formatEggSearchElapsed(progress.elapsedMs, locale)
                : lastElapsedMs
                  ? formatEggSearchElapsed(lastElapsedMs, locale)
                  : '--'}
            </span>
          </div>
        </div>

        {/* 開始/停止ボタン */}
        <div className="flex gap-2">
          {canStart ? (
            <Button
              onClick={handleStart}
              className="flex-1 gap-2"
              disabled={isStopping}
            >
              <Play size={16} weight="fill" />
              {eggSearchButtonLabels.start[locale]}
            </Button>
          ) : (
            <Button
              onClick={handleStop}
              variant="destructive"
              className="flex-1 gap-2"
              disabled={isStopping}
            >
              <Stop size={16} weight="fill" />
              {isStopping ? eggSearchButtonLabels.stopping[locale] : eggSearchButtonLabels.stop[locale]}
            </Button>
          )}
        </div>
      </div>
    </PanelCard>
  );
}
