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

  const handleStart = () => {
    startSearch();
  };

  const handleStop = () => {
    stopSearch();
  };

  const statusLabels: Record<string, Record<typeof status, string>> = {
    ja: {
      idle: 'アイドル',
      starting: '開始中',
      running: '検索中',
      stopping: '停止中',
      completed: '完了',
      error: 'エラー',
    },
    en: {
      idle: 'Idle',
      starting: 'Starting',
      running: 'Searching',
      stopping: 'Stopping',
      completed: 'Completed',
      error: 'Error',
    },
  };

  const labels = {
    title: locale === 'ja' ? '検索制御' : 'Search Control',
    start: locale === 'ja' ? '検索開始' : 'Start Search',
    stop: locale === 'ja' ? '停止' : 'Stop',
    stopping: locale === 'ja' ? '停止中...' : 'Stopping...',
    status: locale === 'ja' ? 'ステータス' : 'Status',
    found: locale === 'ja' ? '発見数' : 'Found',
    elapsed: locale === 'ja' ? '経過時間' : 'Elapsed',
    progress: locale === 'ja' ? '進捗' : 'Progress',
  };

  const formatElapsed = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) {
      return locale === 'ja' ? `${seconds}秒` : `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return locale === 'ja'
      ? `${minutes}分${remainingSeconds}秒`
      : `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <PanelCard
      icon={<Clock size={20} className="opacity-80" />}
      title={labels.title}
    >
      <div className="space-y-4">
        {/* ステータス表示 */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{labels.status}:</span>
          <Badge variant={status === 'error' ? 'destructive' : 'secondary'}>
            {statusLabels[locale]?.[status] ?? status}
          </Badge>
        </div>

        {/* 進捗バー */}
        {progress && isRunning && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>{labels.progress}</span>
              <span>{progress.progressPercent.toFixed(1)}%</span>
            </div>
            <Progress value={progress.progressPercent} className="h-2" />
          </div>
        )}

        {/* 結果数・経過時間 */}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">{labels.found}:</span>
            <span className="font-mono">{progress?.foundCount ?? results.length}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">{labels.elapsed}:</span>
            <span className="font-mono">
              {progress?.elapsedMs
                ? formatElapsed(progress.elapsedMs)
                : lastElapsedMs
                  ? formatElapsed(lastElapsedMs)
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
              {labels.start}
            </Button>
          ) : (
            <Button
              onClick={handleStop}
              variant="destructive"
              className="flex-1 gap-2"
              disabled={isStopping}
            >
              <Stop size={16} weight="fill" />
              {isStopping ? labels.stopping : labels.stop}
            </Button>
          )}
        </div>
      </div>
    </PanelCard>
  );
}
