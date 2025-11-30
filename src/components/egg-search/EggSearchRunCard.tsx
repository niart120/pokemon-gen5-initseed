/**
 * EggSearchRunCard
 * 検索実行制御カード - GenerationRunCard と同様のレイアウト
 */

import React from 'react';
import { Play, Square, ChartBar, Warning } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggSearchRunCardTitle,
  eggSearchStatusPrefix,
  eggSearchButtonLabels,
  getEggSearchStatusLabel,
  eggSearchControlsLabel,
  eggSearchResultsLabel,
} from '@/lib/i18n/strings/egg-search';
import {
  formatRunProgressPercent,
  formatRunProgressCount,
} from '@/lib/i18n/strings/run-progress';

export function EggSearchRunCard() {
  const locale = useLocale();
  const {
    status,
    progress,
    startSearch,
    stopSearch,
    results,
    errorMessage,
    params,
    draftParams,
  } = useEggBootTimingSearchStore();

  const isStarting = status === 'starting';
  const isRunning = status === 'running';
  const isStopping = status === 'stopping';
  const canStart = status === 'idle' || status === 'completed' || status === 'error';

  const handleStart = async () => {
    await startSearch();
  };

  const handleStop = () => {
    stopSearch();
  };

  // 進捗計算
  const foundCount = progress?.foundCount ?? results.length;
  const maxResults = params?.maxResults ?? draftParams.maxResults ?? 1000;
  const pct = progress?.progressPercent ?? (maxResults > 0 ? (foundCount / maxResults) * 100 : 0);

  const statusDisplay = getEggSearchStatusLabel(status, locale);
  const percentDisplay = formatRunProgressPercent(pct, locale);
  const countDisplay = formatRunProgressCount(foundCount, maxResults, locale);

  return (
    <PanelCard
      icon={<ChartBar size={20} className="opacity-80" />}
      title={<span id="egg-search-run-title">{eggSearchRunCardTitle[locale]}</span>}
      role="region"
      aria-labelledby="egg-search-run-title"
    >
      {/* エラーメッセージ表示 */}
      {status === 'error' && errorMessage && (
        <div className="flex items-start gap-2 p-2 rounded-md bg-destructive/10 border border-destructive/20 text-xs">
          <Warning size={14} className="text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-destructive break-all">{errorMessage}</p>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap" role="group" aria-label={eggSearchControlsLabel[locale]}>
        {canStart && (
          <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1" data-testid="egg-search-start-btn">
            <Play size={16} className="mr-2" />
            {isStarting ? eggSearchButtonLabels.stopping[locale] : eggSearchButtonLabels.start[locale]}
          </Button>
        )}
        {(isRunning || isStopping) && (
          <Button size="sm" variant="destructive" onClick={handleStop} disabled={isStopping} data-testid="egg-search-stop-btn">
            <Square size={16} className="mr-2" />
            {eggSearchButtonLabels.stop[locale]}
          </Button>
        )}
        <div className="text-xs text-muted-foreground ml-auto">
          {eggSearchStatusPrefix[locale]}: {statusDisplay}
        </div>
      </div>

      {/* Result summary - 1行表示: 12.3%  xxx / yyy results */}
      <div className="space-y-1" aria-label={eggSearchResultsLabel[locale]}>
        <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono flex-wrap gap-x-2">
          <span>{percentDisplay}</span>
          <span>{countDisplay}</span>
        </div>
      </div>
    </PanelCard>
  );
}
