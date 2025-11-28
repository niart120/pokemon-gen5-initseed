import React, { useState } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CaretDown, CaretUp, ChartBar } from '@phosphor-icons/react';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
// import { getResponsiveSizes } from '../../../utils/responsive-sizes';
import {
  // formatElapsedTime,
  // formatRemainingTime,
  formatProcessingRate,
  // calculateOverallProcessingRate,
  // calculateWorkerProcessingRate
} from '../../../lib/utils/format-helpers';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatSearchProgressCount,
  formatSearchProgressPercent,
  formatSearchProgressWorkerBadge,
  formatSearchProgressWorkerCompletion,
  formatSearchProgressWorkerFooter,
  formatSearchProgressWorkerMatches,
  formatSearchProgressWorkerOverview,
  formatSearchProgressWorkerSummary,
  formatSearchProgressWorkerTotal,
  getSearchProgressWorkerStatusLabel,
  searchProgressMatchesLabel,
  searchProgressProgressLabel,
  searchProgressReadyMessage,
  searchProgressTitle,
  searchProgressWorkerListLabel,
  searchProgressWorkerToggleLabel,
} from '@/lib/i18n/strings/search-progress';
import { TimeDisplay } from './TimeDisplay';

export function SearchProgressCard() {
  const { searchProgress, parallelProgress, searchExecutionMode } = useAppStore();
  const [isWorkerDetailsExpanded, setIsWorkerDetailsExpanded] = useState(true);
  const { isStack: isMobile } = useResponsiveLayout();
  // const sizes = getResponsiveSizes(uiScale);
  const locale = useLocale();

  const isParallelMode = searchExecutionMode === 'cpu-parallel' || searchExecutionMode === 'cpu-parallel-new';
  const isRunning = searchProgress.isRunning;
  const parallelData = parallelProgress ?? null;

  const baseCurrentStep = isParallelMode ? parallelData?.totalCurrentStep ?? 0 : searchProgress.currentStep;
  const baseTotalSteps = isParallelMode ? parallelData?.totalSteps ?? 0 : searchProgress.totalSteps;
  const baseElapsedTime = isParallelMode ? parallelData?.totalElapsedTime ?? 0 : searchProgress.elapsedTime;
  const baseEstimatedTimeRemaining = isParallelMode
    ? parallelData?.totalEstimatedTimeRemaining ?? 0
    : searchProgress.estimatedTimeRemaining;
  // 進捗バー用：秒数ベースの進捗パーセントを使用
  const progressPercentForBar = isParallelMode ? parallelData?.progressPercent ?? 0 : 0;
  // 処理速度計算用：処理済み秒数を使用
  const processedSecondsForSpeed = isParallelMode ? parallelData?.totalProcessedSeconds : undefined;

  const hasParallelProgress = isParallelMode && parallelData !== null;
  const hasNonParallelProgress = !isParallelMode
    && (searchProgress.totalSteps > 0 || searchProgress.currentStep > 0 || searchProgress.elapsedTime > 0);
  const hasAnyProgress = hasParallelProgress || hasNonParallelProgress;
  const showBaseProgress = isRunning || hasAnyProgress;
  const showReadyState = !isRunning && !hasAnyProgress;

  // 進捗バー：並列モードは秒数ベースのprogressPercent、それ以外はステップベース
  const progressPercent = isParallelMode
    ? progressPercentForBar
    : (baseTotalSteps > 0 ? Math.min(100, (baseCurrentStep / baseTotalSteps) * 100) : 0);
  
  // 起動したWorker総数を基準にする（停止・完了含む）
  const totalWorkerCount = parallelData?.workerProgresses?.size || 0;

  // ワーカー数に応じたレイアウト決定（モバイル考慮）
  const getWorkerLayout = (count: number) => {
    // モバイル表示時は最大2列に制限し、多数の場合は簡略化
    if (isMobile) {
      if (count <= 2) return { cols: 2, showProgress: true }; // 2個以下でも2列で表示
      if (count <= 8) return { cols: 2, showProgress: true };
      if (count <= 16) return { cols: 2, showProgress: true };
      if (count <= 32) return { cols: 2, showProgress: true };
      return { cols: 2, showProgress: false }; // 32超過は簡略化
    }
    
    // デスクトップ表示（元の設定）
    if (count <= 4) return { cols: 2, showProgress: true };
    if (count <= 8) return { cols: 2, showProgress: true };
    if (count <= 16) return { cols: 3, showProgress: true };
    if (count <= 32) return { cols: 4, showProgress: true };
    return { cols: 4, showProgress: false }; // 32超過は簡略化
  };

  const workerLayout = getWorkerLayout(totalWorkerCount);
  const title = resolveLocaleValue(searchProgressTitle, locale);
  const progressLabel = resolveLocaleValue(searchProgressProgressLabel, locale);
  const matchesLabel = resolveLocaleValue(searchProgressMatchesLabel, locale);
  const readyMessage = resolveLocaleValue(searchProgressReadyMessage, locale);
  const workerListLabel = resolveLocaleValue(searchProgressWorkerListLabel, locale);
  const toggleLabel = resolveLocaleValue(searchProgressWorkerToggleLabel, locale);
  const workerValues = parallelData ? Array.from(parallelData.workerProgresses.values()) : [];
  const runningWorkers = workerValues.filter(p => p.status === 'running').length;
  const completedWorkers = workerValues.filter(p => p.status === 'completed').length;
  const aggregateMatches = workerValues.reduce((sum, p) => sum + p.matchesFound, 0);

  return (
    <PanelCard
      icon={<ChartBar size={20} className="opacity-80" />}
      title={title}
      headerActions={
        isParallelMode && parallelData && totalWorkerCount > 0 ? (
          <Badge variant="outline" className="text-xs">
            {formatSearchProgressWorkerBadge(totalWorkerCount, locale)}
          </Badge>
        ) : undefined
      }
      className={isMobile ? 'max-h-96' : 'min-h-80'}
      fullHeight={!isMobile}
    >
        {/* 基本進捗表示 - 実行中・完了後も表示 */}
        {showBaseProgress && (
          <div className="space-y-2 flex-shrink-0">
            <Progress value={progressPercent} />
            
            {/* 時間表示 - 直列・並列共通 */}
            <TimeDisplay
              elapsedTime={baseElapsedTime}
              estimatedTimeRemaining={baseEstimatedTimeRemaining}
              currentStep={baseCurrentStep}
              totalSteps={baseTotalSteps}
              processedSeconds={processedSecondsForSpeed}
            />
            
            {/* 進捗・マッチ情報 - 直列時のみ表示 */}
            {!isParallelMode && (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <div className="text-muted-foreground">{progressLabel}</div>
                  <div className="font-mono text-sm">
                    {formatSearchProgressCount(searchProgress.currentStep, locale)} / {formatSearchProgressCount(searchProgress.totalSteps, locale)}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {formatSearchProgressPercent(
                      searchProgress.totalSteps > 0
                        ? (searchProgress.currentStep / searchProgress.totalSteps) * 100
                        : 0,
                      locale,
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">{matchesLabel}</div>
                  <Badge variant={searchProgress.matchesFound > 0 ? 'default' : 'secondary'} className="text-sm">
                    {formatSearchProgressCount(searchProgress.matchesFound, locale)}
                  </Badge>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 検索未実行時のメッセージ */}
        {showReadyState && (
          <div className="text-center py-4 text-muted-foreground text-sm flex-shrink-0">
            {readyMessage}
          </div>
        )}

        {/* 並列検索ワーカー情報 - 残りの領域を全て使用 */}
        {isParallelMode && parallelData && totalWorkerCount > 0 && (
          <div className="flex-1 flex flex-col min-h-0">
            {/* ワーカー統計情報 */}
            <div className="text-xs text-muted-foreground flex justify-between flex-shrink-0">
              <span>{formatSearchProgressWorkerSummary(parallelData.activeWorkers, parallelData.completedWorkers, locale)}</span>
              <span>{formatSearchProgressWorkerTotal(totalWorkerCount, locale)}</span>
            </div>
            
            {/* ワーカー詳細表示 - 残りのスペースを全て使用 */}
            <div className="flex-1 flex flex-col min-h-0 mt-2">
              <div className="flex items-center justify-between flex-shrink-0">
                <div className="text-xs text-muted-foreground">{workerListLabel}</div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsWorkerDetailsExpanded(!isWorkerDetailsExpanded)}
                  className="h-6 w-6 p-0"
                  aria-label={toggleLabel}
                >
                  {isWorkerDetailsExpanded ? (
                    <CaretUp size={12} />
                  ) : (
                    <CaretDown size={12} />
                  )}
                </Button>
              </div>
              
              {isWorkerDetailsExpanded && (
                <div className="flex-1 flex flex-col min-h-0 mt-2">
                  {workerLayout.showProgress ? (
                    // プログレスバー表示：2-4列可変、残りの高さ全体を使用
                    <div className="flex-1 flex flex-col min-h-0">
                      <div className="flex-1 overflow-y-auto pr-1 min-h-0">
                        <div className={`grid gap-2 ${
                          isMobile 
                            ? 'grid-cols-2' // モバイル: 2列固定
                            : workerLayout.cols === 2 ? 'grid-cols-2' :
                              workerLayout.cols === 3 ? 'grid-cols-3' :
                              'grid-cols-4'
                        }`}>
                          {Array.from(parallelData.workerProgresses.entries()).map(([workerId, progress]) => {
                            // progressPercentを使用（秒数ベース）、なければcurrentStep/totalStepsにフォールバック
                            const workerPercent = progress.progressPercent ?? (progress.totalSteps > 0 ? (progress.currentStep / progress.totalSteps) * 100 : 0);
                            const clampedPercent = progress.status === 'completed' ? 100 : Math.min(100, workerPercent);
                            return (
                              <div
                                key={workerId}
                                className="p-2 rounded border bg-muted/20 space-y-1.5 min-h-[4rem] min-w-[80px] sm:min-w-[80px]"
                              >
                                <div className="flex justify-between items-center text-xs">
                                  <div className="flex items-center gap-1.5">
                                    <span className="font-mono font-medium">W{workerId}</span>
                                    {progress.matchesFound > 0 && (
                                      <span className="px-1 py-0.5 bg-green-100 text-green-800 rounded text-[9px] font-medium">
                                        [{formatSearchProgressCount(progress.matchesFound, locale)}]
                                      </span>
                                    )}
                                  </div>
                                  <span
                                    className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                                      progress.status === 'completed'
                                        ? 'bg-green-100 text-green-800'
                                        : progress.status === 'running'
                                          ? 'bg-blue-100 text-blue-800'
                                          : progress.status === 'error'
                                            ? 'bg-red-100 text-red-800'
                                            : 'bg-gray-100 text-gray-600'
                                    }`}
                                  >
                                    {getSearchProgressWorkerStatusLabel(progress.status, locale)}
                                  </span>
                                </div>
                                <Progress
                                  value={progress.status === 'completed' ? 100 : clampedPercent}
                                  className="h-1.5"
                                />
                                <div className="flex justify-between text-[10px] text-muted-foreground">
                                  <span>
                                    {formatSearchProgressPercent(clampedPercent, locale, 0)}
                                  </span>
                                  <span className="font-mono">
                                    {formatProcessingRate(progress.processedSeconds ?? progress.currentStep, progress.elapsedTime)}
                                  </span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      <div className="text-[10px] text-muted-foreground pt-1 border-t flex-shrink-0">
                        {formatSearchProgressWorkerFooter(runningWorkers, completedWorkers, aggregateMatches, locale)}
                      </div>
                    </div>
                  ) : (
                    // 簡略化表示：大量ワーカー向け
                    <div className="space-y-2 flex-shrink-0">
                      <div className="text-xs text-muted-foreground">{formatSearchProgressWorkerOverview(totalWorkerCount, locale)}</div>
                      <Progress 
                        value={(parallelData.completedWorkers / totalWorkerCount) * 100} 
                        className="h-2"
                      />
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span>{formatSearchProgressWorkerCompletion(parallelData.completedWorkers, totalWorkerCount, locale)}</span>
                        <span>{formatSearchProgressWorkerMatches(aggregateMatches, locale)}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
    </PanelCard>
  );
}
