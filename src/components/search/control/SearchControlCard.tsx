import React, { useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { Play, Pause, Square } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { getSearchWorkerManager, resetSearchWorkerManager } from '../../../lib/search/search-worker-manager';
import { isWebGpuSupported } from '@/lib/search/search-mode';
import { isWakeLockSupported, requestWakeLock, releaseWakeLock, setupAutoWakeLockManagement } from '@/lib/utils/wake-lock';
import type { InitialSeedResult } from '../../../types/search';
import type { SearchExecutionMode } from '@/store/app-store';

export function SearchControlCard() {
  const { isStack } = useResponsiveLayout();
  const {
    searchConditions,
    searchProgress,
    startSearch,
    pauseSearch,
    resumeSearch,
    stopSearch,
    targetSeeds,
    addSearchResult,
    clearSearchResults,
    parallelSearchSettings,
    setMaxWorkers,
    setParallelProgress,
    wakeLockEnabled,
    setWakeLockEnabled,
    searchExecutionMode,
    setSearchExecutionMode,
  } = useAppStore();

  // ワーカー数設定を初期化時に同期
  useEffect(() => {
    const workerManager = getSearchWorkerManager();
    workerManager.setMaxWorkers(parallelSearchSettings.maxWorkers);
    workerManager.setParallelMode(parallelSearchSettings.enabled);
  }, [parallelSearchSettings.maxWorkers, parallelSearchSettings.enabled]);

  // Wake Lock自動管理のセットアップ
  useEffect(() => {
    if (isWakeLockSupported()) {
      // 検索実行中または一時停止中はWake Lockを維持
      // 一時停止中も維持する理由：
      // - ユーザーがすぐに再開する可能性が高い
      // - 画面が暗くなると再開ボタンを押すためにタップが必要
      // - 長時間検索の途中での短時間一時停止では画面を維持したい
      setupAutoWakeLockManagement(() => 
        wakeLockEnabled && (searchProgress.isRunning || searchProgress.isPaused)
      );
    }
  }, [wakeLockEnabled, searchProgress.isRunning, searchProgress.isPaused]);

  // Wake Lock状態管理: 検索開始/一時停止/終了時に制御
  useEffect(() => {
    if (wakeLockEnabled && (searchProgress.isRunning || searchProgress.isPaused)) {
      // 検索実行中または一時停止中はWake Lockを維持
      requestWakeLock();
    } else if (!searchProgress.isRunning && !searchProgress.isPaused) {
      // 検索が完全に終了した場合のみWake Lockを解除
      releaseWakeLock();
    }
  }, [wakeLockEnabled, searchProgress.isRunning, searchProgress.isPaused]);

  // Worker management functions
  const handlePauseSearch = () => {
    pauseSearch();
    const workerManager = getSearchWorkerManager();
    workerManager.pauseSearch();
  };

  const handleResumeSearch = () => {
    resumeSearch();
    const workerManager = getSearchWorkerManager();
    workerManager.resumeSearch();
  };

  const handleStopSearch = () => {
    stopSearch();
    const workerManager = getSearchWorkerManager();
    workerManager.stopSearch();
  };

  const handleStartSearch = async () => {
    if (targetSeeds.seeds.length === 0) {
      alert('Please add target seeds before starting the search.');
      return;
    }

    clearSearchResults();
    startSearch();

    try {
      // Get the worker manager
      const workerManager = getSearchWorkerManager();
      
      // Set parallel mode based on settings
  workerManager.setParallelMode(searchExecutionMode === 'cpu-parallel');
      
      // Start search with worker
      await workerManager.startSearch(
        searchConditions,
        targetSeeds.seeds,
        {
          onProgress: (progress) => {
            useAppStore.getState().setSearchProgress({
              currentStep: progress.currentStep,
              totalSteps: progress.totalSteps,
              elapsedTime: progress.elapsedTime,
              estimatedTimeRemaining: progress.estimatedTimeRemaining,
              matchesFound: progress.matchesFound,
              currentDateTime: progress.currentDateTime,
            });
          },
          onParallelProgress: (aggregatedProgress) => {
            // 並列検索の詳細進捗を保存
            setParallelProgress(aggregatedProgress);
          },
          onResult: (result: InitialSeedResult) => {
            addSearchResult(result);
          },
          onComplete: (message: string) => {
            console.warn('Search completed:', message);
            
            // 検索時間を保存
            const currentProgress = useAppStore.getState().searchProgress;
            const currentParallelProgress = useAppStore.getState().parallelProgress;
            const finalElapsedTime = currentParallelProgress?.totalElapsedTime || currentProgress.elapsedTime;
            useAppStore.getState().setLastSearchDuration(finalElapsedTime);
            
            // 検索状態を停止
            stopSearch();
            
            // ワーカーマネージャーは次回検索開始時にリセット（統計情報を保持）
            // resetSearchWorkerManager(); ← 削除：統計表示を維持するため
            
            // その後でアラートを表示
            const matchesFound = useAppStore.getState().searchProgress.matchesFound;
            const totalSteps = useAppStore.getState().searchProgress.totalSteps;
            
            // 結果が0件の場合のみアラートを表示（状態更新の確実な完了を待つ）
            setTimeout(() => {
              if (matchesFound === 0) {
                alert(`Search completed. No matches found in ${totalSteps.toLocaleString()} combinations.\n\nTry:\n- Expanding the date range\n- Checking Timer0/VCount ranges\n- Verifying target seed format\n\nCheck browser console for detailed debug information.`);
              }
              // 結果が見つかった場合はダイアログを表示しない（ユーザーは結果タブで確認可能）
            }, 100);
          },
          onError: (error: string) => {
            console.error('Search error:', error);
            alert(`Search failed: ${error}`);
            stopSearch();
            // エラー時は即座にリセット（不正な状態を避けるため）
            resetSearchWorkerManager();
          },
          onPaused: () => {
            console.warn('Search paused by worker');
          },
          onResumed: () => {
            console.warn('Search resumed by worker');
          },
          onStopped: () => {
            console.warn('Search stopped by worker');
            stopSearch();
            // 停止時も統計情報保持（並列進捗も維持、次回検索開始時にリセット）
            // setParallelProgress(null); ← 削除：統計表示を維持
            // resetSearchWorkerManager(); ← 削除
          }
        }
      );
    } catch (error) {
      console.error('Failed to start worker search:', error);
      alert(`Failed to start search: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setParallelProgress(null);
      stopSearch();
      // 例外時は即座にリセット（不正な状態を避けるため）
      resetSearchWorkerManager();
    }
  };

  const handleMaxWorkersChange = (values: number[]) => {
    if (searchProgress.isRunning) {
      return;
    }
    const newWorkerCount = values[0];
    setMaxWorkers(newWorkerCount);
    
    // SearchWorkerManagerにも反映
    const workerManager = getSearchWorkerManager();
    workerManager.setMaxWorkers(newWorkerCount);
  };

  const maxCpuCores = navigator.hardwareConcurrency || 4;
  const isParallelAvailable = getSearchWorkerManager().isParallelSearchAvailable();
  const isWakeLockAvailable = isWakeLockSupported();
  const isWebGpuAvailable = isWebGpuSupported();

  const executionModeOptions: Array<{
    value: SearchExecutionMode;
    label: string;
    disabled: boolean;
    hint?: string;
  }> = [
    {
      value: 'cpu-parallel',
      label: 'CPU Parallel',
      disabled: !isParallelAvailable,
      hint: !isParallelAvailable ? 'Parallel workers are not available on this device' : undefined,
    },
    {
      value: 'cpu-single',
      label: 'CPU Single',
      disabled: false,
    },
    {
      value: 'gpu',
      label: 'GPU',
      disabled: !isWebGpuAvailable,
      hint: !isWebGpuAvailable ? 'WebGPU is not available in this browser' : undefined,
    },
  ];

  const handleExecutionModeChange = (value: string) => {
    if (searchProgress.isRunning) {
      alert('Cannot change execution mode while search is running.');
      return;
    }

    const nextMode = value as SearchExecutionMode;

    if (nextMode === 'gpu' && !isWebGpuAvailable) {
      return;
    }

    if (nextMode === 'cpu-parallel' && !isParallelAvailable) {
      return;
    }

    setSearchExecutionMode(nextMode);
  };

  useEffect(() => {
    if (!isWebGpuAvailable && searchExecutionMode === 'gpu') {
      setSearchExecutionMode(isParallelAvailable ? 'cpu-parallel' : 'cpu-single');
      return;
    }

    if (!isParallelAvailable && searchExecutionMode === 'cpu-parallel') {
      setSearchExecutionMode('cpu-single');
    }
  }, [isWebGpuAvailable, isParallelAvailable, searchExecutionMode, setSearchExecutionMode]);

  // Wake Lock設定の変更
  const handleWakeLockChange = (enabled: boolean) => {
    setWakeLockEnabled(enabled);
    if (!enabled) {
      // 無効にした場合は即座にWake Lockを解除
      releaseWakeLock();
    }
  };

  // 統一レイアウト: シンプルな検索制御
  return (
    <Card className={`py-2 flex flex-col ${isStack ? 'max-h-96' : 'h-full'} gap-2`}>
      <StandardCardHeader icon={<Play size={20} className="opacity-80" />} title="Search Control" />
      <StandardCardContent className="overflow-hidden">
        <div className="space-y-2">
          {/* 検索制御ボタンと設定 */}
          <div className="flex gap-2 items-center flex-wrap">
            {/* 検索ボタン */}
            <div className="flex gap-2 flex-1 min-w-0">
              {!searchProgress.isRunning ? (
                <Button 
                  onClick={handleStartSearch} 
                  disabled={targetSeeds.seeds.length === 0}
                  className="flex-1"
                  size="sm"
                >
                  <Play size={16} className="mr-2" />
                  Start Search
                </Button>
              ) : (
                <>
                  {!searchProgress.isPaused ? (
                    <Button 
                      onClick={handlePauseSearch}
                      variant="secondary"
                      className="flex-1"
                      size="sm"
                    >
                      <Pause size={16} className="mr-2" />
                      Pause
                    </Button>
                  ) : (
                    <Button 
                      onClick={handleResumeSearch}
                      className="flex-1"
                      size="sm"
                    >
                      <Play size={16} className="mr-2" />
                      Resume
                    </Button>
                  )}
                  <Button 
                    onClick={handleStopSearch}
                    variant="destructive"
                    size="sm"
                  >
                    <Square size={16} className="mr-2" />
                    Stop
                  </Button>
                </>
              )}
            </div>

            {/* Wake Lock設定 */}
            {isWakeLockAvailable && (
              <div className="flex items-center space-x-1">
                <Checkbox
                  id="wake-lock-inline"
                  checked={wakeLockEnabled}
                  onCheckedChange={handleWakeLockChange}
                />
                <Label htmlFor="wake-lock-inline" className="text-xs whitespace-nowrap">
                  Keep Screen On
                </Label>
              </div>
            )}
          </div>

          {/* 実行モード切り替え */}
          <div className="flex flex-wrap items-center gap-3">
            <RadioGroup
              className="flex w-full flex-wrap items-center gap-3"
              value={searchExecutionMode}
              onValueChange={handleExecutionModeChange}
              aria-label="Search execution mode"
            >
              {executionModeOptions.map((option) => {
                const id = `execution-mode-${option.value}`;
                return (
                  <div key={option.value} className="flex items-center space-x-1">
                    <RadioGroupItem
                      id={id}
                      value={option.value}
                      disabled={option.disabled || searchProgress.isRunning}
                    />
                    <Label
                      htmlFor={id}
                      className="text-xs whitespace-nowrap"
                      title={option.hint}
                    >
                      {option.label}
                    </Label>
                  </div>
                );
              })}
            </RadioGroup>
          </div>

          {/* 並列検索詳細設定 */}
          {isParallelAvailable && searchExecutionMode === 'cpu-parallel' && (
            <>
              <Separator />
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div id="worker-threads-label" className="text-sm">Worker Threads</div>
                  <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
                    {parallelSearchSettings.maxWorkers}
                  </span>
                </div>
                <Slider
                  aria-labelledby="worker-threads-label"
                  value={[parallelSearchSettings.maxWorkers]}
                  onValueChange={([value]) => handleMaxWorkersChange([value])}
                  min={1}
                  max={Math.max(maxCpuCores, 8)}
                  step={1}
                  disabled={searchProgress.isRunning || searchExecutionMode !== 'cpu-parallel'}
                  className="flex-1"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>1 worker</span>
                  <span>CPU cores: {maxCpuCores}</span>
                  <span>{Math.max(maxCpuCores, 8)} max</span>
                </div>
              </div>
            </>
          )}
        </div>
      </StandardCardContent>
    </Card>
  );
}
