import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
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
import type { DeviceProfileDraft } from '@/types/profile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatSearchControlChangeModeWhileRunningAlert,
  formatSearchControlCpuCoresLabel,
  formatSearchControlMaxWorkersLabel,
  formatSearchControlMissingTargetsAlert,
  formatSearchControlNoMatchesAlert,
  formatSearchControlSearchErrorAlert,
  formatSearchControlStartErrorAlert,
  resolveSearchControlButtonLabel,
  resolveSearchControlProfileSyncDialog,
  resolveSearchControlExecutionModeHint,
  resolveSearchControlExecutionModeLabel,
  searchControlExecutionModeAriaLabel,
  searchControlPanelTitle,
  searchControlWakeLockLabel,
  searchControlWorkerMinLabel,
  searchControlWorkerThreadsLabel,
} from '@/lib/i18n/strings/search-control';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useProfileFormStore } from '@/store/profile-form-store';
import { deviceProfileToDraft } from '@/types/profile';
import { areDeviceProfileDraftsEqual } from '@/lib/utils/profile-draft';

export function SearchControlCard() {
  const { isStack } = useResponsiveLayout();
  const {
    searchConditions,
    searchProgress,
    startSearch,
    pauseSearch,
    resumeSearch,
  stopSearch,
  completeSearch,
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
    profiles,
    activeProfileId,
    updateProfile,
  } = useAppStore();
  const locale = useLocale();
  const profileDirty = useProfileFormStore((state) => state.isDirty);
  const profileDraft = useProfileFormStore((state) => state.draft);
  const profileValidationErrors = useProfileFormStore((state) => state.validationErrors);
  const [profileSyncDialogOpen, setProfileSyncDialogOpen] = useState(false);
  const profileSyncPromiseResolveRef = useRef<((value: boolean) => void) | null>(null);
  const pendingProfileDraftRef = useRef<DeviceProfileDraft | null>(null);
  const profileSyncCloseActionRef = useRef<'confirm' | 'cancel' | null>(null);

  const activeProfile = useMemo(() => {
    if (!profiles.length) {
      return null;
    }
    if (activeProfileId) {
      const found = profiles.find((profile) => profile.id === activeProfileId);
      if (found) {
        return found;
      }
    }
    return profiles[0];
  }, [activeProfileId, profiles]);

  const profileSyncDialogText = useMemo(() => resolveSearchControlProfileSyncDialog(locale), [locale]);

  // ワーカー数設定を初期化時に同期
  useEffect(() => {
    const workerManager = getSearchWorkerManager();
    workerManager.setMaxWorkers(parallelSearchSettings.maxWorkers);
  }, [parallelSearchSettings.maxWorkers]);

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

  const resolveProfileSyncRequest = useCallback((result: boolean) => {
    if (profileSyncPromiseResolveRef.current) {
      profileSyncPromiseResolveRef.current(result);
      profileSyncPromiseResolveRef.current = null;
    }
    pendingProfileDraftRef.current = null;
  }, []);

  const handleProfileSyncCancel = useCallback(() => {
    profileSyncCloseActionRef.current = 'cancel';
    setProfileSyncDialogOpen(false);
    resolveProfileSyncRequest(false);
  }, [resolveProfileSyncRequest]);

  const handleProfileSyncConfirm = useCallback(() => {
    const draft = pendingProfileDraftRef.current;
    if (!draft || !activeProfile) {
      profileSyncCloseActionRef.current = 'cancel';
      setProfileSyncDialogOpen(false);
      resolveProfileSyncRequest(false);
      return;
    }
    profileSyncCloseActionRef.current = 'confirm';
    updateProfile(activeProfile.id, draft);
    useProfileFormStore.getState().reset();
    setProfileSyncDialogOpen(false);
    resolveProfileSyncRequest(true);
  }, [activeProfile, resolveProfileSyncRequest, updateProfile]);

  const handleProfileSyncDialogOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        setProfileSyncDialogOpen(true);
        return;
      }
      if (profileSyncCloseActionRef.current) {
        profileSyncCloseActionRef.current = null;
        return;
      }
      handleProfileSyncCancel();
    },
    [handleProfileSyncCancel],
  );

  const ensureProfileSyncedBeforeSearch = useCallback(async (): Promise<boolean> => {
    if (!profileDirty) {
      return true;
    }
    if (!profileDraft) {
      const message = profileValidationErrors[0] ?? profileSyncDialogText.validationError;
      alert(message);
      return false;
    }
    if (!activeProfile) {
      alert(profileSyncDialogText.missingProfile);
      return false;
    }
    const currentDraft = deviceProfileToDraft(activeProfile);
    if (areDeviceProfileDraftsEqual(profileDraft, currentDraft)) {
      return true;
    }
    pendingProfileDraftRef.current = profileDraft;
    profileSyncCloseActionRef.current = null;
    setProfileSyncDialogOpen(true);
    return await new Promise<boolean>((resolve) => {
      profileSyncPromiseResolveRef.current = resolve;
    });
  }, [activeProfile, profileDirty, profileDraft, profileSyncDialogText, profileValidationErrors, setProfileSyncDialogOpen]);

  const performSearchStart = useCallback(async () => {
    clearSearchResults();
    startSearch();

    try {
      const workerManager = getSearchWorkerManager();

      await workerManager.startSearch(searchConditions, targetSeeds.seeds, {
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
          setParallelProgress(aggregatedProgress);
        },
        onResult: (result: InitialSeedResult) => {
          addSearchResult(result);
        },
        onComplete: (message: string) => {
          console.warn('Search completed:', message);
          const currentProgress = useAppStore.getState().searchProgress;
          const currentParallelProgress = useAppStore.getState().parallelProgress;
          const finalElapsedTime = currentParallelProgress?.totalElapsedTime || currentProgress.elapsedTime;
          useAppStore.getState().setLastSearchDuration(finalElapsedTime);
          completeSearch();
          const matchesFound = useAppStore.getState().searchProgress.matchesFound;
          const totalSteps = useAppStore.getState().searchProgress.totalSteps;
          setTimeout(() => {
            if (matchesFound === 0) {
              alert(formatSearchControlNoMatchesAlert(totalSteps, locale));
            }
          }, 100);
        },
        onError: (error: string) => {
          console.error('Search error:', error);
          alert(formatSearchControlSearchErrorAlert(error, locale));
          stopSearch();
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
        },
      });
    } catch (error) {
      console.error('Failed to start worker search:', error);
      const message = error instanceof Error ? error.message : locale === 'ja' ? '不明なエラー' : 'Unknown error';
      alert(formatSearchControlStartErrorAlert(message, locale));
      setParallelProgress(null);
      stopSearch();
      resetSearchWorkerManager();
    }
  }, [
    addSearchResult,
    clearSearchResults,
    completeSearch,
    locale,
    searchConditions,
    setParallelProgress,
    startSearch,
    stopSearch,
    targetSeeds.seeds,
  ]);

  const handleStartSearch = async () => {
    if (targetSeeds.seeds.length === 0) {
      alert(formatSearchControlMissingTargetsAlert(locale));
      return;
    }

    const isProfileReady = await ensureProfileSyncedBeforeSearch();
    if (!isProfileReady) {
      return;
    }

    await performSearchStart();
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
      label: resolveSearchControlExecutionModeLabel('cpuParallel', locale),
      disabled: !isParallelAvailable,
      hint: !isParallelAvailable
        ? resolveSearchControlExecutionModeHint('cpuParallelUnavailable', locale)
        : undefined,
    },
    {
      value: 'gpu',
      label: resolveSearchControlExecutionModeLabel('gpu', locale),
      disabled: !isWebGpuAvailable,
      hint: !isWebGpuAvailable
        ? resolveSearchControlExecutionModeHint('gpuUnavailable', locale)
        : undefined,
    },
  ];

  const handleExecutionModeChange = (value: string) => {
    if (searchProgress.isRunning) {
      alert(formatSearchControlChangeModeWhileRunningAlert(locale));
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
    if (searchExecutionMode === 'gpu' && !isWebGpuAvailable) {
      if (isParallelAvailable) {
        setSearchExecutionMode('cpu-parallel');
      }
      return;
    }

    if (searchExecutionMode === 'cpu-parallel' && !isParallelAvailable && isWebGpuAvailable) {
      setSearchExecutionMode('gpu');
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
    <>
      <PanelCard
      icon={<Play size={20} className="opacity-80" />} 
      title={resolveLocaleValue(searchControlPanelTitle, locale)}
      className={isStack ? 'max-h-96' : undefined}
      fullHeight={!isStack}
    >
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
                  {resolveSearchControlButtonLabel('start', locale)}
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
                      {resolveSearchControlButtonLabel('pause', locale)}
                    </Button>
                  ) : (
                    <Button 
                      onClick={handleResumeSearch}
                      className="flex-1"
                      size="sm"
                    >
                      <Play size={16} className="mr-2" />
                      {resolveSearchControlButtonLabel('resume', locale)}
                    </Button>
                  )}
                  <Button 
                    onClick={handleStopSearch}
                    variant="destructive"
                    size="sm"
                  >
                    <Square size={16} className="mr-2" />
                    {resolveSearchControlButtonLabel('stop', locale)}
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
                  {resolveLocaleValue(searchControlWakeLockLabel, locale)}
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
              aria-label={resolveLocaleValue(searchControlExecutionModeAriaLabel, locale)}
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
                  <div id="worker-threads-label" className="text-sm">
                    {resolveLocaleValue(searchControlWorkerThreadsLabel, locale)}
                  </div>
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
                  <span>{resolveLocaleValue(searchControlWorkerMinLabel, locale)}</span>
                  <span>{formatSearchControlCpuCoresLabel(maxCpuCores, locale)}</span>
                  <span>{formatSearchControlMaxWorkersLabel(Math.max(maxCpuCores, 8), locale)}</span>
                </div>
              </div>
            </>
          )}
        </div>
      </PanelCard>

      <Dialog open={profileSyncDialogOpen} onOpenChange={handleProfileSyncDialogOpenChange}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{profileSyncDialogText.title}</DialogTitle>
            <DialogDescription>{profileSyncDialogText.description}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={handleProfileSyncCancel}>
              {profileSyncDialogText.cancel}
            </Button>
            <Button onClick={handleProfileSyncConfirm}>
              {profileSyncDialogText.confirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
