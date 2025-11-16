import React, { useCallback, useMemo, useRef, useState } from 'react';
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
  resolveGenerationRunProfileSyncDialog,
} from '@/lib/i18n/strings/generation-run';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useProfileFormStore } from '@/store/profile-form-store';
import { deviceProfileToDraft } from '@/types/profile';
import { areDeviceProfileDraftsEqual } from '@/lib/utils/profile-draft';
import type { DeviceProfileDraft } from '@/types/profile';

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
    profiles,
    activeProfileId,
    updateProfile,
  } = useAppStore();

  const locale = useLocale();
  const total = progress?.totalAdvances ?? draftParams.maxAdvances ?? 0;
  const done = progress?.processedAdvances ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;

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

  const profileSyncDialogText = useMemo(() => resolveGenerationRunProfileSyncDialog(locale), [locale]);

  const { isStack } = useResponsiveLayout();
  const isStarting = status === 'starting';
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
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

  const ensureProfileSyncedBeforeGeneration = useCallback(async (): Promise<boolean> => {
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
  }, [activeProfile, profileDirty, profileDraft, profileSyncDialogText, profileValidationErrors]);

  const handleStart = useCallback(async () => {
    const ready = await ensureProfileSyncedBeforeGeneration();
    if (!ready) {
      return;
    }
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  }, [ensureProfileSyncedBeforeGeneration, startGeneration, validateDraft, validationErrors.length]);

  const canStart = status === 'idle' || status === 'completed' || status === 'error';
  const statusPrefix = resolveLocaleValue(generationRunStatusPrefix, locale);
  const startLabel = resolveLocaleValue(generationRunButtonLabels.start, locale);
  const startingLabel = resolveLocaleValue(generationRunButtonLabels.starting, locale);
  const pauseLabel = resolveLocaleValue(generationRunButtonLabels.pause, locale);
  const resumeLabel = resolveLocaleValue(generationRunButtonLabels.resume, locale);
  const stopLabel = resolveLocaleValue(generationRunButtonLabels.stop, locale);
  const title = resolveLocaleValue(generationRunPanelTitle, locale);
  const controlsLabel = resolveLocaleValue(generationRunControlsLabel, locale);
  const progressLabel = resolveLocaleValue(generationRunProgressLabel, locale);
  const progressBarLabel = resolveLocaleValue(generationRunProgressBarLabel, locale);

  const statusDisplay = formatGenerationRunStatusDisplay(status, lastCompletion?.reason ?? null, locale);
  const advancesDisplay = formatGenerationRunAdvancesDisplay(done, total, locale);
  const percentDisplay = formatGenerationRunPercentDisplay(pct, locale);
  const screenReaderSummary = formatGenerationRunScreenReaderSummary(statusDisplay, advancesDisplay, percentDisplay, locale);

  return (
    <>
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
};
