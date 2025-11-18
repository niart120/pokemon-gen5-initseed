import React, { useCallback, useMemo, useRef, useState } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Play, Square, ChartBar } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatGenerationRunAdvancesDisplay,
  formatGenerationRunPercentDisplay,
  formatGenerationRunScreenReaderSummary,
  formatGenerationRunStatusDisplay,
  generationRunButtonLabels,
  generationRunControlsLabel,
  generationRunPanelTitle,
  generationRunResultsLabel,
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
    stopGeneration,
    status,
    lastCompletion,
    draftParams,
    profiles,
    activeProfileId,
    updateProfile,
  } = useAppStore();
  const resultCount = useAppStore(s => s.results.length);
  const params = useAppStore(s => s.params);

  const locale = useLocale();
  const maxResults = params?.maxResults ?? draftParams.maxResults ?? 0;
  const pct = maxResults > 0 ? (resultCount / maxResults) * 100 : 0;

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
  const isStopping = status === 'stopping';
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
  const stopLabel = resolveLocaleValue(generationRunButtonLabels.stop, locale);
  const title = resolveLocaleValue(generationRunPanelTitle, locale);
  const controlsLabel = resolveLocaleValue(generationRunControlsLabel, locale);
  const resultsLabel: string = resolveLocaleValue(generationRunResultsLabel, locale);

  const statusDisplay = formatGenerationRunStatusDisplay(status, lastCompletion?.reason ?? null, locale);
  const advancesDisplay = formatGenerationRunAdvancesDisplay(resultCount, maxResults, locale);
  const percentDisplay = formatGenerationRunPercentDisplay(pct, locale);
  const screenReaderSummary = formatGenerationRunScreenReaderSummary(statusDisplay, advancesDisplay, percentDisplay, locale);

  return (
    <>
      <PanelCard
      icon={<ChartBar size={20} className="opacity-80" />}
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
            {statusPrefix} {statusDisplay}
          </div>
        </div>
        {/* Result summary */}
        <div className="space-y-1" aria-label={resultsLabel}>
          <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
            <span>{percentDisplay}</span>
            <span>{advancesDisplay}</span>
          </div>
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
