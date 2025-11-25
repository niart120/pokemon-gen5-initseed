import React from 'react';
import { toast } from 'sonner';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useAppStore } from '@/store/app-store';
import type { DeviceProfile, DeviceProfileDraft } from '@/types/profile';
import { deviceProfileToDraft } from '@/types/profile';
import { getFullTimer0Range, getValidVCounts } from '@/lib/utils/rom-parameter-helpers';
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import { useLocale } from '@/lib/i18n/locale-context';
import { areDeviceProfileDraftsEqual } from '@/lib/utils/profile-draft';
import {
  formatProfileCreatedToast,
  formatProfileDeleteConfirm,
  resolveProfileDefaultName,
  resolveProfileDeletedToast,
  resolveProfileImportedName,
  resolveProfileImportedToast,
  resolveProfileMinimumError,
  resolveProfileNewName,
} from '@/lib/i18n/strings/profile-messages';
import { resolveProfileTimerFieldLabel } from '@/lib/i18n/strings/profile-timer';
import { resolveProfileGameFieldLabel } from '@/lib/i18n/strings/profile-game';
import { resolveProfileManagementLockLabel } from '@/lib/i18n/strings/profile-management';
import {
  canonicalizeHex,
  enforceMemoryLink,
  enforceShinyCharm,
  formToDraft,
  profileToForm,
  toTimerHex,
  toVCountHex,
} from './profileFormAdapter';
import type { ProfileFormState, SectionKey, SectionState } from './profileFormTypes';
import { buildDraftFromCurrentState } from './profileDraftBuilder';
import { useProfileSections } from './useProfileSections';
import { useMacAddressInput } from './useMacAddressInput';

export type { SectionKey, SectionState, ProfileFormState } from './profileFormTypes';

const TIMER_HEX_PATTERN = /^0x?[0-9a-fA-F]{0,4}$/;
const VCOUNT_HEX_PATTERN = /^0x?[0-9a-fA-F]{0,2}$/;

export const SELECT_NEW_PROFILE = '__new_profile__';
export const SELECT_IMPORT_CURRENT = '__import_current__';

interface ProfileSelectionControls {
  profiles: DeviceProfile[];
  activeId: string;
  onSelect: (value: string) => void;
}

interface HeaderControls {
  profileName: string;
  canModify: boolean;
  disableDelete: boolean;
  lockedReason: string | null;
  onProfileNameChange: (value: string) => void;
  onDelete: () => void;
}

interface LayoutControls {
  sectionOpen: SectionState;
  isStack: boolean;
  toggleSection: (key: SectionKey) => void;
}

interface RomSectionControls {
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macSegments: string[];
  macInputRefs: React.MutableRefObject<Array<HTMLInputElement | null>>;
  onRomVersionChange: (value: ROMVersion) => void;
  onRomRegionChange: (value: ROMRegion) => void;
  onHardwareChange: (value: Hardware) => void;
  onMacSegmentChange: (index: number, value: string) => void;
  onMacSegmentFocus: (event: React.FocusEvent<HTMLInputElement>) => void;
  onMacSegmentMouseDown: (event: React.MouseEvent<HTMLInputElement>) => void;
  onMacSegmentClick: (event: React.MouseEvent<HTMLInputElement>) => void;
  onMacSegmentKeyDown: (index: number, event: React.KeyboardEvent<HTMLInputElement>) => void;
  onMacSegmentPaste: (index: number, event: React.ClipboardEvent<HTMLInputElement>) => void;
  disabled: boolean;
}

interface TimerSectionControls {
  timer0Auto: boolean;
  timer0Min: string;
  timer0Max: string;
  vcountMin: string;
  vcountMax: string;
  onAutoToggle: (checked: boolean) => void;
  onTimerHexChange: (field: 'timer0Min' | 'timer0Max', value: string) => void;
  onTimerHexBlur: (field: 'timer0Min' | 'timer0Max') => void;
  onVCountHexChange: (field: 'vcountMin' | 'vcountMax', value: string) => void;
  onVCountHexBlur: (field: 'vcountMin' | 'vcountMax') => void;
  disabled: boolean;
}

interface GameSectionControls {
  tid: string;
  sid: string;
  newGame: boolean;
  withSave: boolean;
  shinyCharm: boolean;
  memoryLink: boolean;
  withSaveDisabled: boolean;
  shinyCharmDisabled: boolean;
  memoryLinkDisabled: boolean;
  onTidChange: (value: string) => void;
  onSidChange: (value: string) => void;
  onNewGameToggle: (checked: boolean) => void;
  onWithSaveToggle: (checked: boolean) => void;
  onShinyCharmToggle: (checked: boolean) => void;
  onMemoryLinkToggle: (checked: boolean) => void;
  disabled: boolean;
}

export interface UseProfileCardFormResult {
  errors: string[];
  profileSelection: ProfileSelectionControls;
  header: HeaderControls;
  layout: LayoutControls;
  rom: RomSectionControls;
  timer: TimerSectionControls;
  game: GameSectionControls;
}

export function useProfileCardForm(): UseProfileCardFormResult {
  const profiles = useAppStore((state) => state.profiles);
  const activeProfileId = useAppStore((state) => state.activeProfileId);
  const setActiveProfile = useAppStore((state) => state.setActiveProfile);
  const createProfile = useAppStore((state) => state.createProfile);
  const updateProfile = useAppStore((state) => state.updateProfile);
  const deleteProfile = useAppStore((state) => state.deleteProfile);
  const generationStatus = useAppStore((state) => state.status);
  const searchProgress = useAppStore((state) => state.searchProgress);
  const locale = useLocale();

  const defaultProfileName = resolveProfileDefaultName(locale);
  const newProfileName = resolveProfileNewName(locale);
  const importedProfileName = resolveProfileImportedName(locale);
  const importedToast = resolveProfileImportedToast(locale);
  const deletedToast = resolveProfileDeletedToast(locale);
  const minimumProfileError = resolveProfileMinimumError(locale);
  const timer0MinLabel = resolveProfileTimerFieldLabel('timer0Min', locale);
  const timer0MaxLabel = resolveProfileTimerFieldLabel('timer0Max', locale);
  const vcountMinLabel = resolveProfileTimerFieldLabel('vcountMin', locale);
  const vcountMaxLabel = resolveProfileTimerFieldLabel('vcountMax', locale);
  const tidLabel = resolveProfileGameFieldLabel('tid', locale);
  const sidLabel = resolveProfileGameFieldLabel('sid', locale);

  const activeProfile = React.useMemo(
    () => resolveActiveProfile(profiles, activeProfileId),
    [profiles, activeProfileId],
  );

  const activeProfileSnapshot = React.useMemo<DeviceProfileDraft | null>(
    () => (activeProfile ? deviceProfileToDraft(activeProfile) : null),
    [activeProfile],
  );

  const lastSubmittedDraftRef = React.useRef<DeviceProfileDraft | null>(null);

  const [form, setForm] = React.useState<ProfileFormState>(() =>
    activeProfileSnapshot
      ? profileToForm(activeProfileSnapshot)
      : profileToForm(buildDraftFromCurrentState(defaultProfileName, undefined)),
  );
  const [errors, setErrors] = React.useState<string[]>([]);
  const { isStack } = useResponsiveLayout();

  const { sectionOpen, toggleSection } = useProfileSections(isStack);

  const validationLabels = React.useMemo(
    () => ({
      timer0Min: timer0MinLabel,
      timer0Max: timer0MaxLabel,
      vcountMin: vcountMinLabel,
      vcountMax: vcountMaxLabel,
      tid: tidLabel,
      sid: sidLabel,
    }),
    [sidLabel, tidLabel, timer0MaxLabel, timer0MinLabel, vcountMaxLabel, vcountMinLabel],
  );

  React.useEffect(() => {
    if (!activeProfileSnapshot) {
      setForm(profileToForm(buildDraftFromCurrentState(defaultProfileName, undefined)));
      setErrors([]);
      lastSubmittedDraftRef.current = null;
      return;
    }
    if (lastSubmittedDraftRef.current && areDeviceProfileDraftsEqual(activeProfileSnapshot, lastSubmittedDraftRef.current)) {
      lastSubmittedDraftRef.current = null;
      return;
    }
    setForm(profileToForm(activeProfileSnapshot));
    setErrors([]);
  }, [activeProfileSnapshot, defaultProfileName]);

  const isGenerationBusy =
    generationStatus === 'starting' || generationStatus === 'running' || generationStatus === 'stopping';
  const isSearchBusy = searchProgress.isRunning || searchProgress.isPaused;
  const profileLocked = isGenerationBusy || isSearchBusy;
  const lockedReason = profileLocked ? resolveProfileManagementLockLabel(locale) : null;
  const canModify = Boolean(activeProfile) && !profileLocked;

  React.useEffect(() => {
    const { draft, validationErrors } = formToDraft(form, locale, validationLabels);
    setErrors(validationErrors);
    if (!draft || !activeProfile || !activeProfileSnapshot || profileLocked) {
      return;
    }
    if (areDeviceProfileDraftsEqual(draft, activeProfileSnapshot)) {
      return;
    }
    updateProfile(activeProfile.id, draft);
    lastSubmittedDraftRef.current = draft;
  }, [form, activeProfile, activeProfileSnapshot, locale, validationLabels, profileLocked, updateProfile]);

  const notifyLock = React.useCallback(() => {
    if (lockedReason) {
      toast.error(lockedReason);
    }
  }, [lockedReason]);

  const updateMacSegments = React.useCallback(
    (updater: (prev: string[]) => string[]) => {
      if (!canModify) return;
      setForm((prev) => {
        const nextSegments = updater(prev.macSegments);
        if (nextSegments === prev.macSegments) {
          return prev;
        }
        return { ...prev, macSegments: nextSegments };
      });
    },
    [canModify],
  );

  const {
    macInputRefs,
    handleMacSegmentChange,
    handleMacSegmentFocus,
    handleMacSegmentMouseDown,
    handleMacSegmentClick,
    handleMacSegmentKeyDown,
    handleMacSegmentPaste,
  } = useMacAddressInput({ macSegments: form.macSegments, updateSegments: updateMacSegments });

  const handleSelectProfile = React.useCallback(
    (value: string) => {
      if (!value || value === activeProfile?.id) {
        return;
      }
      if (value === SELECT_NEW_PROFILE) {
        if (!canModify) {
          notifyLock();
          return;
        }
        const draft = buildDraftFromCurrentState(newProfileName, undefined);
        const profile = createProfile(draft);
        setActiveProfile(profile.id);
        setForm(profileToForm(deviceProfileToDraft(profile)));
        toast.success(formatProfileCreatedToast(profile.name, locale));
        return;
      }
      if (value === SELECT_IMPORT_CURRENT) {
        if (!canModify || !activeProfile) {
          notifyLock();
          return;
        }
        const draft = buildDraftFromCurrentState(form.name || importedProfileName, activeProfile);
        setForm(profileToForm(draft));
        updateProfile(activeProfile.id, draft);
        toast.success(importedToast);
        return;
      }
      if (!canModify) {
        notifyLock();
        return;
      }
      setActiveProfile(value);
    },
    [
      activeProfile,
      canModify,
      createProfile,
      form.name,
      importedProfileName,
      importedToast,
      locale,
      newProfileName,
      notifyLock,
      setActiveProfile,
      updateProfile,
    ],
  );

  const handleRomVersionChange = React.useCallback((value: ROMVersion) => {
    if (!canModify) return;
    setForm((prev) => {
      const next = {
        ...prev,
        romVersion: value,
        memoryLink: enforceMemoryLink(prev.memoryLink, value, prev.withSave),
        shinyCharm: enforceShinyCharm(prev.shinyCharm, value, prev.withSave),
      };
      if (prev.timer0Auto) {
        const range = getFullTimer0Range(value, prev.romRegion);
        const vcounts = getValidVCounts(value, prev.romRegion);
        if (range) {
          next.timer0Min = toTimerHex(range.min);
          next.timer0Max = toTimerHex(range.max);
        }
        if (vcounts.length > 0) {
          const minV = Math.min(...vcounts);
          const maxV = Math.max(...vcounts);
          next.vcountMin = toVCountHex(minV);
          next.vcountMax = toVCountHex(maxV);
        }
      }
      return next;
    });
  }, [canModify]);

  const handleRomRegionChange = React.useCallback((value: ROMRegion) => {
    if (!canModify) return;
    setForm((prev) => {
      const next = { ...prev, romRegion: value };
      if (prev.timer0Auto) {
        const range = getFullTimer0Range(prev.romVersion, value);
        const vcounts = getValidVCounts(prev.romVersion, value);
        if (range) {
          next.timer0Min = toTimerHex(range.min);
          next.timer0Max = toTimerHex(range.max);
        }
        if (vcounts.length > 0) {
          const minV = Math.min(...vcounts);
          const maxV = Math.max(...vcounts);
          next.vcountMin = toVCountHex(minV);
          next.vcountMax = toVCountHex(maxV);
        }
      }
      return next;
    });
  }, [canModify]);

  const handleHardwareChange = React.useCallback((value: Hardware) => {
    if (!canModify) return;
    setForm((prev) => ({ ...prev, hardware: value }));
  }, [canModify]);

  const handleTimerAutoToggle = React.useCallback((checked: boolean) => {
    if (!canModify) return;
    setForm((prev) => {
      if (!checked) {
        return { ...prev, timer0Auto: false };
      }
      const next = { ...prev, timer0Auto: true };
      const range = getFullTimer0Range(prev.romVersion, prev.romRegion);
      const vcounts = getValidVCounts(prev.romVersion, prev.romRegion);
      if (range) {
        next.timer0Min = toTimerHex(range.min);
        next.timer0Max = toTimerHex(range.max);
      }
      if (vcounts.length > 0) {
        const minV = Math.min(...vcounts);
        const maxV = Math.max(...vcounts);
        next.vcountMin = toVCountHex(minV);
        next.vcountMax = toVCountHex(maxV);
      }
      return next;
    });
  }, [canModify]);

  const handleTimerHexChange = React.useCallback(
    (field: 'timer0Min' | 'timer0Max', value: string) => {
      if (!canModify) return;
      if (value === '' || TIMER_HEX_PATTERN.test(value)) {
        setForm((prev) => ({ ...prev, [field]: value }));
      }
    },
    [canModify],
  );

  const handleTimerHexBlur = React.useCallback((field: 'timer0Min' | 'timer0Max') => {
    if (!canModify) return;
    setForm((prev) => {
      const canonical = canonicalizeHex(prev[field], 4);
      if (canonical === prev[field]) {
        return prev;
      }
      return { ...prev, [field]: canonical };
    });
  }, [canModify]);

  const handleVCountHexChange = React.useCallback(
    (field: 'vcountMin' | 'vcountMax', value: string) => {
      if (!canModify) return;
      if (value === '' || VCOUNT_HEX_PATTERN.test(value)) {
        setForm((prev) => ({ ...prev, [field]: value }));
      }
    },
    [canModify],
  );

  const handleVCountHexBlur = React.useCallback((field: 'vcountMin' | 'vcountMax') => {
    if (!canModify) return;
    setForm((prev) => {
      const canonical = canonicalizeHex(prev[field], 2);
      if (canonical === prev[field]) {
        return prev;
      }
      return { ...prev, [field]: canonical };
    });
  }, [canModify]);

  const handleNameChange = React.useCallback((value: string) => {
    if (!canModify) return;
    setForm((prev) => ({ ...prev, name: value }));
  }, [canModify]);

  const handleTidChange = React.useCallback((value: string) => {
    if (!canModify) return;
    if (/^\d{0,5}$/.test(value)) {
      setForm((prev) => ({ ...prev, tid: value }));
    }
  }, [canModify]);

  const handleSidChange = React.useCallback((value: string) => {
    if (!canModify) return;
    if (/^\d{0,5}$/.test(value)) {
      setForm((prev) => ({ ...prev, sid: value }));
    }
  }, [canModify]);

  const handleShinyCharmToggle = React.useCallback((checked: boolean) => {
    if (!canModify) return;
    setForm((prev) => ({ ...prev, shinyCharm: checked }));
  }, [canModify]);

  const handleNewGameToggle = React.useCallback((checked: boolean) => {
    if (!canModify) return;
    setForm((prev) => {
      const nextWithSave = checked ? false : true;
      const nextMemoryLink = enforceMemoryLink(prev.memoryLink, prev.romVersion, nextWithSave);
      const nextShinyCharm = enforceShinyCharm(prev.shinyCharm, prev.romVersion, nextWithSave);
      return {
        ...prev,
        newGame: checked,
        withSave: nextWithSave,
        memoryLink: nextMemoryLink,
        shinyCharm: nextShinyCharm,
      };
    });
  }, [canModify]);

  const handleWithSaveToggle = React.useCallback((checked: boolean) => {
    if (!canModify) return;
    setForm((prev) => {
      if (!prev.newGame) {
        return prev;
      }
      const nextWithSave = Boolean(checked);
      return {
        ...prev,
        withSave: nextWithSave,
        memoryLink: enforceMemoryLink(prev.memoryLink, prev.romVersion, nextWithSave),
        shinyCharm: enforceShinyCharm(prev.shinyCharm, prev.romVersion, nextWithSave),
      };
    });
  }, [canModify]);

  const handleMemoryLinkToggle = React.useCallback((checked: boolean) => {
    if (!canModify) return;
    setForm((prev) => ({
      ...prev,
      memoryLink: enforceMemoryLink(checked, prev.romVersion, prev.withSave),
    }));
  }, [canModify]);

  const handleDelete = React.useCallback(() => {
    if (!activeProfile) return;
    if (!canModify) {
      notifyLock();
      return;
    }
    if (profiles.length <= 1) {
      toast.error(minimumProfileError);
      return;
    }
    if (!window.confirm(formatProfileDeleteConfirm(activeProfile.name, locale))) {
      return;
    }
    deleteProfile(activeProfile.id);
    toast.success(deletedToast);
  }, [activeProfile, canModify, deleteProfile, deletedToast, locale, minimumProfileError, notifyLock, profiles.length]);

  const disableDelete = profiles.length <= 1 || !canModify;
  const memoryLinkDisabled = form.romVersion === 'B' || form.romVersion === 'W' || !form.withSave;
  const withSaveDisabled = !form.newGame;
  const shinyCharmDisabled = form.romVersion === 'B' || form.romVersion === 'W' || !form.withSave;

  const profilesForDisplay = React.useMemo(() => {
    if (!activeProfile) {
      return profiles;
    }
    if (form.name === activeProfile.name) {
      return profiles;
    }
    return profiles.map((profile) => (profile.id === activeProfile.id ? { ...profile, name: form.name } : profile));
  }, [activeProfile, form.name, profiles]);

  const profileSelection: ProfileSelectionControls = {
    profiles: profilesForDisplay,
    activeId: activeProfile?.id ?? '',
    onSelect: handleSelectProfile,
  };

  const header: HeaderControls = {
    profileName: form.name,
    canModify,
    disableDelete,
    lockedReason,
    onProfileNameChange: handleNameChange,
    onDelete: handleDelete,
  };

  const layout: LayoutControls = {
    sectionOpen,
    isStack,
    toggleSection,
  };

  const rom: RomSectionControls = {
    romVersion: form.romVersion,
    romRegion: form.romRegion,
    hardware: form.hardware,
    macSegments: form.macSegments,
    macInputRefs,
    onRomVersionChange: handleRomVersionChange,
    onRomRegionChange: handleRomRegionChange,
    onHardwareChange: handleHardwareChange,
    onMacSegmentChange: handleMacSegmentChange,
    onMacSegmentFocus: handleMacSegmentFocus,
    onMacSegmentMouseDown: handleMacSegmentMouseDown,
    onMacSegmentClick: handleMacSegmentClick,
    onMacSegmentKeyDown: handleMacSegmentKeyDown,
    onMacSegmentPaste: handleMacSegmentPaste,
    disabled: !canModify,
  };

  const timer: TimerSectionControls = {
    timer0Auto: form.timer0Auto,
    timer0Min: form.timer0Min,
    timer0Max: form.timer0Max,
    vcountMin: form.vcountMin,
    vcountMax: form.vcountMax,
    onAutoToggle: handleTimerAutoToggle,
    onTimerHexChange: handleTimerHexChange,
    onTimerHexBlur: handleTimerHexBlur,
    onVCountHexChange: handleVCountHexChange,
    onVCountHexBlur: handleVCountHexBlur,
    disabled: !canModify,
  };

  const game: GameSectionControls = {
    tid: form.tid,
    sid: form.sid,
    newGame: form.newGame,
    withSave: form.withSave,
    shinyCharm: form.shinyCharm,
    memoryLink: form.memoryLink,
    withSaveDisabled,
    shinyCharmDisabled,
    memoryLinkDisabled,
    onTidChange: handleTidChange,
    onSidChange: handleSidChange,
    onNewGameToggle: handleNewGameToggle,
    onWithSaveToggle: handleWithSaveToggle,
    onShinyCharmToggle: handleShinyCharmToggle,
    onMemoryLinkToggle: handleMemoryLinkToggle,
    disabled: !canModify,
  };

  return {
    errors,
    profileSelection,
    header,
    layout,
    rom,
    timer,
    game,
  };
}

function resolveActiveProfile(profiles: DeviceProfile[], activeId: string | null): DeviceProfile | null {
  if (!profiles.length) return null;
  if (activeId) {
    const found = profiles.find((profile) => profile.id === activeId);
    if (found) return found;
  }
  return profiles[0];
}
