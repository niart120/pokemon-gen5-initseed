import React from 'react';
import { toast } from 'sonner';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useAppStore } from '@/store/app-store';
import type { DeviceProfile } from '@/types/profile';
import { deviceProfileToDraft } from '@/types/profile';
import { getFullTimer0Range, getValidVCounts } from '@/lib/utils/rom-parameter-helpers';
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatProfileCreatedToast,
  formatProfileDeleteConfirm,
  resolveProfileDefaultName,
  resolveProfileDeletedToast,
  resolveProfileImportedName,
  resolveProfileImportedToast,
  resolveProfileMinimumError,
  resolveProfileNewName,
  resolveProfileSavedToast,
} from '@/lib/i18n/strings/profile-messages';
import { resolveProfileTimerFieldLabel } from '@/lib/i18n/strings/profile-timer';
import { resolveProfileGameFieldLabel } from '@/lib/i18n/strings/profile-game';
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
  dirty: boolean;
  profileName: string;
  canModify: boolean;
  disableDelete: boolean;
  onProfileNameChange: (value: string) => void;
  onSave: () => void;
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
  const locale = useLocale();

  const defaultProfileName = resolveProfileDefaultName(locale);
  const newProfileName = resolveProfileNewName(locale);
  const importedProfileName = resolveProfileImportedName(locale);
  const importedToast = resolveProfileImportedToast(locale);
  const savedToast = resolveProfileSavedToast(locale);
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

  const initialDraft = React.useMemo(
    () => (activeProfile ? deviceProfileToDraft(activeProfile) : buildDraftFromCurrentState(defaultProfileName, undefined)),
    [activeProfile, defaultProfileName],
  );

  const [form, setForm] = React.useState<ProfileFormState>(() => profileToForm(initialDraft));
  const [errors, setErrors] = React.useState<string[]>([]);
  const [dirty, setDirty] = React.useState(false);
  const { isStack } = useResponsiveLayout();

  const { sectionOpen, toggleSection } = useProfileSections(isStack);

  React.useEffect(() => {
    if (!dirty && activeProfile) {
      setForm(profileToForm(deviceProfileToDraft(activeProfile)));
    }
  }, [activeProfile, dirty]);

  const {
    macInputRefs,
    handleMacSegmentChange,
    handleMacSegmentFocus,
    handleMacSegmentMouseDown,
    handleMacSegmentClick,
    handleMacSegmentKeyDown,
    handleMacSegmentPaste,
  } = useMacAddressInput({ macSegments: form.macSegments, setForm, setDirty });

  const handleSelectProfile = React.useCallback(
    (value: string) => {
      if (!value) return;
      if (value === SELECT_NEW_PROFILE) {
        const draft = buildDraftFromCurrentState(newProfileName, undefined);
        const profile = createProfile(draft);
        setActiveProfile(profile.id);
        setForm(profileToForm(deviceProfileToDraft(profile)));
        setDirty(false);
        setErrors([]);
        toast.success(formatProfileCreatedToast(profile.name, locale));
        return;
      }
      if (value === SELECT_IMPORT_CURRENT) {
        const draft = buildDraftFromCurrentState(form.name || importedProfileName, activeProfile ?? undefined);
        setForm(profileToForm(draft));
        setDirty(true);
        setErrors([]);
        toast.success(importedToast);
        return;
      }
      if (value === activeProfile?.id) return;
      setDirty(false);
      setErrors([]);
      setActiveProfile(value);
    },
    [
      activeProfile,
      createProfile,
      form.name,
      setActiveProfile,
      newProfileName,
      importedProfileName,
      importedToast,
      locale,
    ],
  );

  const handleRomVersionChange = React.useCallback((value: ROMVersion) => {
    setForm((prev) => ({
      ...prev,
      romVersion: value,
      memoryLink: enforceMemoryLink(prev.memoryLink, value, prev.withSave),
      shinyCharm: enforceShinyCharm(prev.shinyCharm, value, prev.withSave),
    }));
    setDirty(true);
  }, []);

  const handleRomRegionChange = React.useCallback((value: ROMRegion) => {
    setForm((prev) => ({ ...prev, romRegion: value }));
    setDirty(true);
  }, []);

  const handleHardwareChange = React.useCallback((value: Hardware) => {
    setForm((prev) => ({ ...prev, hardware: value }));
    setDirty(true);
  }, []);

  const handleTimerAutoToggle = React.useCallback((checked: boolean) => {
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
    setDirty(true);
  }, []);

  const handleTimerHexChange = React.useCallback(
    (field: 'timer0Min' | 'timer0Max', value: string) => {
      if (value === '' || TIMER_HEX_PATTERN.test(value)) {
        setForm((prev) => ({ ...prev, [field]: value }));
        setDirty(true);
      }
    },
    [],
  );

  const handleTimerHexBlur = React.useCallback((field: 'timer0Min' | 'timer0Max') => {
    setForm((prev) => {
      const canonical = canonicalizeHex(prev[field], 4);
      if (canonical === prev[field]) {
        return prev;
      }
      return { ...prev, [field]: canonical };
    });
  }, []);

  const handleVCountHexChange = React.useCallback(
    (field: 'vcountMin' | 'vcountMax', value: string) => {
      if (value === '' || VCOUNT_HEX_PATTERN.test(value)) {
        setForm((prev) => ({ ...prev, [field]: value }));
        setDirty(true);
      }
    },
    [],
  );

  const handleVCountHexBlur = React.useCallback((field: 'vcountMin' | 'vcountMax') => {
    setForm((prev) => {
      const canonical = canonicalizeHex(prev[field], 2);
      if (canonical === prev[field]) {
        return prev;
      }
      return { ...prev, [field]: canonical };
    });
  }, []);

  const handleNameChange = React.useCallback((value: string) => {
    setForm((prev) => ({ ...prev, name: value }));
    setDirty(true);
  }, []);

  const handleTidChange = React.useCallback((value: string) => {
    if (/^\d{0,5}$/.test(value)) {
      setForm((prev) => ({ ...prev, tid: value }));
      setDirty(true);
    }
  }, []);

  const handleSidChange = React.useCallback((value: string) => {
    if (/^\d{0,5}$/.test(value)) {
      setForm((prev) => ({ ...prev, sid: value }));
      setDirty(true);
    }
  }, []);

  const handleShinyCharmToggle = React.useCallback((checked: boolean) => {
    setForm((prev) => ({ ...prev, shinyCharm: checked }));
    setDirty(true);
  }, []);

  const handleNewGameToggle = React.useCallback((checked: boolean) => {
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
    setDirty(true);
  }, []);

  const handleWithSaveToggle = React.useCallback((checked: boolean) => {
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
    setDirty(true);
  }, []);

  const handleMemoryLinkToggle = React.useCallback((checked: boolean) => {
    setForm((prev) => ({
      ...prev,
      memoryLink: enforceMemoryLink(checked, prev.romVersion, prev.withSave),
    }));
    setDirty(true);
  }, []);

  const handleDelete = React.useCallback(() => {
    if (!activeProfile) return;
    if (profiles.length <= 1) {
      toast.error(minimumProfileError);
      return;
    }
    if (!window.confirm(formatProfileDeleteConfirm(activeProfile.name, locale))) {
      return;
    }
    deleteProfile(activeProfile.id);
    setErrors([]);
    setDirty(false);
    toast.success(deletedToast);
  }, [activeProfile, deleteProfile, profiles.length, minimumProfileError, deletedToast, locale]);

  const handleSave = React.useCallback(() => {
    if (!activeProfile) return;
    const { draft, validationErrors } = formToDraft(form, locale, {
      timer0Min: timer0MinLabel,
      timer0Max: timer0MaxLabel,
      vcountMin: vcountMinLabel,
      vcountMax: vcountMaxLabel,
      tid: tidLabel,
      sid: sidLabel,
    });
    if (!draft) {
      setErrors(validationErrors);
      return;
    }
    updateProfile(activeProfile.id, draft);
    setErrors([]);
    setDirty(false);
    toast.success(savedToast);
  }, [
    activeProfile,
    form,
    locale,
    updateProfile,
    timer0MinLabel,
    timer0MaxLabel,
    vcountMinLabel,
    vcountMaxLabel,
    tidLabel,
    sidLabel,
    savedToast,
  ]);

  const disableDelete = profiles.length <= 1;
  const memoryLinkDisabled = form.romVersion === 'B' || form.romVersion === 'W' || !form.withSave;
  const withSaveDisabled = !form.newGame;
  const shinyCharmDisabled = form.romVersion === 'B' || form.romVersion === 'W' || !form.withSave;

  const profileSelection: ProfileSelectionControls = {
    profiles,
    activeId: activeProfile?.id ?? '',
    onSelect: handleSelectProfile,
  };

  const header: HeaderControls = {
    dirty,
    profileName: form.name,
    canModify: Boolean(activeProfile),
    disableDelete,
    onProfileNameChange: handleNameChange,
    onSave: handleSave,
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
