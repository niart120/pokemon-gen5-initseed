import React from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { DeviceMobileSpeaker, CaretDown } from '@phosphor-icons/react';
import { toast } from 'sonner';
import { useAppStore } from '@/store/app-store';
import type { DeviceProfile, DeviceProfileDraft } from '@/types/profile';
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import { deviceProfileToDraft } from '@/types/profile';
import { formatHexDisplay, parseHexInput, parseMacByte } from '@/lib/utils/hex-parser';
import { getFullTimer0Range, getValidVCounts } from '@/lib/utils/rom-parameter-helpers';
import { useResponsiveLayout } from '@/hooks/use-mobile';

const ROM_VERSIONS: ROMVersion[] = ['B', 'W', 'B2', 'W2'];
const ROM_REGIONS: ROMRegion[] = ['JPN', 'KOR', 'USA', 'GER', 'FRA', 'SPA', 'ITA'];
const HARDWARE_OPTIONS: { value: Hardware; label: string }[] = [
  { value: 'DS', label: 'DS' },
  { value: 'DS_LITE', label: 'DS Lite' },
  { value: '3DS', label: '3DS' },
];

const TIMER_HEX_PATTERN = /^0x?[0-9a-fA-F]{0,4}$/;
const VCOUNT_HEX_PATTERN = /^0x?[0-9a-fA-F]{0,2}$/;
const SELECT_NEW_PROFILE = '__new_profile__';
const SELECT_IMPORT_CURRENT = '__import_current__';

interface ProfileFormState {
  name: string;
  description: string;
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  timer0Auto: boolean;
  timer0Min: string;
  timer0Max: string;
  vcountMin: string;
  vcountMax: string;
  macSegments: string[];
  tid: string;
  sid: string;
  shinyCharm: boolean;
  newGame: boolean;
  withSave: boolean;
  memoryLink: boolean;
}

type SectionKey = 'rom' | 'timer' | 'game';
type SectionState = Record<SectionKey, boolean>;

export function ProfilePanel() {
  const profiles = useAppStore((state) => state.profiles);
  const activeProfileId = useAppStore((state) => state.activeProfileId);
  const setActiveProfile = useAppStore((state) => state.setActiveProfile);
  const createProfile = useAppStore((state) => state.createProfile);
  const updateProfile = useAppStore((state) => state.updateProfile);
  const deleteProfile = useAppStore((state) => state.deleteProfile);
  const applyProfileToSearch = useAppStore((state) => state.applyProfileToSearch);
  const applyProfileToGeneration = useAppStore((state) => state.applyProfileToGeneration);

  const activeProfile = React.useMemo(() => resolveActiveProfile(profiles, activeProfileId), [profiles, activeProfileId]);
  const initialDraft = activeProfile
    ? deviceProfileToDraft(activeProfile)
    : buildDraftFromCurrentState('Default Device', undefined);

  const [form, setForm] = React.useState<ProfileFormState>(() => profileToForm(initialDraft));
  const [errors, setErrors] = React.useState<string[]>([]);
  const [dirty, setDirty] = React.useState(false);
  const { isStack } = useResponsiveLayout();

  const sectionDefaults = React.useMemo<SectionState>(
    () => (isStack ? { rom: true, timer: false, game: false } : { rom: true, timer: true, game: true }),
    [isStack]
  );
  const [sectionOpen, setSectionOpen] = React.useState<SectionState>(sectionDefaults);

  React.useEffect(() => {
    setSectionOpen(sectionDefaults);
  }, [sectionDefaults]);

  const toggleSection = (key: SectionKey) => {
    setSectionOpen((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const renderCollapsedSection = (
    key: SectionKey,
    headingId: string,
    title: string,
    content: React.ReactNode,
  ) => {
    const isOpen = sectionOpen[key];
    return (
      <section
        key={key}
        aria-labelledby={headingId}
        className="rounded-xl border bg-card/60 px-3 py-2"
      >
        <button
          type="button"
          className="flex w-full items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground"
          aria-expanded={isOpen}
          aria-controls={`${headingId}-content`}
          onClick={() => toggleSection(key)}
        >
          <span id={headingId}>{title}</span>
          <CaretDown size={16} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>
        <div
          id={`${headingId}-content`}
          className={isOpen ? 'mt-3 space-y-3' : 'hidden'}
        >
          {content}
        </div>
      </section>
    );
  };

  const disableDelete = profiles.length <= 1;
  const memoryLinkDisabled = form.romVersion === 'B' || form.romVersion === 'W' || !form.withSave;
  const withSaveDisabled = !form.newGame;
  const shinyCharmDisabled = form.romVersion === 'B' || form.romVersion === 'W' || !form.withSave;

  React.useEffect(() => {
    if (!dirty && activeProfile) {
      setForm(profileToForm(deviceProfileToDraft(activeProfile)));
    }
  }, [activeProfile, dirty]);

  const handleSelectProfile = (value: string) => {
    if (!value) return;
    if (value === SELECT_NEW_PROFILE) {
      const draft = buildDraftFromCurrentState('New Profile', undefined);
      const profile = createProfile(draft);
      setActiveProfile(profile.id);
      setForm(profileToForm(deviceProfileToDraft(profile)));
      setDirty(false);
      setErrors([]);
      toast.success(`Profile "${profile.name}" created`);
      return;
    }
    if (value === SELECT_IMPORT_CURRENT) {
      const draft = buildDraftFromCurrentState(form.name || 'Imported Profile', activeProfile ?? undefined);
      setForm(profileToForm(draft));
      setDirty(true);
      setErrors([]);
      toast.success('Current settings imported into form');
      return;
    }
    if (value === activeProfile?.id) return;
    setDirty(false);
    setErrors([]);
    setActiveProfile(value);
  };

  const handleRomVersionChange = (value: ROMVersion) => {
    setForm((prev) => ({
      ...prev,
      romVersion: value,
      memoryLink: enforceMemoryLink(prev.memoryLink, value, prev.withSave),
      shinyCharm: enforceShinyCharm(prev.shinyCharm, value, prev.withSave),
    }));
    setDirty(true);
  };

  const handleRomRegionChange = (value: ROMRegion) => {
    setForm((prev) => ({ ...prev, romRegion: value }));
    setDirty(true);
  };

  const handleHardwareChange = (value: Hardware) => {
    setForm((prev) => ({ ...prev, hardware: value }));
    setDirty(true);
  };

  const handleTimerAutoToggle = (checked: boolean) => {
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
  };

  const handleTimerHexChange = (field: 'timer0Min' | 'timer0Max', value: string) => {
    if (value === '' || TIMER_HEX_PATTERN.test(value)) {
      setForm((prev) => ({ ...prev, [field]: value }));
      setDirty(true);
    }
  };

  const handleTimerHexBlur = (field: 'timer0Min' | 'timer0Max') => {
    setForm((prev) => {
      const canonical = canonicalizeHex(prev[field], 4);
      if (canonical === prev[field]) {
        return prev;
      }
      return { ...prev, [field]: canonical };
    });
  };

  const handleVCountHexChange = (field: 'vcountMin' | 'vcountMax', value: string) => {
    if (value === '' || VCOUNT_HEX_PATTERN.test(value)) {
      setForm((prev) => ({ ...prev, [field]: value }));
      setDirty(true);
    }
  };

  const handleVCountHexBlur = (field: 'vcountMin' | 'vcountMax') => {
    setForm((prev) => {
      const canonical = canonicalizeHex(prev[field], 2);
      if (canonical === prev[field]) {
        return prev;
      }
      return { ...prev, [field]: canonical };
    });
  };

  const macInputRefs = React.useRef<Array<HTMLInputElement | null>>([]);

  const focusMacSegment = React.useCallback((index: number) => {
    const field = macInputRefs.current[index];
    if (field) {
      field.focus();
      field.select();
    }
  }, []);

  const handleMacSegmentChange = (index: number, rawValue: string) => {
    const sanitized = rawValue.replace(/[^0-9a-fA-F]/g, '').toUpperCase().slice(0, 2);
    const currentValue = form.macSegments[index];
    if (sanitized !== currentValue) {
      setForm((prev) => {
        const nextSegments = [...prev.macSegments];
        nextSegments[index] = sanitized;
        return { ...prev, macSegments: nextSegments };
      });
      setDirty(true);
    }
    if (sanitized.length === 2 && currentValue.length < 2 && index < form.macSegments.length - 1) {
      focusMacSegment(index + 1);
    }
  };

  const handleMacSegmentFocus = (event: React.FocusEvent<HTMLInputElement>) => {
    event.target.select();
  };

  const handleMacSegmentMouseDown = (event: React.MouseEvent<HTMLInputElement>) => {
    if (event.currentTarget === document.activeElement) {
      event.preventDefault();
      event.currentTarget.select();
    }
  };

  const handleMacSegmentClick = (event: React.MouseEvent<HTMLInputElement>) => {
    event.currentTarget.select();
  };

  const handleMacSegmentKeyDown = (index: number, event: React.KeyboardEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const selectionStart = input.selectionStart ?? 0;
    const selectionEnd = input.selectionEnd ?? 0;
    if (event.key === 'ArrowLeft' && selectionStart === 0 && selectionEnd === 0 && index > 0) {
      event.preventDefault();
      focusMacSegment(index - 1);
    }
    if (event.key === 'ArrowRight' && selectionStart === input.value.length && selectionEnd === input.value.length && index < form.macSegments.length - 1) {
      event.preventDefault();
      focusMacSegment(index + 1);
    }
  };

  const handleMacSegmentPaste = (index: number, event: React.ClipboardEvent<HTMLInputElement>) => {
    const pasted = event.clipboardData.getData('text');
    const sanitized = pasted.replace(/[^0-9a-fA-F]/g, '').toUpperCase();
    if (!sanitized) {
      return;
    }
    event.preventDefault();
    const segmentCapacity = (form.macSegments.length - index) * 2;
    const usableDigits = Math.min(sanitized.length, segmentCapacity);
    if (usableDigits === 0) {
      return;
    }
    let changed = false;
    setForm((prev) => {
      const nextSegments = [...prev.macSegments];
      let cursor = 0;
      for (let i = index; i < nextSegments.length && cursor < usableDigits; i += 1) {
        const segmentValue = sanitized.slice(cursor, cursor + 2);
        if (segmentValue !== nextSegments[i]) {
          nextSegments[i] = segmentValue;
          changed = true;
        }
        cursor += 2;
      }
      if (!changed) {
        return prev;
      }
      return { ...prev, macSegments: nextSegments };
    });
    if (changed) {
      setDirty(true);
      const segmentsAdvanced = Math.floor(usableDigits / 2);
      const hasRemainder = usableDigits % 2 !== 0;
      let targetIndex = index + segmentsAdvanced;
      if (!hasRemainder && usableDigits > 0 && targetIndex < form.macSegments.length - 1) {
        targetIndex += 1;
      }
      if (targetIndex >= form.macSegments.length) {
        targetIndex = form.macSegments.length - 1;
      }
      focusMacSegment(targetIndex);
    }
  };

  const handleNameChange = (value: string) => {
    setForm((prev) => ({ ...prev, name: value }));
    setDirty(true);
  };

  const handleTidChange = (value: string) => {
    if (/^\d{0,5}$/.test(value)) {
      setForm((prev) => ({ ...prev, tid: value }));
      setDirty(true);
    }
  };

  const handleSidChange = (value: string) => {
    if (/^\d{0,5}$/.test(value)) {
      setForm((prev) => ({ ...prev, sid: value }));
      setDirty(true);
    }
  };

  const handleShinyCharmToggle = (checked: boolean) => {
    setForm((prev) => ({ ...prev, shinyCharm: checked }));
    setDirty(true);
  };

  const handleNewGameToggle = (checked: boolean) => {
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
  };

  const handleWithSaveToggle = (checked: boolean) => {
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
  };

  const handleMemoryLinkToggle = (checked: boolean) => {
    setForm((prev) => ({ ...prev, memoryLink: enforceMemoryLink(checked, prev.romVersion, prev.withSave) }));
    setDirty(true);
  };

  const handleDelete = () => {
    if (!activeProfile) return;
    if (profiles.length <= 1) {
      toast.error('少なくとも1件のプロファイルが必要です');
      return;
    }
    if (!window.confirm(`プロファイル「${activeProfile.name}」を削除しますか?`)) {
      return;
    }
    deleteProfile(activeProfile.id);
    setErrors([]);
    setDirty(false);
    toast.success('Profile deleted');
  };

  const handleSave = () => {
    if (!activeProfile) return;
    const { draft, validationErrors } = formToDraft(form);
    if (!draft) {
      setErrors(validationErrors);
      return;
    }
    updateProfile(activeProfile.id, draft);
    setErrors([]);
    setDirty(false);
    toast.success('Profile saved');
  };

  const handleLoad = () => {
    if (!activeProfile) return;
    applyProfileToSearch(activeProfile.id);
    applyProfileToGeneration(activeProfile.id);
    toast.success('Profile loaded');
  };

  const romSectionContent = (
    <div className="flex flex-wrap items-start gap-2 sm:gap-3">
      <div className="flex flex-col gap-1 flex-none">
        <Label htmlFor="profile-rom-version" className="text-xs">Version</Label>
        <Select value={form.romVersion} onValueChange={(value) => handleRomVersionChange(value as ROMVersion)}>
          <SelectTrigger id="profile-rom-version" className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ROM_VERSIONS.map((version) => (
              <SelectItem key={version} value={version}>{version}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1 flex-none">
        <Label htmlFor="profile-rom-region" className="text-xs">Region</Label>
        <Select value={form.romRegion} onValueChange={(value) => handleRomRegionChange(value as ROMRegion)}>
          <SelectTrigger id="profile-rom-region" className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ROM_REGIONS.map((region) => (
              <SelectItem key={region} value={region}>{region}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1 flex-none">
        <Label htmlFor="profile-hardware" className="text-xs">Hardware</Label>
        <Select value={form.hardware} onValueChange={(value) => handleHardwareChange(value as Hardware)}>
          <SelectTrigger id="profile-hardware" className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {HARDWARE_OPTIONS.map((hardware) => (
              <SelectItem key={hardware.value} value={hardware.value}>{hardware.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1 flex-1">
        <Label className="text-xs">MAC Address</Label>
        <div className="flex gap-1.5 overflow-x-auto pb-1 sm:overflow-visible">
          {form.macSegments.map((segment, index) => (
            <input
              key={index}
              ref={(element) => {
                macInputRefs.current[index] = element;
              }}
              value={segment}
              onChange={(event) => handleMacSegmentChange(index, event.target.value)}
              onFocus={handleMacSegmentFocus}
              onMouseDown={handleMacSegmentMouseDown}
              onClick={handleMacSegmentClick}
              onKeyDown={(event) => handleMacSegmentKeyDown(index, event)}
              onPaste={(event) => handleMacSegmentPaste(index, event)}
              inputMode="text"
              autoComplete="off"
              spellCheck={false}
              maxLength={2}
              className="h-9 w-7 min-w-[1rem] rounded-md border border-input bg-muted/40 px-1 text-center text-[11px] font-mono uppercase tracking-tight shadow-xs focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/60"
              aria-label={`MAC segment ${index + 1}`}
            />
          ))}
        </div>
      </div>
    </div>
  );

  const timerAutoToggle = (
    <div className="flex items-center gap-2">
      <Checkbox id="profile-timer-auto" checked={form.timer0Auto} onCheckedChange={(checked) => handleTimerAutoToggle(Boolean(checked))} />
      <Label htmlFor="profile-timer-auto" className="text-xs">Auto</Label>
    </div>
  );

  const timerGridContent = (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
      <div className="flex flex-col gap-1 min-w-0">
        <Label htmlFor="timer0-min" className="text-xs">Timer0 Min</Label>
        <Input
          id="timer0-min"
          value={form.timer0Min}
          onChange={(e) => handleTimerHexChange('timer0Min', e.target.value)}
          onBlur={() => handleTimerHexBlur('timer0Min')}
          disabled={form.timer0Auto}
          className="font-mono text-xs h-9 w-full min-w-0 px-2"
          placeholder="0x0"
        />
      </div>
      <div className="flex flex-col gap-1 min-w-0">
        <Label htmlFor="timer0-max" className="text-xs">Timer0 Max</Label>
        <Input
          id="timer0-max"
          value={form.timer0Max}
          onChange={(e) => handleTimerHexChange('timer0Max', e.target.value)}
          onBlur={() => handleTimerHexBlur('timer0Max')}
          disabled={form.timer0Auto}
          className="font-mono text-xs h-9 w-full min-w-0 px-2"
          placeholder="0x0"
        />
      </div>
      <div className="flex flex-col gap-1 min-w-0">
        <Label htmlFor="vcount-min" className="text-xs">VCount Min</Label>
        <Input
          id="vcount-min"
          value={form.vcountMin}
          onChange={(e) => handleVCountHexChange('vcountMin', e.target.value)}
          onBlur={() => handleVCountHexBlur('vcountMin')}
          disabled={form.timer0Auto}
          className="font-mono text-xs h-9 w-full min-w-0 px-2"
          placeholder="0x0"
        />
      </div>
      <div className="flex flex-col gap-1 min-w-0">
        <Label htmlFor="vcount-max" className="text-xs">VCount Max</Label>
        <Input
          id="vcount-max"
          value={form.vcountMax}
          onChange={(e) => handleVCountHexChange('vcountMax', e.target.value)}
          onBlur={() => handleVCountHexBlur('vcountMax')}
          disabled={form.timer0Auto}
          className="font-mono text-xs h-9 w-full min-w-0 px-2"
          placeholder="0x0"
        />
      </div>
    </div>
  );

  const collapsedTimerContent = (
    <>
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Timer0 / VCount</div>
        {timerAutoToggle}
      </div>
      {timerGridContent}
    </>
  );

  const gameSectionContent = (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 items-end">
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-tid" className="text-xs">TID</Label>
        <Input id="profile-tid" value={form.tid} onChange={(e) => handleTidChange(e.target.value)} placeholder="00000" inputMode="numeric" className="h-9" />
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-sid" className="text-xs">SID</Label>
        <Input id="profile-sid" value={form.sid} onChange={(e) => handleSidChange(e.target.value)} placeholder="00000" inputMode="numeric" className="h-9" />
      </div>
      <div className="flex items-center gap-2">
        <Checkbox id="profile-new-game" checked={form.newGame} onCheckedChange={(checked) => handleNewGameToggle(Boolean(checked))} />
        <Label htmlFor="profile-new-game" className="text-xs">New Game</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-with-save"
          checked={form.withSave}
          onCheckedChange={(checked) => handleWithSaveToggle(Boolean(checked))}
          disabled={withSaveDisabled}
        />
        <Label htmlFor="profile-with-save" className="text-xs">With Save</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-shiny"
          checked={form.shinyCharm}
          onCheckedChange={(checked) => handleShinyCharmToggle(Boolean(checked))}
          disabled={shinyCharmDisabled}
        />
        <Label htmlFor="profile-shiny" className="text-xs">Shiny Charm</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-memory-link"
          checked={form.memoryLink}
          onCheckedChange={(checked) => handleMemoryLinkToggle(Boolean(checked))}
          disabled={memoryLinkDisabled}
        />
        <Label htmlFor="profile-memory-link" className="text-xs">Memory Link</Label>
      </div>
    </div>
  );

  return (
    <Card className="py-3">
      <StandardCardHeader
        icon={<DeviceMobileSpeaker size={20} className="opacity-80" />}
        title="Device Profile"
      />
      <StandardCardContent className="space-y-4">
        {errors.length > 0 && (
          <Alert variant="destructive" className="mb-3">
            <AlertDescription>
              <ul className="list-disc list-inside space-y-1 text-xs">
                {errors.map((error) => (
                  <li key={error}>{error}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1 min-w-[12rem] sm:w-56">
            <Label htmlFor="profile-select" className="text-xs">Profile</Label>
            <Select value={activeProfile?.id ?? ''} onValueChange={handleSelectProfile}>
              <SelectTrigger id="profile-select" className="h-9">
                <SelectValue placeholder="プロファイルを選択" />
              </SelectTrigger>
              <SelectContent>
                {profiles.map((profile) => (
                  <SelectItem key={profile.id} value={profile.id}>
                    {profile.name}
                  </SelectItem>
                ))}
                <div className="my-1 h-px bg-border" role="separator" />
                <SelectItem value={SELECT_NEW_PROFILE}>+ New profile</SelectItem>
                <SelectItem value={SELECT_IMPORT_CURRENT}>Import current settings</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1 flex-1 min-w-[12rem] sm:w-64">
            <Label htmlFor="profile-name" className="text-xs">Profile Name</Label>
            <Input id="profile-name" value={form.name} onChange={(e) => handleNameChange(e.target.value)} placeholder="My DS profile" className="h-9" />
          </div>
          <div className="flex items-center gap-2 ml-auto">
            {dirty && <Badge variant="secondary">未保存</Badge>}
            <Button size="sm" variant="outline" onClick={handleSave} disabled={!activeProfile}>Save</Button>
            <Button size="sm" variant="outline" onClick={handleLoad} disabled={!activeProfile}>Load</Button>
            <Button size="sm" variant="destructive" onClick={handleDelete} disabled={disableDelete}>Delete</Button>
          </div>
        </div>

        <Separator />

        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(22rem,1.2fr)]">
          {isStack ? (
            <>
              {renderCollapsedSection('rom', 'profile-rom', 'ROM & Hardware', romSectionContent)}
              {renderCollapsedSection('timer', 'profile-timer', 'Timer0 / VCount', collapsedTimerContent)}
              {renderCollapsedSection('game', 'profile-game', 'Game State', gameSectionContent)}
            </>
          ) : (
            <>
              <section className="space-y-3" aria-labelledby="profile-rom">
                <h4 id="profile-rom" className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">ROM & Hardware</h4>
                {romSectionContent}
              </section>
              <section className="space-y-3" aria-labelledby="profile-timer">
                <div className="flex items-center justify-between">
                  <h4 id="profile-timer" className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">Timer0 / VCount</h4>
                  {timerAutoToggle}
                </div>
                {timerGridContent}
              </section>
              <section className="space-y-3" aria-labelledby="profile-game">
                <h4 id="profile-game" className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">Game State</h4>
                {gameSectionContent}
              </section>
            </>
          )}
        </div>
      </StandardCardContent>
    </Card>
  );
}

function resolveActiveProfile(profiles: DeviceProfile[], activeId: string | null): DeviceProfile | null {
  if (!profiles.length) return null;
  if (activeId) {
    const found = profiles.find((profile) => profile.id === activeId);
    if (found) return found;
  }
  return profiles[0];
}

function enforceMemoryLink(current: boolean, version: ROMVersion, hasSave: boolean): boolean {
  if (version === 'B' || version === 'W') return false;
  if (!hasSave) return false;
  return current;
}

function enforceShinyCharm(current: boolean, version: ROMVersion, hasSave: boolean): boolean {
  if (version === 'B' || version === 'W') return false;
  if (!hasSave) return false;
  return current;
}

function canonicalizeHex(value: string, minDigits: number): string {
  const trimmed = value.trim();
  if (!trimmed) return value;
  const parsed = parseHexInput(trimmed);
  if (parsed === null) return value;
  return `0x${formatHexDisplay(parsed, minDigits)}`;
}

function toTimerHex(value: number): string {
  return `0x${formatHexDisplay(value, 4)}`;
}

function toVCountHex(value: number): string {
  return `0x${formatHexDisplay(value, 2)}`;
}

function profileToForm(draft: DeviceProfileDraft): ProfileFormState {
  const withSave = draft.newGame ? draft.withSave : true;
  return {
    name: draft.name,
    description: draft.description ?? '',
    romVersion: draft.romVersion,
    romRegion: draft.romRegion,
    hardware: draft.hardware,
    timer0Auto: draft.timer0Auto,
    timer0Min: toTimerHex(draft.timer0Range.min),
    timer0Max: toTimerHex(draft.timer0Range.max),
    vcountMin: toVCountHex(draft.vcountRange.min),
    vcountMax: toVCountHex(draft.vcountRange.max),
    macSegments: draft.macAddress.map((byte) => formatHexDisplay(byte, 2)),
    tid: String(draft.tid),
    sid: String(draft.sid),
    shinyCharm: enforceShinyCharm(draft.shinyCharm, draft.romVersion, withSave),
    newGame: draft.newGame,
    withSave,
    memoryLink: enforceMemoryLink(draft.memoryLink, draft.romVersion, withSave),
  };
}

function formToDraft(form: ProfileFormState): { draft: DeviceProfileDraft | null; validationErrors: string[] } {
  const errors: string[] = [];
  if (!form.name.trim()) {
    errors.push('Profile name is required');
  }

  const timer0Min = parseAndValidateHex(form.timer0Min, 'Timer0 min', 0xffff, errors);
  const timer0Max = parseAndValidateHex(form.timer0Max, 'Timer0 max', 0xffff, errors);
  const vcountMin = parseAndValidateHex(form.vcountMin, 'VCount min', 0xff, errors);
  const vcountMax = parseAndValidateHex(form.vcountMax, 'VCount max', 0xff, errors);

  if (timer0Min !== null && timer0Max !== null && timer0Min > timer0Max) {
    errors.push('Timer0 min must be less than or equal to max');
  }
  if (vcountMin !== null && vcountMax !== null && vcountMin > vcountMax) {
    errors.push('VCount min must be less than or equal to max');
  }

  const macAddress: number[] = [];
  form.macSegments.forEach((segment, index) => {
    const parsed = parseMacByte(segment);
    if (parsed === null) {
      errors.push(`MAC segment ${index + 1} is invalid`);
    } else {
      macAddress.push(parsed);
    }
  });

  const tid = parseInteger(form.tid, 'TID', errors);
  const sid = parseInteger(form.sid, 'SID', errors);

  if (macAddress.length !== 6) {
    errors.push('MAC address must contain six segments');
  }

  if (errors.length > 0 || timer0Min === null || timer0Max === null || vcountMin === null || vcountMax === null || tid === null || sid === null) {
    return { draft: null, validationErrors: errors };
  }

  const normalizedWithSave = form.newGame ? form.withSave : true;

  const draft: DeviceProfileDraft = {
    name: form.name.trim(),
    description: form.description.trim() ? form.description.trim() : undefined,
    romVersion: form.romVersion,
    romRegion: form.romRegion,
    hardware: form.hardware,
    timer0Auto: form.timer0Auto,
    timer0Range: { min: timer0Min, max: timer0Max },
    vcountRange: { min: vcountMin, max: vcountMax },
    macAddress,
    tid,
    sid,
    shinyCharm: enforceShinyCharm(form.shinyCharm, form.romVersion, normalizedWithSave),
    newGame: form.newGame,
    withSave: normalizedWithSave,
    memoryLink: enforceMemoryLink(form.memoryLink, form.romVersion, normalizedWithSave),
  };

  return { draft, validationErrors: errors };
}

function parseAndValidateHex(value: string, label: string, max: number, errors: string[]): number | null {
  const parsed = parseHexInput(value, max);
  if (parsed === null) {
    errors.push(`${label} must be a hexadecimal value`);
    return null;
  }
  return parsed;
}

function parseInteger(value: string, label: string, errors: string[]): number | null {
  if (!value.trim()) {
    errors.push(`${label} is required`);
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 65535) {
    errors.push(`${label} must be between 0 and 65535`);
    return null;
  }
  return parsed;
}

function buildDraftFromCurrentState(label: string, base: DeviceProfile | undefined): DeviceProfileDraft {
  const state = useAppStore.getState();
  const { searchConditions } = state;
  const generationDraft = state.draftParams ?? {};
  const macAddress = Array.isArray(searchConditions.macAddress)
    ? searchConditions.macAddress.slice(0, 6)
    : [];
  while (macAddress.length < 6) macAddress.push(0);

  const romVersion = searchConditions.romVersion;
  const newGame = Boolean(generationDraft.newGame);
  const withSave = newGame ? Boolean(generationDraft.withSave) : true;
  const memoryLink = enforceMemoryLink(Boolean(generationDraft.memoryLink), romVersion, withSave);
  const shinyCharm = enforceShinyCharm(Boolean(generationDraft.shinyCharm), romVersion, withSave);

  return {
    name: base?.name ?? label,
    description: base?.description,
    romVersion,
    romRegion: searchConditions.romRegion,
    hardware: searchConditions.hardware,
    timer0Auto: searchConditions.timer0VCountConfig.useAutoConfiguration,
    timer0Range: {
      min: searchConditions.timer0VCountConfig.timer0Range.min,
      max: searchConditions.timer0VCountConfig.timer0Range.max,
    },
    vcountRange: {
      min: searchConditions.timer0VCountConfig.vcountRange.min,
      max: searchConditions.timer0VCountConfig.vcountRange.max,
    },
    macAddress,
    tid: typeof generationDraft.tid === 'number' ? generationDraft.tid : 0,
    sid: typeof generationDraft.sid === 'number' ? generationDraft.sid : 0,
    shinyCharm,
    newGame,
    withSave,
    memoryLink,
  };
}
