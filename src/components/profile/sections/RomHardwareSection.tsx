import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveProfileRomLabel, formatProfileMacSegmentAria } from '@/lib/i18n/strings/profile-rom-hardware';

const MAC_SEGMENT_CLASS =
  'h-9 w-7 min-w-[1rem] rounded-md border border-input bg-muted/40 px-1 text-center font-mono uppercase tracking-tight shadow-xs focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/60';

interface HardwareOption {
  value: Hardware;
  label: string;
}

interface RomHardwareSectionProps {
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  macSegments: string[];
  romVersions: ROMVersion[];
  romRegions: ROMRegion[];
  hardwareOptions: HardwareOption[];
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
  disabled?: boolean;
}

export function RomHardwareSection({
  romVersion,
  romRegion,
  hardware,
  macSegments,
  romVersions,
  romRegions,
  hardwareOptions,
  macInputRefs,
  onRomVersionChange,
  onRomRegionChange,
  onHardwareChange,
  onMacSegmentChange,
  onMacSegmentFocus,
  onMacSegmentMouseDown,
  onMacSegmentClick,
  onMacSegmentKeyDown,
  onMacSegmentPaste,
  disabled = false,
}: RomHardwareSectionProps) {
  const locale = useLocale();

  return (
    <div className="flex flex-wrap items-start gap-2 sm:gap-3">
      <div className="flex flex-col gap-1 flex-none">
        <Label htmlFor="profile-rom-version" className="text-xs">
          {resolveProfileRomLabel('version', locale)}
        </Label>
        <Select value={romVersion} onValueChange={(value) => onRomVersionChange(value as ROMVersion)} disabled={disabled}>
          <SelectTrigger id="profile-rom-version" className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {romVersions.map((version) => (
              <SelectItem key={version} value={version}>{version}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1 flex-none">
        <Label htmlFor="profile-rom-region" className="text-xs">
          {resolveProfileRomLabel('region', locale)}
        </Label>
        <Select value={romRegion} onValueChange={(value) => onRomRegionChange(value as ROMRegion)} disabled={disabled}>
          <SelectTrigger id="profile-rom-region" className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {romRegions.map((region) => (
              <SelectItem key={region} value={region}>{region}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1 flex-none">
        <Label htmlFor="profile-hardware" className="text-xs">
          {resolveProfileRomLabel('hardware', locale)}
        </Label>
        <Select value={hardware} onValueChange={(value) => onHardwareChange(value as Hardware)} disabled={disabled}>
          <SelectTrigger id="profile-hardware" className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {hardwareOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1 flex-1">
        <Label className="text-xs">
          {resolveProfileRomLabel('macAddress', locale)}
        </Label>
        <div className="flex gap-1.5 overflow-x-auto pb-1 sm:overflow-visible">
          {macSegments.map((segment, index) => (
            <input
              key={index}
              ref={(element) => {
                macInputRefs.current[index] = element;
              }}
              value={segment}
              onChange={(event) => onMacSegmentChange(index, event.target.value)}
              onFocus={onMacSegmentFocus}
              onMouseDown={onMacSegmentMouseDown}
              onClick={onMacSegmentClick}
              onKeyDown={(event) => onMacSegmentKeyDown(index, event)}
              onPaste={(event) => onMacSegmentPaste(index, event)}
              inputMode="text"
              autoComplete="off"
              spellCheck={false}
              maxLength={2}
              className={MAC_SEGMENT_CLASS}
              aria-label={formatProfileMacSegmentAria(index, locale)}
              disabled={disabled}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
