import React from 'react';
import { useAppStore } from '@/store/app-store';
import { DEFAULT_GENERATION_DRAFT_PARAMS } from '@/store/generation-store';
import type { BootTimingDraft } from '@/types/generation';
import { keyMaskToNames, keyNamesToMask, KEY_INPUT_DEFAULT, formatKeyInputForDisplay, type KeyName } from '@/lib/utils/key-input';

const DEFAULT_TIME_VALUE = '00:00:00';

interface UseBootTimingDraftOptions {
  disabled: boolean;
  isActive: boolean;
}

export interface BootTimingDialogState {
  isOpen: boolean;
  availableKeys: KeyName[];
  open: () => void;
  close: () => void;
  onOpenChange: (open: boolean) => void;
  toggleKey: (key: KeyName) => void;
  resetKeys: () => void;
  applyKeys: () => void;
  canOpen: boolean;
}

export interface BootTimingSnapshot {
  bootDateValue: string;
  bootTimeValue: string;
  keyDisplay: string;
  timer0RangeDisplay: string;
  vcountRangeDisplay: string;
  macDisplay: string;
  profileSummaryLines: string[];
}

export interface BootTimingDraftController {
  bootTiming: BootTimingDraft;
  version: string;
  isActive: boolean;
  dialog: BootTimingDialogState;
  snapshot: BootTimingSnapshot;
  handleDateInput: (value: string) => void;
  handleTimeInput: (value: string) => void;
}

export function useBootTimingDraft({ disabled, isActive }: UseBootTimingDraftOptions): BootTimingDraftController {
  const bootTiming = useAppStore((state) => state.draftParams.bootTiming ?? DEFAULT_GENERATION_DRAFT_PARAMS.bootTiming);
  const version = useAppStore((state) => state.draftParams.version ?? 'B');
  const setDraftParams = useAppStore((state) => state.setDraftParams);

  const [isKeyDialogOpen, setIsKeyDialogOpen] = React.useState(false);
  const [tempKeyMask, setTempKeyMask] = React.useState(bootTiming.keyMask);

  React.useEffect(() => {
    if (!isKeyDialogOpen) {
      setTempKeyMask(bootTiming.keyMask);
    }
  }, [bootTiming.keyMask, isKeyDialogOpen]);

  React.useEffect(() => {
    if (!isActive && isKeyDialogOpen) {
      setIsKeyDialogOpen(false);
    }
  }, [isActive, isKeyDialogOpen]);

  const updateBootTiming = React.useCallback((partial: Partial<BootTimingDraft>) => {
    setDraftParams({ bootTiming: { ...bootTiming, ...partial } });
  }, [bootTiming, setDraftParams]);

  const { dateValue, timeValue } = React.useMemo(() => toLocalDateTimeParts(bootTiming.timestampIso), [bootTiming.timestampIso]);

  const applyBootTimestamp = React.useCallback((dateValueNext: string, timeValueNext: string) => {
    const isoString = toIsoStringFromLocal(dateValueNext, timeValueNext);
    updateBootTiming({ timestampIso: isoString });
  }, [updateBootTiming]);

  const handleBootDateInput = React.useCallback((value: string) => {
    if (!value) return;
    const nextTime = timeValue || DEFAULT_TIME_VALUE;
    applyBootTimestamp(value, nextTime);
  }, [applyBootTimestamp, timeValue]);

  const handleBootTimeInput = React.useCallback((value: string) => {
    if (!dateValue) return;
    applyBootTimestamp(dateValue, value);
  }, [applyBootTimestamp, dateValue]);

  const bootKeyDisplay = React.useMemo(() => {
    const names = keyMaskToNames(bootTiming.keyMask);
    return formatKeyInputForDisplay(null, names);
  }, [bootTiming.keyMask]);

  const tempAvailableKeys = React.useMemo(() => keyMaskToNames(tempKeyMask), [tempKeyMask]);

  const handleToggleBootKey = React.useCallback((key: KeyName) => {
    setTempKeyMask((prev) => {
      const current = keyMaskToNames(prev);
      const next = current.includes(key)
        ? (current.filter(k => k !== key) as KeyName[])
        : [...current, key];
      return keyNamesToMask(next);
    });
  }, []);

  const handleResetBootKeys = React.useCallback(() => {
    setTempKeyMask(KEY_INPUT_DEFAULT);
  }, []);

  const handleApplyBootKeys = React.useCallback(() => {
    updateBootTiming({ keyMask: tempKeyMask });
    setIsKeyDialogOpen(false);
  }, [tempKeyMask, updateBootTiming]);

  const handleKeyDialogOpenChange = React.useCallback((open: boolean) => {
    if (!isActive) {
      setIsKeyDialogOpen(false);
      return;
    }
    if (disabled && open) {
      return;
    }
    setIsKeyDialogOpen(open);
    if (!open) {
      setTempKeyMask(bootTiming.keyMask);
    }
  }, [bootTiming.keyMask, disabled, isActive]);

  const openKeyDialog = React.useCallback(() => {
    if (disabled || !isActive) return;
    setTempKeyMask(bootTiming.keyMask);
    setIsKeyDialogOpen(true);
  }, [bootTiming.keyMask, disabled, isActive]);

  const timer0RangeDisplay = formatHexRange(bootTiming.timer0Range, 4);
  const vcountRangeDisplay = formatHexRange(bootTiming.vcountRange, 2);
  const macDisplay = formatMacAddress(bootTiming.macAddress);

  const profileSummaryLines = React.useMemo(() => {
    return [
      `${version} (${bootTiming.romRegion}) · ${bootTiming.hardware} · MAC ${macDisplay}`,
      `Timer0 ${timer0RangeDisplay} · VCount ${vcountRangeDisplay}`,
    ];
  }, [bootTiming.hardware, bootTiming.romRegion, macDisplay, timer0RangeDisplay, vcountRangeDisplay, version]);

  const dialogState: BootTimingDialogState = {
    isOpen: isKeyDialogOpen && isActive,
    availableKeys: tempAvailableKeys,
    open: openKeyDialog,
    close: () => setIsKeyDialogOpen(false),
    onOpenChange: handleKeyDialogOpenChange,
    toggleKey: handleToggleBootKey,
    resetKeys: handleResetBootKeys,
    applyKeys: handleApplyBootKeys,
    canOpen: !disabled && isActive,
  };

  return {
    bootTiming,
    version,
    isActive,
    dialog: dialogState,
    snapshot: {
      bootDateValue: dateValue,
      bootTimeValue: timeValue,
      keyDisplay: bootKeyDisplay,
      timer0RangeDisplay,
      vcountRangeDisplay,
      macDisplay,
      profileSummaryLines,
    },
    handleDateInput: handleBootDateInput,
    handleTimeInput: handleBootTimeInput,
  };
}

function toLocalDateTimeParts(iso?: string): { dateValue: string; timeValue: string } {
  if (!iso) return { dateValue: '', timeValue: '' };
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return { dateValue: '', timeValue: '' };
  const pad = (value: number) => value.toString().padStart(2, '0');
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return {
    dateValue: `${year}-${month}-${day}`,
    timeValue: `${hours}:${minutes}:${seconds}`,
  };
}

function normalizeTimeValue(value: string): string | null {
  if (!value) return null;
  if (/^\d{2}:\d{2}:\d{2}$/.test(value)) return value;
  if (/^\d{2}:\d{2}$/.test(value)) return `${value}:00`;
  return null;
}

function toIsoStringFromLocal(dateValue: string, timeValue: string): string | undefined {
  const normalizedTime = normalizeTimeValue(timeValue);
  if (!dateValue || !normalizedTime) return undefined;
  const combined = `${dateValue}T${normalizedTime}`;
  const date = new Date(combined);
  if (Number.isNaN(date.getTime())) {
    return undefined;
  }
  return date.toISOString();
}

function formatHexRange(range: { min: number; max: number }, width: number): string {
  const normalize = (input: number) => Math.max(0, input >>> 0);
  const formatValue = (input: number) => `0x${normalize(input).toString(16).toUpperCase().padStart(width, '0')}`;
  return `${formatValue(range.min)}-${formatValue(range.max)}`;
}

function formatMacAddress(address: readonly [number, number, number, number, number, number]): string {
  return Array.from(address)
    .map((value) => Math.max(0, Math.min(255, value))
      .toString(16)
      .toUpperCase()
      .padStart(2, '0'))
    .join(':');
}
