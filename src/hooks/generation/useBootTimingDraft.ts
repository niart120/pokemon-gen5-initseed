import React from 'react';
import { useAppStore } from '@/store/app-store';
import { DEFAULT_GENERATION_DRAFT_PARAMS } from '@/store/generation-store';
import type { BootTimingDraft } from '@/types/generation';
import { keyMaskToNames, keyNamesToMask, KEY_INPUT_DEFAULT, type KeyName } from '@/lib/utils/key-input';
import { formatKeyInputDisplay } from '@/lib/i18n/strings/search-results';

interface UseBootTimingDraftOptions {
  locale: 'ja' | 'en';
  disabled: boolean;
  isActive: boolean;
  pairsLabel: string;
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
  bootTimestampValue: string;
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
  handleTimestampInput: (value: string) => void;
}

export function useBootTimingDraft({ locale, disabled, isActive, pairsLabel }: UseBootTimingDraftOptions): BootTimingDraftController {
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

  const handleBootTimestampInput = React.useCallback((value: string) => {
    if (!value) {
      updateBootTiming({ timestampIso: undefined });
      return;
    }
    const isoString = toIsoStringFromLocal(value);
    if (isoString) {
      updateBootTiming({ timestampIso: isoString });
    }
  }, [updateBootTiming]);

  const bootKeyDisplay = React.useMemo(() => {
    const names = keyMaskToNames(bootTiming.keyMask);
    return formatKeyInputDisplay(names, locale);
  }, [bootTiming.keyMask, locale]);

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

  const bootTimestampValue = React.useMemo(() => formatDateTimeLocalValue(bootTiming.timestampIso), [bootTiming.timestampIso]);

  const localeTag = locale === 'ja' ? 'ja-JP' : 'en-US';
  const pairCountFormatter = React.useMemo(() => new Intl.NumberFormat(localeTag), [localeTag]);
  const timer0Count = Math.max(0, bootTiming.timer0Range.max - bootTiming.timer0Range.min + 1);
  const vcountCount = Math.max(0, bootTiming.vcountRange.max - bootTiming.vcountRange.min + 1);
  const pairCount = timer0Count * vcountCount;
  const pairCountDisplay = `${pairCountFormatter.format(pairCount)} ${pairsLabel}`;
  const timer0RangeDisplay = formatHexRange(bootTiming.timer0Range, 4);
  const vcountRangeDisplay = formatHexRange(bootTiming.vcountRange, 2);
  const macDisplay = formatMacAddress(bootTiming.macAddress);

  const profileSummaryLines = React.useMemo(() => {
    return [
      `${version} (${bootTiming.romRegion}) · ${bootTiming.hardware}`,
      `MAC ${macDisplay}`,
      `Timer0 ${timer0RangeDisplay} · VCount ${vcountRangeDisplay}`,
      `Timer0×VCount ${pairCountDisplay}`,
    ];
  }, [bootTiming.hardware, bootTiming.romRegion, macDisplay, pairCountDisplay, timer0RangeDisplay, vcountRangeDisplay, version]);

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
      bootTimestampValue,
      keyDisplay: bootKeyDisplay,
      timer0RangeDisplay,
      vcountRangeDisplay,
      macDisplay,
      profileSummaryLines,
    },
    handleTimestampInput: handleBootTimestampInput,
  };
}

function formatDateTimeLocalValue(iso?: string): string {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  const pad = (value: number) => value.toString().padStart(2, '0');
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
}

function toIsoStringFromLocal(value: string): string | undefined {
  const date = new Date(value);
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
