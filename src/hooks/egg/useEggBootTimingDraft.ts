/**
 * Egg Boot-Timing Draft Hook
 * 
 * Hook for managing boot-timing parameters in Egg Generation Panel.
 * Wraps egg-store with BootTimingControls-compatible interface.
 */

import React from 'react';
import { useEggStore } from '@/store/egg-store';
import { createDefaultEggBootTimingDraft, type EggBootTimingDraft } from '@/types/egg';
import { keyMaskToNames, keyNamesToMask, KEY_INPUT_DEFAULT, type KeyName } from '@/lib/utils/key-input';
import { formatKeyInputDisplay } from '@/lib/i18n/strings/search-results';

interface UseEggBootTimingDraftOptions {
  locale: 'ja' | 'en';
  disabled: boolean;
  isActive: boolean;
}

export interface EggBootTimingDialogState {
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

export interface EggBootTimingSnapshot {
  bootTimestampValue: string;
  keyDisplay: string;
  timer0RangeDisplay: string;
  vcountRangeDisplay: string;
  macDisplay: string;
  profileSummaryLines: string[];
}

export interface EggBootTimingDraftController {
  bootTiming: EggBootTimingDraft;
  isActive: boolean;
  dialog: EggBootTimingDialogState;
  snapshot: EggBootTimingSnapshot;
  handleTimestampInput: (value: string) => void;
}

export function useEggBootTimingDraft({ locale, disabled, isActive }: UseEggBootTimingDraftOptions): EggBootTimingDraftController {
  const bootTiming = useEggStore((state) => state.draftParams.bootTiming ?? createDefaultEggBootTimingDraft());
  const gameMode = useEggStore((state) => state.draftParams.gameMode);
  const updateDraftBootTiming = useEggStore((state) => state.updateDraftBootTiming);

  const [isKeyDialogOpen, setIsKeyDialogOpen] = React.useState(false);
  const [tempKeyMask, setTempKeyMask] = React.useState(bootTiming.keyMask);

  // Sync temp key mask when dialog closes or boot timing changes
  React.useEffect(() => {
    if (!isKeyDialogOpen) {
      setTempKeyMask(bootTiming.keyMask);
    }
  }, [bootTiming.keyMask, isKeyDialogOpen]);

  // Close dialog when mode becomes inactive
  React.useEffect(() => {
    if (!isActive && isKeyDialogOpen) {
      setIsKeyDialogOpen(false);
    }
  }, [isActive, isKeyDialogOpen]);

  const handleBootTimestampInput = React.useCallback((value: string) => {
    if (!value) {
      updateDraftBootTiming({ timestampIso: undefined });
      return;
    }
    const isoString = toIsoStringFromLocal(value);
    if (isoString) {
      updateDraftBootTiming({ timestampIso: isoString });
    }
  }, [updateDraftBootTiming]);

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
    updateDraftBootTiming({ keyMask: tempKeyMask });
    setIsKeyDialogOpen(false);
  }, [tempKeyMask, updateDraftBootTiming]);

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

  const timer0RangeDisplay = formatHexRange(bootTiming.timer0Range, 4);
  const vcountRangeDisplay = formatHexRange(bootTiming.vcountRange, 2);
  const macDisplay = formatMacAddress(bootTiming.macAddress);

  // GameMode から ROM Version を導出
  const version = deriveVersionFromGameMode(gameMode);

  const profileSummaryLines = React.useMemo(() => {
    return [
      `${version} (${bootTiming.romRegion}) · ${bootTiming.hardware} · MAC ${macDisplay}`,
      `Timer0 ${timer0RangeDisplay} · VCount ${vcountRangeDisplay}`,
    ];
  }, [bootTiming.hardware, bootTiming.romRegion, macDisplay, timer0RangeDisplay, vcountRangeDisplay, version]);

  const dialogState: EggBootTimingDialogState = {
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

// === Utility Functions ===

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

function deriveVersionFromGameMode(gameMode: number): string {
  // EggGameMode: 0=BwNew, 1=BwContinue, 2=Bw2New, 3=Bw2Continue
  switch (gameMode) {
    case 0:
    case 1:
      return 'BW';
    case 2:
    case 3:
      return 'BW2';
    default:
      return 'BW';
  }
}
