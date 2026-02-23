import { KEY_INPUT_DEFAULT, normalizeKeyMask } from '@/lib/utils/key-input';
import type { BootTimingDraft } from '@/types/generation';
import type { DeviceProfile } from '@/types/profile';

export type BootTimingProfileSource = Pick<
  DeviceProfile,
  'romRegion' | 'hardware' | 'timer0Range' | 'vcountRange' | 'macAddress'
>;

function clampToBounds(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function clampNumber(value: number | undefined, min: number, max: number, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return clampToBounds(fallback, min, max);
  }
  return clampToBounds(Math.round(value), min, max);
}

function normalizeNumericRange(
  partial: { min?: number; max?: number } | undefined,
  fallback: { min: number; max: number },
  minBound: number,
  maxBound: number,
): { min: number; max: number } {
  const nextMin = clampNumber(partial?.min, minBound, maxBound, fallback.min);
  const nextMax = clampNumber(partial?.max, minBound, maxBound, fallback.max);
  if (nextMin > nextMax) {
    return { min: nextMax, max: nextMin };
  }
  return { min: nextMin, max: nextMax };
}

function ensureMacAddressTuple(
  values: readonly number[] | undefined,
  fallback: BootTimingDraft['macAddress'] = [0, 0, 0, 0, 0, 0],
): BootTimingDraft['macAddress'] {
  const result: number[] = [];
  for (let i = 0; i < 6; i += 1) {
    const fallbackValue = fallback[i] ?? 0;
    const raw = values?.[i];
    result.push(clampNumber(raw, 0, 255, fallbackValue));
  }
  return result as unknown as BootTimingDraft['macAddress'];
}

export function createBootTimingDraftFromProfile(profile: BootTimingProfileSource): BootTimingDraft {
  return {
    timestampIso: undefined,
    keyMask: KEY_INPUT_DEFAULT,
    timer0Range: { ...profile.timer0Range },
    vcountRange: { ...profile.vcountRange },
    romRegion: profile.romRegion,
    hardware: profile.hardware,
    macAddress: ensureMacAddressTuple(profile.macAddress),
  };
}

export function cloneBootTimingDraft(source: BootTimingDraft): BootTimingDraft {
  return {
    timestampIso: source.timestampIso,
    keyMask: source.keyMask,
    timer0Range: { ...source.timer0Range },
    vcountRange: { ...source.vcountRange },
    romRegion: source.romRegion,
    hardware: source.hardware,
    macAddress: ensureMacAddressTuple(source.macAddress),
  };
}

export function normalizeBootTimingDraft(
  partial: Partial<BootTimingDraft> | undefined,
  fallback: BootTimingDraft,
): BootTimingDraft {
  const base = fallback;
  const timer0Range = normalizeNumericRange(partial?.timer0Range, base.timer0Range, 0, 0xFFFF);
  const vcountRange = normalizeNumericRange(partial?.vcountRange, base.vcountRange, 0, 0xFF);
  const keyMask = partial?.keyMask != null ? normalizeKeyMask(partial.keyMask) : base.keyMask;
  return {
    timestampIso: partial?.timestampIso ?? base.timestampIso,
    keyMask,
    timer0Range,
    vcountRange,
    romRegion: partial?.romRegion ?? base.romRegion,
    hardware: partial?.hardware ?? base.hardware,
    macAddress: ensureMacAddressTuple(partial?.macAddress, base.macAddress),
  };
}

export function bootTimingDraftFromProfile(
  profile: DeviceProfile,
  current: BootTimingDraft,
): BootTimingDraft {
  return normalizeBootTimingDraft(
    {
      timestampIso: current.timestampIso,
      keyMask: current.keyMask,
      romRegion: profile.romRegion,
      hardware: profile.hardware,
      timer0Range: { ...profile.timer0Range },
      vcountRange: { ...profile.vcountRange },
      macAddress: Array.from(profile.macAddress) as unknown as BootTimingDraft['macAddress'],
    },
    current,
  );
}
