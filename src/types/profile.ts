import type { Hardware, ROMRegion, ROMVersion } from './rom';

export interface NumericRange {
  min: number;
  max: number;
}

export interface DeviceProfile {
  id: string;
  name: string;
  description?: string;
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  timer0Auto: boolean;
  timer0Range: NumericRange;
  vcountRange: NumericRange;
  macAddress: readonly [number, number, number, number, number, number];
  tid: number;
  sid: number;
  shinyCharm: boolean;
  newGame: boolean;
  withSave: boolean;
  memoryLink: boolean;
  createdAt: string;
  updatedAt?: string;
}

export type DeviceProfileDraft = Omit<DeviceProfile, 'id' | 'createdAt' | 'updatedAt' | 'macAddress'> & {
  macAddress: number[];
};

export function generateDeviceProfileId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  const random = Math.random().toString(36).slice(2, 8);
  return `profile-${Date.now().toString(36)}-${random}`;
}

export function createDeviceProfile(draft: DeviceProfileDraft, id?: string): DeviceProfile {
  const createdAt = new Date().toISOString();
  return {
    id: id ?? generateDeviceProfileId(),
    createdAt,
    updatedAt: createdAt,
    ...normalizeDraft(draft),
  };
}

export function applyDeviceProfileDraft(
  profile: DeviceProfile,
  draft: Partial<DeviceProfileDraft>,
): DeviceProfile {
  const nextDraft: DeviceProfileDraft = {
    name: draft.name ?? profile.name,
    description: draft.description ?? profile.description,
    romVersion: draft.romVersion ?? profile.romVersion,
    romRegion: draft.romRegion ?? profile.romRegion,
    hardware: draft.hardware ?? profile.hardware,
    timer0Auto: draft.timer0Auto ?? profile.timer0Auto,
    timer0Range: draft.timer0Range ?? profile.timer0Range,
    vcountRange: draft.vcountRange ?? profile.vcountRange,
    macAddress: draft.macAddress ?? [...profile.macAddress],
    tid: draft.tid ?? profile.tid,
    sid: draft.sid ?? profile.sid,
    shinyCharm: draft.shinyCharm ?? profile.shinyCharm,
    newGame: draft.newGame ?? profile.newGame,
    withSave: draft.withSave ?? profile.withSave,
    memoryLink: draft.memoryLink ?? profile.memoryLink,
  };
  return {
    ...profile,
    ...normalizeDraft(nextDraft),
    id: profile.id,
    createdAt: profile.createdAt,
    updatedAt: new Date().toISOString(),
  };
}

export function normalizeDraft(draft: DeviceProfileDraft): Omit<DeviceProfile, 'id' | 'createdAt' | 'updatedAt'> {
  const romVersion = draft.romVersion;
  const newGame = Boolean(draft.newGame);
  let withSave = Boolean(draft.withSave);
  if (!newGame) {
    withSave = true;
  }
  let memoryLink = Boolean(draft.memoryLink);
  if (romVersion === 'B' || romVersion === 'W') {
    memoryLink = false;
  }
  if (!withSave) {
    memoryLink = false;
  }
  return {
    name: draft.name.trim() || 'Unnamed Profile',
    description: draft.description?.trim() || undefined,
    romVersion,
    romRegion: draft.romRegion,
    hardware: draft.hardware,
    timer0Auto: Boolean(draft.timer0Auto),
    timer0Range: normalizeRange(draft.timer0Range, { min: 0, max: 65535 }),
    vcountRange: normalizeRange(draft.vcountRange, { min: 0, max: 255 }),
    macAddress: normalizeMac(draft.macAddress),
    tid: clampNumber(draft.tid, 0, 65535),
    sid: clampNumber(draft.sid, 0, 65535),
    shinyCharm: Boolean(draft.shinyCharm),
    newGame,
    withSave,
    memoryLink,
  };
}

export function createDefaultDeviceProfile(): DeviceProfile {
  return createDeviceProfile({
    name: 'Default Device',
    description: undefined,
    romVersion: 'B',
    romRegion: 'JPN',
    hardware: 'DS',
    timer0Auto: true,
    timer0Range: { min: 3193, max: 3194 },
    vcountRange: { min: 95, max: 95 },
    macAddress: [0x00, 0x1B, 0x2C, 0x3D, 0x4E, 0x5F],
    tid: 1,
    sid: 2,
    shinyCharm: false,
    newGame: false,
    withSave: true,
    memoryLink: false,
  });
}

export function deviceProfileToDraft(profile: DeviceProfile): DeviceProfileDraft {
  return {
    name: profile.name,
    description: profile.description,
    romVersion: profile.romVersion,
    romRegion: profile.romRegion,
    hardware: profile.hardware,
    timer0Auto: profile.timer0Auto,
    timer0Range: { ...profile.timer0Range },
    vcountRange: { ...profile.vcountRange },
    macAddress: Array.from(profile.macAddress),
    tid: profile.tid,
    sid: profile.sid,
    shinyCharm: profile.shinyCharm,
    newGame: profile.newGame,
    withSave: profile.withSave,
    memoryLink: profile.memoryLink,
  };
}

function normalizeRange(
  range: NumericRange,
  bounds: { min: number; max: number },
): NumericRange {
  const min = clampNumber(range.min, bounds.min, bounds.max);
  const max = clampNumber(range.max, bounds.min, bounds.max);
  if (min > max) {
    return { min: max, max: min };
  }
  return { min, max };
}

function normalizeMac(input: number[]): readonly [number, number, number, number, number, number] {
  const sanitized = [...input];
  while (sanitized.length < 6) sanitized.push(0);
  const normalized = sanitized.slice(0, 6).map((value) => clampNumber(value, 0, 255));
  return [
    normalized[0],
    normalized[1],
    normalized[2],
    normalized[3],
    normalized[4],
    normalized[5],
  ];
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  const rounded = Math.round(value);
  if (rounded < min) return min;
  if (rounded > max) return max;
  return rounded;
}
