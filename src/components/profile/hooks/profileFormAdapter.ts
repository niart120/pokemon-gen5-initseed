import { formatHexDisplay, parseHexInput, parseMacByte } from '@/lib/utils/hex-parser';
import type { DeviceProfileDraft } from '@/types/profile';
import type { ProfileFormState } from './profileFormTypes';
import type { ROMVersion } from '@/types/rom';

export function enforceMemoryLink(current: boolean, version: ROMVersion, hasSave: boolean): boolean {
  if (version === 'B' || version === 'W') return false;
  if (!hasSave) return false;
  return current;
}

export function enforceShinyCharm(current: boolean, version: ROMVersion, hasSave: boolean): boolean {
  if (version === 'B' || version === 'W') return false;
  if (!hasSave) return false;
  return current;
}

export function canonicalizeHex(value: string, minDigits: number): string {
  const trimmed = value.trim();
  if (!trimmed) return value;
  const parsed = parseHexInput(trimmed);
  if (parsed === null) return value;
  return `0x${formatHexDisplay(parsed, minDigits)}`;
}

export function toTimerHex(value: number): string {
  return `0x${formatHexDisplay(value, 4)}`;
}

export function toVCountHex(value: number): string {
  return `0x${formatHexDisplay(value, 2)}`;
}

export function profileToForm(draft: DeviceProfileDraft): ProfileFormState {
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

export function formToDraft(form: ProfileFormState): { draft: DeviceProfileDraft | null; validationErrors: string[] } {
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

export function parseAndValidateHex(value: string, label: string, max: number, errors: string[]): number | null {
  const parsed = parseHexInput(value, max);
  if (parsed === null) {
    errors.push(`${label} must be a hexadecimal value`);
    return null;
  }
  return parsed;
}

export function parseInteger(value: string, label: string, errors: string[]): number | null {
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
