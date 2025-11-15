import { formatHexDisplay, parseHexInput, parseMacByte } from '@/lib/utils/hex-parser';
import type { DeviceProfileDraft } from '@/types/profile';
import type { ProfileFormState } from './profileFormTypes';
import type { ROMVersion } from '@/types/rom';
import type { SupportedLocale } from '@/types/i18n';
import {
  formatProfileHexInvalid,
  formatProfileIntegerRange,
  formatProfileIntegerRequired,
  formatProfileMacSegmentInvalid,
  formatProfileRangeOrderError,
  resolveProfileMacSegmentsCountError,
  resolveProfileNameRequired,
} from '@/lib/i18n/strings/profile-validation';

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

export function formToDraft(
  form: ProfileFormState,
  locale: SupportedLocale,
  labels: {
    timer0Min: string;
    timer0Max: string;
    vcountMin: string;
    vcountMax: string;
    tid: string;
    sid: string;
  },
): { draft: DeviceProfileDraft | null; validationErrors: string[] } {
  const errors: string[] = [];
  if (!form.name.trim()) {
    errors.push(resolveProfileNameRequired(locale));
  }

  const timer0Min = parseAndValidateHex(form.timer0Min, labels.timer0Min, 0xffff, locale, errors);
  const timer0Max = parseAndValidateHex(form.timer0Max, labels.timer0Max, 0xffff, locale, errors);
  const vcountMin = parseAndValidateHex(form.vcountMin, labels.vcountMin, 0xff, locale, errors);
  const vcountMax = parseAndValidateHex(form.vcountMax, labels.vcountMax, 0xff, locale, errors);

  if (timer0Min !== null && timer0Max !== null && timer0Min > timer0Max) {
    errors.push(formatProfileRangeOrderError('Timer0', locale));
  }
  if (vcountMin !== null && vcountMax !== null && vcountMin > vcountMax) {
    errors.push(formatProfileRangeOrderError('VCount', locale));
  }

  const macAddress: number[] = [];
  form.macSegments.forEach((segment, index) => {
    const parsed = parseMacByte(segment);
    if (parsed === null) {
      errors.push(formatProfileMacSegmentInvalid(index + 1, locale));
    } else {
      macAddress.push(parsed);
    }
  });

  const tid = parseInteger(form.tid, labels.tid, locale, errors);
  const sid = parseInteger(form.sid, labels.sid, locale, errors);

  if (macAddress.length !== 6) {
    errors.push(resolveProfileMacSegmentsCountError(locale));
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

export function parseAndValidateHex(
  value: string,
  label: string,
  max: number,
  locale: SupportedLocale,
  errors: string[],
): number | null {
  const parsed = parseHexInput(value, max);
  if (parsed === null) {
    errors.push(formatProfileHexInvalid(label, locale));
    return null;
  }
  return parsed;
}

export function parseInteger(
  value: string,
  label: string,
  locale: SupportedLocale,
  errors: string[],
): number | null {
  if (!value.trim()) {
    errors.push(formatProfileIntegerRequired(label, locale));
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 65535) {
    errors.push(formatProfileIntegerRange(label, locale));
    return null;
  }
  return parsed;
}
