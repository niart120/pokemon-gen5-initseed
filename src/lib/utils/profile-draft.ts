import type { DeviceProfileDraft, NumericRange } from '@/types/profile';

const compareRange = (a: NumericRange, b: NumericRange): boolean => a.min === b.min && a.max === b.max;

const compareMacAddress = (a: number[], b: number[]): boolean => {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
};

export const areDeviceProfileDraftsEqual = (
  a: DeviceProfileDraft | null | undefined,
  b: DeviceProfileDraft | null | undefined,
): boolean => {
  if (a === b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  return (
    a.name === b.name &&
    a.description === b.description &&
    a.romVersion === b.romVersion &&
    a.romRegion === b.romRegion &&
    a.hardware === b.hardware &&
    a.timer0Auto === b.timer0Auto &&
    compareRange(a.timer0Range, b.timer0Range) &&
    compareRange(a.vcountRange, b.vcountRange) &&
    compareMacAddress(a.macAddress, b.macAddress) &&
    a.tid === b.tid &&
    a.sid === b.sid &&
    a.shinyCharm === b.shinyCharm &&
    a.newGame === b.newGame &&
    a.withSave === b.withSave &&
    a.memoryLink === b.memoryLink
  );
};
