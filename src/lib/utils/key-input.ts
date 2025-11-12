const KEY_DEFINITIONS = [
  ['A', 0],
  ['B', 1],
  ['Select', 2],
  ['Start', 3],
  ['Right', 4],
  ['Left', 5],
  ['Up', 6],
  ['Down', 7],
  ['R', 8],
  ['L', 9],
  ['X', 10],
  ['Y', 11],
] as const;

export type KeyName = (typeof KEY_DEFINITIONS)[number][0];

const KEY_TO_BIT: Record<KeyName, number> = KEY_DEFINITIONS.reduce((acc, [key, bit]) => {
  acc[key] = bit;
  return acc;
}, {} as Record<KeyName, number>);

const KEY_MASK_LIMIT = 0x0FFF;
export const KEY_CODE_BASE = 0x2FFF;
export const KEY_INPUT_DEFAULT = 0x0000;

function normalizeMask(mask: number): number {
  if (!Number.isFinite(mask)) return 0;
  return mask & KEY_MASK_LIMIT;
}

export function keyMaskToNames(mask: number): KeyName[] {
  const normalized = normalizeMask(mask);

  const names: KeyName[] = [];
  for (const [key, bit] of KEY_DEFINITIONS) {
    if ((normalized & (1 << bit)) !== 0) {
      names.push(key);
    }
  }
  return names;
}

export function keyNamesToMask(names: Iterable<KeyName>): number {
  let mask = 0;
  for (const name of names) {
    const bit = KEY_TO_BIT[name];
    if (bit !== undefined) {
      mask |= 1 << bit;
    }
  }
  return normalizeMask(mask);
}

export function toggleKeyInMask(mask: number, key: KeyName): number {
  const bit = KEY_TO_BIT[key];
  if (bit === undefined) return mask;
  return normalizeMask(mask ^ (1 << bit));
}

export function keyMaskToKeyCode(mask: number): number {
  return KEY_CODE_BASE ^ normalizeMask(mask);
}

export function keyCodeToMask(keyCode: number): number {
  if (!Number.isFinite(keyCode)) return 0;
  return KEY_CODE_BASE ^ (keyCode & KEY_CODE_BASE);
}

export function keyCodeToNames(keyCode: number): KeyName[] {
  return keyMaskToNames(keyCodeToMask(keyCode));
}
