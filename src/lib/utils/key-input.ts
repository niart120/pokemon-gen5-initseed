const KEY_DEFINITIONS = [
  ['A', 0],
  ['B', 1],
  ['Select', 2],
  ['Start', 3],
  ['[→]', 4],
  ['[←]', 5],
  ['[↑]', 6],
  ['[↓]', 7],
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

const KEY_BIT_COUNT = KEY_DEFINITIONS.length;
const KEY_MASK_LIMIT = (1 << KEY_BIT_COUNT) - 1;
export const KEY_CODE_BASE = 0x2FFF;
export const KEY_INPUT_DEFAULT = 0x0000;

export const RAW_INVALID_KEY_COMBINATION_MASKS = [
  (1 << KEY_TO_BIT['[↑]']) | (1 << KEY_TO_BIT['[↓]']),
  (1 << KEY_TO_BIT['[←]']) | (1 << KEY_TO_BIT['[→]']),
  (1 << KEY_TO_BIT['Select']) | (1 << KEY_TO_BIT['Start']) | (1 << KEY_TO_BIT['L']) | (1 << KEY_TO_BIT['R']),
] as const;

type KeyRepresentation = 'mask' | 'code';

function toMask(value: number, representation: KeyRepresentation): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  if (representation === 'mask') {
    return value & KEY_MASK_LIMIT;
  }

  const normalizedCode = value & KEY_CODE_BASE;
  const mask = normalizedCode ^ KEY_CODE_BASE;
  return mask & KEY_MASK_LIMIT;
}

function toCode(mask: number): number {
  const normalizedMask = toMask(mask, 'mask');
  return KEY_CODE_BASE ^ normalizedMask;
}

function collectKeyNames(mask: number): KeyName[] {
  const names: KeyName[] = [];
  for (const [key, bit] of KEY_DEFINITIONS) {
    if ((mask & (1 << bit)) !== 0) {
      names.push(key);
    }
  }
  return names;
}

export function normalizeKeyMask(mask: number): number {
  return toMask(mask, 'mask');
}

export function hasImpossibleKeyCombination(rawMask: number): boolean {
  const normalized = toMask(rawMask, 'mask');
  for (const invalidMask of RAW_INVALID_KEY_COMBINATION_MASKS) {
    if ((normalized & invalidMask) === invalidMask) {
      return true;
    }
  }
  return false;
}

export function keyMaskToNames(mask: number): KeyName[] {
  const normalized = toMask(mask, 'mask');
  return collectKeyNames(normalized);
}

export function keyNamesToMask(names: Iterable<KeyName>): number {
  let mask = 0;
  for (const name of names) {
    const bit = KEY_TO_BIT[name];
    if (bit !== undefined) {
      mask |= 1 << bit;
    }
  }
  return toMask(mask, 'mask');
}

export function toggleKeyInMask(mask: number, key: KeyName): number {
  const bit = KEY_TO_BIT[key];
  if (bit === undefined) {
    return toMask(mask, 'mask');
  }
  const normalized = toMask(mask, 'mask');
  return toMask(normalized ^ (1 << bit), 'mask');
}

export function keyMaskToKeyCode(mask: number): number {
  return toCode(mask);
}

export function keyCodeToMask(keyCode: number): number {
  return toMask(keyCode, 'code');
}

export function keyCodeToNames(keyCode: number): KeyName[] {
  const normalizedMask = toMask(keyCode, 'code');
  return collectKeyNames(normalizedMask);
}

export function countValidKeyCombinations(mask: number): number {
  const normalized = toMask(mask, 'mask');
  const enabledBits: number[] = [];
  for (let bit = 0; bit < KEY_BIT_COUNT; bit += 1) {
    if ((normalized & (1 << bit)) !== 0) {
      enabledBits.push(bit);
    }
  }

  const totalCombinations = 1 << enabledBits.length;
  let validCount = 0;

  for (let combinationIndex = 0; combinationIndex < totalCombinations; combinationIndex += 1) {
    let combinationMask = 0;
    for (let bitIndex = 0; bitIndex < enabledBits.length; bitIndex += 1) {
      if ((combinationIndex & (1 << bitIndex)) !== 0) {
        combinationMask |= 1 << enabledBits[bitIndex]!;
      }
    }

    if (hasImpossibleKeyCombination(combinationMask)) {
      continue;
    }
    validCount += 1;
  }

  return validCount > 0 ? validCount : 1;
}
