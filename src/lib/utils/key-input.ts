/**
 * キー定義（ビット位置マッピング用）
 * ゲームの内部表現に対応
 */
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

/**
 * 表示用キー順序
 * A,B,X,Y,Start,Select,L,R,[↑],[↓],[←],[→] の順で表示
 */
const KEY_DISPLAY_ORDER: readonly KeyName[] = [
  'A', 'B', 'X', 'Y', 'Start', 'Select', 'L', 'R',
  '[↑]', '[↓]', '[←]', '[→]',
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

/**
 * マスクからキー名配列を取得（表示順序でソート済み）
 */
function collectKeyNamesForDisplay(mask: number): KeyName[] {
  const presentKeys = new Set<KeyName>();
  for (const [key, bit] of KEY_DEFINITIONS) {
    if ((mask & (1 << bit)) !== 0) {
      presentKeys.add(key);
    }
  }
  // 表示順序に従ってソート
  return KEY_DISPLAY_ORDER.filter(key => presentKeys.has(key));
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

/**
 * キーマスクから有効なキーコードの配列を生成する
 *
 * keyInputMaskで指定されたキーの全べき集合組み合わせを列挙し、
 * 不可能なキー組み合わせ（上下同時押し等）を除外した上で、
 * 各組み合わせを 0x2FFF とXORしたキーコードを返す。
 *
 * @param keyInputMask 有効なキーのビットマスク（押す可能性のあるキー）
 * @returns 有効なキーコードの配列
 */
export function generateValidKeyCodes(keyInputMask: number): number[] {
  const normalized = toMask(keyInputMask, 'mask');
  const enabledBits: number[] = [];
  for (let bit = 0; bit < KEY_BIT_COUNT; bit += 1) {
    if ((normalized & (1 << bit)) !== 0) {
      enabledBits.push(bit);
    }
  }

  const keyCodes: number[] = [];
  const totalCombinations = 1 << enabledBits.length;

  for (let combinationIndex = 0; combinationIndex < totalCombinations; combinationIndex += 1) {
    let pressedMask = 0;
    for (let bitIndex = 0; bitIndex < enabledBits.length; bitIndex += 1) {
      if ((combinationIndex & (1 << bitIndex)) !== 0) {
        pressedMask |= 1 << enabledBits[bitIndex]!;
      }
    }

    if (hasImpossibleKeyCombination(pressedMask)) {
      continue;
    }

    keyCodes.push((pressedMask ^ KEY_CODE_BASE) >>> 0);
  }

  return keyCodes;
}

export function countValidKeyCombinations(mask: number): number {
  const count = generateValidKeyCodes(mask).length;
  return count > 0 ? count : 1;
}

/** キー入力表示の既定フォールバック値 */
export const KEY_INPUT_DISPLAY_FALLBACK = '-';

/** キー入力表示のセパレータ */
const KEY_INPUT_DISPLAY_JOINER = '-';

/**
 * キー入力を表示用文字列にフォーマットする
 * 
 * UI表示・Export両方で使用する統一関数。
 * 
 * @param keyCode - keyCode値（0x2FFF XOR mask形式）
 * @param keyInputNames - 事前解決済みのキー名配列（あれば優先）
 * @param fallback - キー入力がない場合のフォールバック値
 * @returns 表示用文字列（例: "A-B-Start"）
 */
export function formatKeyInputForDisplay(
  keyCode: number | null | undefined,
  keyInputNames: KeyName[] | undefined,
  fallback: string = KEY_INPUT_DISPLAY_FALLBACK
): string {
  // keyInputNamesがあればそれを使用（表示順序でソート）
  if (keyInputNames && keyInputNames.length > 0) {
    const sortedNames = sortKeyNamesForDisplay(keyInputNames);
    return sortedNames.join(KEY_INPUT_DISPLAY_JOINER);
  }
  
  // keyCodeがあれば変換
  if (keyCode != null) {
    const mask = toMask(keyCode, 'code');
    const names = collectKeyNamesForDisplay(mask);
    if (names.length > 0) {
      return names.join(KEY_INPUT_DISPLAY_JOINER);
    }
  }
  
  return fallback;
}

/**
 * キー名配列を表示順序でソートする
 */
function sortKeyNamesForDisplay(names: KeyName[]): KeyName[] {
  const nameSet = new Set(names);
  return KEY_DISPLAY_ORDER.filter(key => nameSet.has(key));
}
