// Deprecated: This barrel is maintained for backward compatibility only.
// New imports should use './rom', './search', or './parallel' directly.

export * from './rom';
export * from './search';
export * from './parallel';

export const KEY_MAPPINGS = {
  A: 0,
  B: 1,
  SELECT: 2,
  START: 3,
  RIGHT: 4,
  LEFT: 5,
  UP: 6,
  DOWN: 7,
  R: 8,
  L: 9,
  X: 10,
  Y: 11
} as const;

export type KeyName = keyof typeof KEY_MAPPINGS;