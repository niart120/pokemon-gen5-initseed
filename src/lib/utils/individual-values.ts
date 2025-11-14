import { MersenneTwister19937 } from 'random-js';

export interface IndividualValues {
  h: number;
  a: number;
  b: number;
  c: number;
  d: number;
  s: number;
}

const STANDARD_ORDER = ['h', 'a', 'b', 'c', 'd', 's'] as const;
const ROAMER_ORDER = ['h', 'a', 'b', 's', 'c', 'd'] as const;

function toIv(value: number): number {
  return (value >>> 27) & 0x1f;
}

function normalizeSeed(value: number): number {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new RangeError('Seed must be a finite integer.');
  }
  if (value < -0x8000_0000 || value > 0xffff_ffff) {
    throw new RangeError('Seed must fit within 32 bits.');
  }
  return value >>> 0;
}

function normalizeOffset(value: number): number {
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 0) {
    throw new RangeError('Offset must be a non-negative integer.');
  }
  if (value > 0xffff_ffff) {
    throw new RangeError('Offset must fit within 32 bits.');
  }
  return value >>> 0;
}

export function generateIndividualValues(seed: number, offset: number, isRoamer: boolean): IndividualValues {
  const normalizedSeed = normalizeSeed(seed);
  const normalizedOffset = normalizeOffset(offset);

  const engine = MersenneTwister19937.seed(normalizedSeed | 0);

  if (normalizedOffset > 0) {
    engine.discard(normalizedOffset);
  }

  const statsOrder = isRoamer ? ROAMER_ORDER : STANDARD_ORDER;
  const values: IndividualValues = { h: 0, a: 0, b: 0, c: 0, d: 0, s: 0 };

  for (const stat of statsOrder) {
    values[stat] = toIv(engine.next());
  }

  return values;
}
