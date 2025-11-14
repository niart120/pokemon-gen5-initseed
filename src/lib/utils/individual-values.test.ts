import { describe, expect, it } from 'vitest';
import { MersenneTwister19937 } from 'random-js';

import { generateIndividualValues, type IndividualValues } from './individual-values';
import { SEED_TEMPLATES } from '@/data/seed-templates';

function createExpectedValues(seed: number, offset: number, order: readonly (keyof IndividualValues)[]): IndividualValues {
  const normalizedSeed = seed >>> 0;
  const engine = MersenneTwister19937.seed(normalizedSeed | 0);
  const normalizedOffset = offset >>> 0;
  if (normalizedOffset > 0) {
    engine.discard(normalizedOffset);
  }

  const values: IndividualValues = { h: 0, a: 0, b: 0, c: 0, d: 0, s: 0 };
  for (const stat of order) {
    values[stat] = (engine.next() >>> 27) & 0x1f;
  }
  return values;
}

describe('generateIndividualValues', () => {
  it('generates IVs in standard order when not a roamer', () => {
    const seed = 0x12345678;
    const offset = 0;
    const result = generateIndividualValues(seed, offset, false);

    const expected = createExpectedValues(seed, offset, ['h', 'a', 'b', 'c', 'd', 's']);
    expect(result).toEqual(expected);
  });

  it('applies the provided offset before generating IVs', () => {
    const seed = 0x0f0f0f0f;
    const offset = 5;
    const result = generateIndividualValues(seed, offset, false);

    const expected = createExpectedValues(seed, offset, ['h', 'a', 'b', 'c', 'd', 's']);
    expect(result).toEqual(expected);
  });

  it('uses roamer-specific stat order', () => {
    const seed = 0xdeadbeef;
    const offset = 2;
    const result = generateIndividualValues(seed, offset, true);

    const expected = createExpectedValues(seed, offset, ['h', 'a', 'b', 's', 'c', 'd']);
    expect(result).toEqual(expected);
  });

  it('rejects negative offsets', () => {
    expect(() => generateIndividualValues(1, -1, false)).toThrow(RangeError);
  });
});

describe('generateIndividualValues with seed templates', () => {
  const digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

  const patternAssertions: Record<string, (values: IndividualValues) => void> = {
    '6V': values => {
      expect(Object.values(values)).toEqual([31, 31, 31, 31, 31, 31]);
    },
    '5VA0': values => {
      expect(values.a).toBe(0);
      const others = [values.h, values.b, values.c, values.d, values.s];
      expect(others).toEqual([31, 31, 31, 31, 31]);
    },
    'V0VVV0': values => {
      expect(values).toEqual({ h: 31, a: 0, b: 31, c: 31, d: 31, s: 0 });
    },
    'V2UVVV': values => {
      expect(values).toEqual({ h: 31, a: 2, b: 30, c: 31, d: 31, s: 31 });
    },
    'U2UUUV': values => {
      expect(values).toEqual({ h: 30, a: 2, b: 30, c: 31, d: 30, s: 30 });
    },
  };

  function extractConsumption(description: string | undefined, isRoamer: boolean): number {
    if (!description) {
      return isRoamer ? 1 : 0;
    }

    const match = description.match(/消費(\d+)/);
    if (match) {
      return Number(match[1]);
    }

    if (description.includes('消費なし')) {
      return isRoamer ? 1 : 0;
    }

    return isRoamer ? 1 : 0;
  }

  function extractPatternToken(description: string | undefined): string | null {
    if (!description) return null;
    const match = description.match(/([0-9A-Z]{2,6})（/);
    return match ? match[1] : null;
  }

  function formatPattern(values: IndividualValues): string {
    return [values.h, values.a, values.b, values.c, values.d, values.s]
      .map(value => digits[value] ?? '?')
      .join('');
  }

  for (const template of SEED_TEMPLATES) {
    const patternToken = extractPatternToken(template.description);
    if (!patternToken) {
      continue;
    }

    const assertion = patternAssertions[patternToken];
    if (!assertion) {
      continue;
    }

    const isRoamer = template.name.includes('徘徊');
    const offset = extractConsumption(template.description, isRoamer);

    describe(template.name, () => {
      it('matches expected IV pattern across all seeds', () => {
        for (const seed of template.seeds) {
          const values = generateIndividualValues(seed, offset, isRoamer);
          assertion(values);
        }
      });

      it('produces a consistent IV signature for each seed', () => {
        const baseline = generateIndividualValues(template.seeds[0], offset, isRoamer);
        const baselinePattern = formatPattern(baseline);

        for (const seed of template.seeds.slice(1)) {
          const values = generateIndividualValues(seed, offset, isRoamer);
          expect(formatPattern(values)).toBe(baselinePattern);
        }
      });
    });
  }
});
