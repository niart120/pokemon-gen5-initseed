/**
 * Gender boundary tests (femaleThreshold semantics)
 */

import { describe, it, expect } from 'vitest';
import { determineGenderFromSpec } from '../lib/utils/gender-utils';

describe('Gender determination by femaleThreshold', () => {
  const ratio = (t: number) => ({ type: 'ratio' as const, femaleThreshold: t });

  it('threshold=31 (approx 12.1% female): v<31 -> Female, v>=31 -> Male', () => {
    expect(determineGenderFromSpec(0, ratio(31))).toBe('Female');
    expect(determineGenderFromSpec(30, ratio(31))).toBe('Female');
    expect(determineGenderFromSpec(31, ratio(31))).toBe('Male');
    expect(determineGenderFromSpec(255, ratio(31))).toBe('Male');
  });

  it('threshold=63 (approx 24.6% female): v<63 -> Female, v>=63 -> Male', () => {
    expect(determineGenderFromSpec(0, ratio(63))).toBe('Female');
    expect(determineGenderFromSpec(62, ratio(63))).toBe('Female');
    expect(determineGenderFromSpec(63, ratio(63))).toBe('Male');
  });

  it('threshold=127 (~49.6% female): v<127 -> Female, v>=127 -> Male', () => {
    expect(determineGenderFromSpec(0, ratio(127))).toBe('Female');
    expect(determineGenderFromSpec(126, ratio(127))).toBe('Female');
    expect(determineGenderFromSpec(127, ratio(127))).toBe('Male');
  });

  it('threshold=191 (~74.6% female): v<191 -> Female, v>=191 -> Male', () => {
    expect(determineGenderFromSpec(0, ratio(191))).toBe('Female');
    expect(determineGenderFromSpec(190, ratio(191))).toBe('Female');
    expect(determineGenderFromSpec(191, ratio(191))).toBe('Male');
  });

  it('threshold=255 (100% female): v<255 -> Female, v>=255 -> Male(never occurs)', () => {
    expect(determineGenderFromSpec(0, ratio(255))).toBe('Female');
    expect(determineGenderFromSpec(254, ratio(255))).toBe('Female');
    // v==255 -> Male by rule, but in practice female-only species use fixed
    expect(determineGenderFromSpec(255, ratio(255))).toBe('Male');
  });

  it('genderless and fixed genders', () => {
    expect(determineGenderFromSpec(100, { type: 'genderless' })).toBe('Genderless');
    expect(determineGenderFromSpec(0, { type: 'fixed', fixed: 'male' })).toBe('Male');
    expect(determineGenderFromSpec(255, { type: 'fixed', fixed: 'female' })).toBe('Female');
  });
});
