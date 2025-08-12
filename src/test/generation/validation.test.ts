import { describe, it, expect } from 'vitest';
import { validateGenerationParams, type GenerationParams } from '@/types/generation';

function baseParams(overrides: Partial<GenerationParams> = {}): GenerationParams {
  return {
    baseSeed: 1n,
    offset: 0n,
    maxAdvances: 5000,
    maxResults: 1000,
    version: 'B',
    encounterType: 0,
    tid: 12345,
    sid: 54321,
    syncEnabled: false,
    syncNatureId: 0,
    stopAtFirstShiny: false,
    stopOnCap: true,
    batchSize: 1000,
    ...overrides,
  };
}

describe('validateGenerationParams', () => {
  it('accepts valid params', () => {
    const errs = validateGenerationParams(baseParams());
    expect(errs).toHaveLength(0);
  });
  it('rejects out of range maxAdvances', () => {
    const errs = validateGenerationParams(baseParams({ maxAdvances: 2_000_000 }));
    expect(errs.some(e => e.includes('maxAdvances'))).toBe(true);
  });
  it('rejects out of range batchSize', () => {
    const errs = validateGenerationParams(baseParams({ batchSize: 20_000 }));
    expect(errs.some(e => e.includes('batchSize'))).toBe(true);
  });
  it('rejects tid/sid out of range', () => {
    const errs = validateGenerationParams(baseParams({ tid: 70000, sid: -1 } as any));
    expect(errs.filter(e => e.includes('tid') || e.includes('sid')).length).toBeGreaterThan(0);
  });
  it('rejects negative baseSeed / offset', () => {
    const errs = validateGenerationParams(baseParams({ baseSeed: -1n, offset: -5n } as any));
    expect(errs.some(e => e.includes('baseSeed'))).toBe(true);
    expect(errs.some(e => e.includes('offset must be non-negative'))).toBe(true);
  });
  it('rejects offset >= maxAdvances', () => {
    const errs = validateGenerationParams(baseParams({ offset: 5000n }));
    expect(errs.some(e => e.includes('offset must be < maxAdvances'))).toBe(true);
  });
  it('rejects batchSize > maxAdvances', () => {
    const errs = validateGenerationParams(baseParams({ batchSize: 6000 }));
    expect(errs.some(e => e.includes('batchSize must be <= maxAdvances'))).toBe(true);
  });
  it('rejects invalid encounterType', () => {
    const errs = validateGenerationParams(baseParams({ encounterType: 99 }));
    expect(errs.some(e => e.includes('encounterType invalid'))).toBe(true);
  });
  it('rejects maxResults > maxAdvances', () => {
    const errs = validateGenerationParams(baseParams({ maxResults: 6000 }));
    expect(errs.some(e => e.includes('maxResults should be <= maxAdvances'))).toBe(true);
  });
});
