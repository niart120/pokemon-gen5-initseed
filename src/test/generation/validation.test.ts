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
    progressIntervalMs: 100,
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
});
