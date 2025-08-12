import { describe, test, expect, beforeAll } from 'vitest';
import { initWasm } from '@/lib/core/wasm-interface';
import { DomainEncounterType } from '@/types/domain';
import { ensureEncounterTypeAlignment, domainEncounterTypeToWasm, wasmEncounterTypeToDomain } from '@/lib/core/mapping/encounter-type';

describe('EncounterType mapping alignment', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('domain <-> wasm enum values are aligned', () => {
    expect(() => ensureEncounterTypeAlignment()).not.toThrow();
  });

  test('round-trip conversion preserves value', () => {
    const domainKeys = Object.keys(DomainEncounterType).filter(k => isNaN(Number(k)));
    for (const k of domainKeys) {
      const v = (DomainEncounterType as any)[k] as number;
      const wasmVal = domainEncounterTypeToWasm(v);
      const back = wasmEncounterTypeToDomain(wasmVal);
      expect(back).toBe(v);
    }
  });

  test('invalid wasm value throws', () => {
    expect(() => wasmEncounterTypeToDomain(9999)).toThrow();
  });
});
