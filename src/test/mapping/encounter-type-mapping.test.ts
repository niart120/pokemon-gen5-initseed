import { describe, test, expect, beforeAll } from 'vitest';
import { initWasm } from '@/lib/core/wasm-interface';
import { DomainEncounterType, DomainEncounterTypeNames } from '@/types/domain';
import { ensureEncounterTypeAlignment, domainEncounterTypeToWasm, wasmEncounterTypeToDomain, isExactDomainEncounterType } from '@/lib/core/mapping/encounter-type';

describe('EncounterType mapping alignment', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('domain <-> wasm enum values are aligned', () => {
    expect(() => ensureEncounterTypeAlignment()).not.toThrow();
  });

  test('round-trip conversion preserves value', () => {
    for (const name of DomainEncounterTypeNames) {
      const v = (DomainEncounterType as Record<string, number>)[name] as DomainEncounterType;
      const wasmVal = domainEncounterTypeToWasm(v);
      const back = wasmEncounterTypeToDomain(wasmVal);
      if (isExactDomainEncounterType(v)) {
        expect(back).toBe(v);
      } else {
        expect(back).not.toBe(v);
        expect(back).toBe(DomainEncounterType.StaticSymbol);
      }
    }
  });

  test('invalid wasm value throws', () => {
    expect(() => wasmEncounterTypeToDomain(9999)).toThrow();
  });
});
