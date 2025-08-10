import { describe, it, expect } from 'vitest';
import { DomainEncounterType, DomainGameMode, DomainGameVersion } from '../../types/domain';
import {
  WasmEncounterType,
  WasmGameMode,
  WasmGameVersion,
  romVersionToGameVersion,
  stringToEncounterType,
  configToGameMode,
} from '../../lib/integration/wasm-enums';

describe('wasm-enums conversions', () => {
  it('domain enums mirror WASM numeric values', () => {
    expect(WasmGameVersion.B).toBe(DomainGameVersion.B);
    expect(WasmGameVersion.W).toBe(DomainGameVersion.W);
    expect(WasmGameVersion.B2).toBe(DomainGameVersion.B2);
    expect(WasmGameVersion.W2).toBe(DomainGameVersion.W2);

    expect(WasmEncounterType.Normal).toBe(DomainEncounterType.Normal);
    expect(WasmEncounterType.StaticEvent).toBe(DomainEncounterType.StaticEvent);

    expect(WasmGameMode.BwContinue).toBe(DomainGameMode.BwContinue);
  });

  it('romVersionToGameVersion converts correctly', () => {
    expect(romVersionToGameVersion('B')).toBe(WasmGameVersion.B);
    expect(romVersionToGameVersion('W')).toBe(WasmGameVersion.W);
    expect(romVersionToGameVersion('B2')).toBe(WasmGameVersion.B2);
    expect(romVersionToGameVersion('W2')).toBe(WasmGameVersion.W2);
  });

  it('stringToEncounterType converts correctly', () => {
    expect(stringToEncounterType('normal')).toBe(WasmEncounterType.Normal);
    expect(stringToEncounterType('static_event')).toBe(WasmEncounterType.StaticEvent);
  });

  it('configToGameMode converts correctly', () => {
    expect(configToGameMode('B', true, true)).toBe(WasmGameMode.BwNewGameWithSave);
    expect(configToGameMode('B2', false, false, true)).toBe(WasmGameMode.Bw2ContinueWithMemoryLink);
  });
});
