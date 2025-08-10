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
    expect(WasmGameVersion.BlackWhite).toBe(DomainGameVersion.BlackWhite);
    expect(WasmGameVersion.BlackWhite2).toBe(DomainGameVersion.BlackWhite2);

    expect(WasmEncounterType.Normal).toBe(DomainEncounterType.Normal);
    expect(WasmEncounterType.StaticEvent).toBe(DomainEncounterType.StaticEvent);

    expect(WasmGameMode.BwContinue).toBe(DomainGameMode.BwContinue);
  });

  it('romVersionToGameVersion converts correctly', () => {
    expect(romVersionToGameVersion('B')).toBe(WasmGameVersion.BlackWhite);
    expect(romVersionToGameVersion('W2')).toBe(WasmGameVersion.BlackWhite2);
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
