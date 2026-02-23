import { describe, test, expect, beforeAll } from 'vitest';
import { buildResolutionContext, enrichForSpecies } from '../../lib/initialization/build-resolution-context';
import { resolvePokemon, toUiReadyPokemon } from '../../lib/generation/pokemon-resolver';
import { initWasm, getWasm } from '../../lib/core/wasm-interface';
import { parseFromWasmRaw } from '../../lib/generation/raw-parser';
import { DomainEncounterType } from '@/types/domain';

describe('Integration: BW2 続きから (Route1, seed=0x12345678)', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('初回エンカウントが期待個体 (ハーデリア) と一致する', () => {
    const seed = 0x12345678n;
    const {
      BWGenerationConfig,
      GameVersion,
      GameMode,
      EncounterType,
      SeedEnumerator,
    } = getWasm();

    const mode = GameMode.Bw2ContinueNoMemoryLink;
    const config = new BWGenerationConfig(
      GameVersion.B2,
      EncounterType.Normal,
      12345,
      54321,
      true,
      3, // いじっぱり
      false,
      false,
    );

    const enumerator = new SeedEnumerator(seed, 0n, 1, config, mode);
    const wasmRaw = enumerator.next_pokemon();
    expect(wasmRaw).toBeTruthy();

    const raw = parseFromWasmRaw(wasmRaw);
    expect(raw.pid >>> 0).toBe(0xe5d60c63);
    expect(raw.sync_applied).toBe(true);
    expect(raw.nature).toBe(3); // いじっぱり
    expect(raw.encounter_type).toBe(EncounterType.Normal);

    const ctx = buildResolutionContext({
      version: 'B2',
      location: 'Route1',
      encounterType: DomainEncounterType.Normal,
    });
    const speciesId = ctx.encounterTable?.slots[raw.encounter_slot_value]?.speciesId;
    if (speciesId) enrichForSpecies(ctx, speciesId);

    const resolved = resolvePokemon(raw, ctx);
    expect(resolved.speciesId).toBe(507); // ハーデリア
    expect(resolved.level).toBe(59);
    expect(resolved.gender).toBe('F');
    expect(resolved.abilityIndex).toBe(0); // いかく
    expect(resolved.natureId).toBe(3);

    const ui = toUiReadyPokemon(resolved, { baseSeed: seed, version: 'B2' });
    expect(ui.speciesName).toBe('ハーデリア');
    expect(ui.gender).toBe('F');
    expect(ui.stats).toEqual({
      hp: 154,
      attack: 123,
      defense: 97,
      specialAttack: 44,
      specialDefense: 95,
      speed: 82,
    });
  });
});

describe('Integration: BW 続きから (ジャイアントホール地底森林, seed=0x1C40524D87E80030)', () => {
  beforeAll(async () => {
    await initWasm();
  });

  test('初回エンカウントが期待個体 (イノムー) と一致する', () => {
    const seed = 0x1C40524D87E80030n;
    const { BWGenerationConfig, GameVersion, GameMode, EncounterType, SeedEnumerator } = getWasm();

    const mode = GameMode.BwContinue;
    const config = new BWGenerationConfig(
      GameVersion.W,
      EncounterType.Normal,
      12345,
      54321,
      false,
      0,
      false,
      false,
    );

    const enumerator = new SeedEnumerator(seed, 0n, 1, config, mode);
    const wasmRaw = enumerator.next_pokemon();
    expect(wasmRaw).toBeTruthy();

    const raw = parseFromWasmRaw(wasmRaw);
    expect(raw.pid >>> 0).toBe(0xdf8fece9);
    expect(raw.sync_applied).toBe(false);
    expect(raw.nature).toBe(14); // むじゃき
    expect(raw.ability_slot).toBe(1); // ゆきがくれ
    expect(raw.encounter_slot_value).toBe(1);
    expect(raw.encounter_type).toBe(EncounterType.Normal);

    const ctx = buildResolutionContext({
      version: 'W',
      location: 'ジャイアントホール地底森林',
      encounterType: DomainEncounterType.Normal,
    });
    const speciesId = ctx.encounterTable?.slots[raw.encounter_slot_value]?.speciesId;
    if (speciesId) enrichForSpecies(ctx, speciesId);

    const resolved = resolvePokemon(raw, ctx);
    expect(resolved.speciesId).toBe(221); // イノムー
    expect(resolved.level).toBe(52);
    expect(resolved.gender).toBe('M');
    expect(resolved.abilityIndex).toBe(1);
    expect(resolved.natureId).toBe(14);

    const ui = toUiReadyPokemon(resolved, { baseSeed: seed, version: 'W' });
    expect(ui.speciesName).toBe('イノムー');
    expect(ui.gender).toBe('M');
    expect(ui.stats).toEqual({
      hp: 182,
      attack: 125,
      defense: 104,
      specialAttack: 83,
      specialDefense: 74,
      speed: 80,
    });
  });
});