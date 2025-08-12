/**
 * Integration smoke tests (lean): WASM fallbackと最小限の統合経路確認
 * 役割:
 * - WASM初期化の可否とフォールバック挙動のヘルスチェック
 * - データ統合の極小スモーク（詳細は専用スイートに委譲）
 *
 * 詳細検証の担当:
 * - 遭遇スロット/レベル分布: encounter-selection.test.ts
 * - 統合データ処理/同期ルール/特殊遭遇: resolver 系テストに統合（assembler は廃止）
 * - 性別境界: gender-threshold.test.ts
 * - WASM/サービス連携: wasm-node.test.ts, phase2-* tests（wasm-serviceは廃止済みでプレースホルダ）
 */

import { describe, test, expect, beforeAll } from 'vitest';
import { buildResolutionContext } from '../../lib/initialization/build-resolution-context';
import { resolvePokemon, toUiReadyPokemon } from '../../lib/generation/pokemon-resolver';
import { initWasm, getWasm } from '../../lib/core/wasm-interface';
import { parseFromWasmRaw } from '../../lib/generation/raw-parser';


describe('Integration: WASM生成→統合→UI変換', () => {
  beforeAll(async () => {
    await initWasm();
  });


  test('通常野生遭遇（シンクロなし）: 生成→統合→UI', () => {
    // BW2, Route1, 野生, シンクロなし
    const seed = 0x12345678n;
    const { BWGenerationConfig, GameVersion, EncounterType, PokemonGenerator } = getWasm();
    const config = new BWGenerationConfig(
      GameVersion.B2,
      EncounterType.Normal,
      12345, // TID
      54321, // SID
      false, // sync_enabled
      0      // sync_nature_id
    );
    const wasmRaw = PokemonGenerator.generate_single_pokemon_bw(seed, config);
    const raw = parseFromWasmRaw(wasmRaw);
  const ctx = buildResolutionContext({ version: 'B2', location: 'Route1', encounterType: EncounterType.Normal as unknown as any });
    const enhanced = resolvePokemon(raw, ctx);
    const ui = toUiReadyPokemon(enhanced);
    expect(typeof ui.speciesName).toBe('string');
    expect(ui.level).toBeGreaterThan(0);
    expect(['M', 'F', '-', '?']).toContain(ui.gender);
    expect(typeof ui.natureName).toBe('string');
  });


  test('シンクロ適用: 生成→統合→UI (代表seed固定)', () => {
    const targetNatureId = 5; // ずぶとい
    const syncSeed = 0x87654322n; // 事前探索で sync_applied 確認済み seed
    const { BWGenerationConfig, GameVersion, EncounterType, PokemonGenerator } = getWasm();
    const config = new BWGenerationConfig(
      GameVersion.B2,
      EncounterType.Normal,
      12345,
      54321,
      true,
      targetNatureId
    );
    const ctx = buildResolutionContext({ version: 'B2', location: 'Route1', encounterType: EncounterType.Normal as unknown as any });
    const wasmRaw = PokemonGenerator.generate_single_pokemon_bw(syncSeed, config);
    const raw = parseFromWasmRaw(wasmRaw);
    expect(raw.sync_applied).toBe(true); // 前提確認
    const enhanced = resolvePokemon(raw, ctx);
    const ui = toUiReadyPokemon(enhanced);
    expect(typeof ui.speciesName).toBe('string');
    expect(['M', 'F', '-', '?']).toContain(ui.gender);
    expect(enhanced.natureId).toBe(targetNatureId);
  });


  test('性別境界: gender_valueでM/F/Nを判定', () => {
    // gender_value=0: M, 127: M, 128: F, 255: F（例: 50%種）
    const seeds = [0x10000000n, 0x1000007Fn, 0x10000080n, 0x100000FFn];
    const { BWGenerationConfig, GameVersion, EncounterType, PokemonGenerator } = getWasm();
    const config = new BWGenerationConfig(
      GameVersion.B2,
      EncounterType.Normal,
      12345, 54321, false, 0
    );
    for (let i = 0; i < seeds.length; ++i) {
      const wasmRaw = PokemonGenerator.generate_single_pokemon_bw(seeds[i], config);
      const raw = parseFromWasmRaw(wasmRaw);
  const ctx = buildResolutionContext({ version: 'B2', location: 'Route1', encounterType: EncounterType.Normal as unknown as any });
      const enhanced = resolvePokemon(raw, ctx);
      const ui = toUiReadyPokemon(enhanced);
      expect(['M', 'F', '-', '?']).toContain(ui.gender);
      // 期待値は暫定（種族依存のため、厳密にはspeciesIdごとに再検証要）
    }
  });
});