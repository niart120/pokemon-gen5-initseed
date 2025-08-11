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
 * - WASM/サービス連携: wasm-service.test.ts, wasm-node.test.ts, phase2-* tests
 */

import { describe, test, expect, beforeAll } from 'vitest';
import { SeedCalculator } from '../../lib/core/seed-calculator';
import { isWasmReady } from '../../lib/core/wasm-interface';
import { buildResolutionContext } from '../../lib/initialization/build-resolution-context';
import { resolvePokemon, toUiReadyPokemon } from '../../lib/integration/pokemon-resolver';
import type { RawPokemonData } from '../../types/pokemon-raw';

describe('Integration smoke (WASM fallback + tiny pipeline)', () => {
  let calculator: SeedCalculator;

  beforeAll(async () => {
    calculator = new SeedCalculator();
    // 初期化に失敗しても許容（TSフォールバック）
    await calculator.initializeWasm();
  });

  test('WASM初期化の可否とROMパラメータ取得', () => {
    // isUsingWasmはWASM未取得でもfalseで健全
    expect(typeof calculator.isUsingWasm()).toBe('boolean');
    // 直接モジュール状態も確認（副作用なし）
    expect(typeof isWasmReady()).toBe('boolean');

    // ROMパラメータの健全性（有効なversion/regionで非null）
    const params = calculator.getROMParameters('B', 'JPN');
    expect(params).not.toBeNull();
    if (params) {
      expect(params.nazo.length).toBe(5);
      expect(params.vcountTimerRanges.length).toBeGreaterThan(0);
    }
  });

  test('最小統合パス: resolverで基本解決が得られる', () => {
    // Route1 の通常エンカウントテーブルで解決
    const ctx = buildResolutionContext({ version: 'B', location: 'Route1', encounterType: 0 as any });
    const raw: RawPokemonData = {
      seed: 0x12345678n,
      pid: 0x87654321,
      nature: 12,
      sync_applied: false,
      ability_slot: 1,
      gender_value: 100,
      encounter_slot_value: 0,
      encounter_type: 0,
      level_rand_value: 2,
      shiny_type: 0,
    };

    const resolved = resolvePokemon(raw, ctx);
    const ui = toUiReadyPokemon(resolved);
    // 極小スモーク: 正常に主要フィールドが生成されることのみ確認
    expect(resolved.speciesId).toBeGreaterThan(0);
    expect(resolved.level).toBeGreaterThan(0);
    expect(['M', 'F', 'N', undefined]).toContain(resolved.gender);
    expect(typeof ui.natureName).toBe('string');
  });
});