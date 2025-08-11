/**
 * Phase 2 Integration Test - Simple demonstration
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { initWasmForTesting } from './wasm-loader';
import { parseWasmLikeToRawPokemonData } from '../lib/integration/raw-parser';
import { resolvePokemon, toUiReadyPokemon } from '../lib/integration/pokemon-resolver';
import { buildResolutionContext } from '../lib/initialization/build-resolution-context';
import { getGeneratedSpeciesById } from '../data/species/generated';
import { calculateLevel } from '../data/encounter-tables';

describe('Phase 2 Integration Demo', () => {
  beforeAll(async () => {
    await initWasmForTesting();
  });

  it('should demonstrate the complete data flow conceptually', () => {
    // Simulate raw WASM data (would come from WASM in real usage)
    const mockWasmData = {
      get_seed: () => 123456789n,
      get_pid: () => 0x12345678,
      get_nature: () => 5, // Bold
      get_sync_applied: () => false,
      get_ability_slot: () => 0,
      get_gender_value: () => 100,
      get_encounter_slot_value: () => 0,
      get_encounter_type: () => 0,
      get_level_rand_value: () => 42,
      get_shiny_type: () => 0,
    };

  // Step 1: Parse raw data (snake_case)
  const rawData = parseWasmLikeToRawPokemonData(mockWasmData);
    expect(rawData.nature).toBe(5);
    expect(rawData.pid).toBe(0x12345678);

  // Step 2: Resolve domain outputs and get UI-ready view
  const ctx = buildResolutionContext({ version: 'B', location: 'Route 1', encounterType: 0 });
  const resolved = resolvePokemon(rawData, ctx);
  const uiReady = toUiReadyPokemon(resolved);
  expect(uiReady.natureName).toBeDefined();

    // Step 3: Get species data (assuming encounter slot 0 gives us Patrat)
  const species = getGeneratedSpeciesById(504); // Patrat
  expect(species).toBeDefined();
  expect(species?.names.en).toBe('Patrat');

    // Step 4: Calculate level
  const level = calculateLevel(rawData.level_rand_value, { min: 2, max: 4 });
    expect(level).toBeGreaterThanOrEqual(2);
    expect(level).toBeLessThanOrEqual(4);

  console.log('✅ Phase 2 integration demo successful:');
    console.log(`   Seed: ${rawData.seed}`);
  console.log(`   Species: ${species?.names.en}`);
  console.log(`   Nature: ${uiReady.natureName}`);
    console.log(`   Level: ${level}`);
    console.log(`   PID: 0x${rawData.pid.toString(16)}`);
  });

  it('should validate all Phase 2 components work together', () => {
    // Test that all our main utilities work
  // Nature name validation is now via Domain table in resolver; omitted here

  const snivy = getGeneratedSpeciesById(495);
  expect(snivy?.names.en).toBe('Snivy');

    const level = calculateLevel(5, { min: 10, max: 15 });
    expect(level).toBe(15); // 10 + (5 % 6) = 15

    console.log('✅ All Phase 2 utilities validated');
  });
});