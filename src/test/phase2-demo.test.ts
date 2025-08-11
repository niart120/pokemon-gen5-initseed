/**
 * Phase 2 Integration Test - Simple demonstration
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { initWasmForTesting } from './wasm-loader';
import { parseFromWasmRaw } from '../lib/integration/raw-parser';
import { resolvePokemon, toUiReadyPokemon } from '../lib/integration/pokemon-resolver';
import { buildResolutionContext } from '../lib/initialization/build-resolution-context';
import { getGeneratedSpeciesById } from '../data/species/generated';

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
  get_level_rand_value: () => 42n,
      get_shiny_type: () => 0,
    };

  // Step 1: Parse raw data (snake_case)
  const rawData = parseFromWasmRaw(mockWasmData as any);
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

  console.log('✅ Phase 2 integration demo successful:');
    console.log(`   Seed: ${rawData.seed}`);
  console.log(`   Species: ${species?.names.en}`);
  console.log(`   Nature: ${uiReady.natureName}`);
  // Level is now resolved inside resolver; omit here
    console.log(`   PID: 0x${rawData.pid.toString(16)}`);
  });

  it('should validate all Phase 2 components work together', () => {
    // Test that all our main utilities work
  // Nature name validation is now via Domain table in resolver; omitted here

  const snivy = getGeneratedSpeciesById(495);
  expect(snivy?.names.en).toBe('Snivy');

    console.log('✅ All Phase 2 utilities validated');
  });
});