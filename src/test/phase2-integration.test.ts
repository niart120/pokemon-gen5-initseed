/**
 * Integration tests for Phase 2 — TypeScript Integration
 * 
 * Tests the complete pipeline from WASM generation to enhanced Pokemon data
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { initWasmForTesting } from './wasm-loader';
import { 
  parseWasmLikeToRawPokemonData,
} from '../lib/integration/raw-parser';
import type { RawPokemonData } from '../types/pokemon-raw';
import { DomainNatureNames } from '../types/domain';
import { determineGenderFromSpec } from '../lib/utils/gender-utils';
import { 
  WasmPokemonService, 
  WasmServiceError,
  type WasmGenerationConfig 
} from '../lib/services/wasm-pokemon-service';
import { buildResolutionContext, enrichForSpecies } from '../lib/initialization/build-resolution-context';
import { resolvePokemon, toUiReadyPokemon, type ResolutionContext } from '../lib/integration/pokemon-resolver';
import { 
  getEncounterTable, 
  getEncounterSlot, 
  calculateLevel,
  validateEncounterTable 
} from '../data/encounter-tables';
import { getGeneratedSpeciesById, selectAbilityBySlot } from '../data/species/generated';

describe('Phase 2 Integration Tests', () => {
  let wasmService: WasmPokemonService;

  beforeAll(async () => {
    await initWasmForTesting();
    wasmService = new WasmPokemonService();
    await wasmService.initialize();
  });

  describe('Task #21: RawPokemonData Parser', () => {
    it('should parse WASM data correctly', async () => {
      const config: WasmGenerationConfig = {
        version: 'B',
        region: 'JPN',
        hardware: 'DS',
        tid: 12345,
        sid: 54321,
        syncEnabled: false,
        syncNatureId: 0,
        macAddress: [0x00, 0x16, 0x56, 0x12, 0x34, 0x56],
        keyInput: 0,
        frame: 1,
      };

      const result = await wasmService.generateSnakeRawPokemon({
        seed: 0x123456789ABCDEFn,
        config,
      });

      expect(result).toBeDefined();
      expect(typeof result.seed).toBe('bigint');
      expect(typeof result.pid).toBe('number');
      expect(result.nature).toBeGreaterThanOrEqual(0);
      expect(result.nature).toBeLessThan(25);
      expect(typeof result.sync_applied).toBe('boolean');
      expect(result.ability_slot).toBeGreaterThanOrEqual(0);
      expect(result.ability_slot).toBeLessThan(2);
      expect(result.gender_value).toBeGreaterThanOrEqual(0);
      expect(result.gender_value).toBeLessThan(256);
      expect(typeof result.encounter_slot_value).toBe('number');
      expect(typeof result.encounter_type).toBe('number');
      expect(typeof result.level_rand_value).toBe('number');
      expect(result.shiny_type).toBeGreaterThanOrEqual(0);
      expect(result.shiny_type).toBeLessThan(3);
    });

    it('should handle invalid WASM data gracefully', () => {
      expect(() => parseWasmLikeToRawPokemonData(null as any)).toThrow('WASM data is null or undefined');
      expect(() => parseWasmLikeToRawPokemonData({})).toThrow('Missing required property or method');
      expect(() => parseWasmLikeToRawPokemonData({ get_seed: 'not a function' } as any)).toThrow('Missing required property or method');
    });

    it('should expose nature names via domain table', () => {
      expect(DomainNatureNames[0]).toBe('Hardy');
      expect(DomainNatureNames[12]).toBe('Serious');
      expect(DomainNatureNames[24]).toBe('Quirky');
    });

    it('should determine gender correctly (femaleThreshold semantics)', () => {
      // Example thresholds:
      // 12.5% female → threshold 31
      expect(determineGenderFromSpec(30, { type: 'ratio', femaleThreshold: 31 })).toBe('Female');
      expect(determineGenderFromSpec(31, { type: 'ratio', femaleThreshold: 31 })).toBe('Male');

      // 50% female → threshold 128
      expect(determineGenderFromSpec(100, { type: 'ratio', femaleThreshold: 128 })).toBe('Female');
      expect(determineGenderFromSpec(150, { type: 'ratio', femaleThreshold: 128 })).toBe('Male');

      // Genderless
      expect(determineGenderFromSpec(100, { type: 'genderless' })).toBe('Genderless');
    });
  });

  describe('Task #22: Encounter Tables', () => {
    it('should have valid encounter table structure', () => {
      const table = getEncounterTable('B', 'Route1', 0); // Normal encounter
      
      if (table) {
        expect(validateEncounterTable(table)).toBe(true);
        expect(table.slots.length).toBeGreaterThan(0);
        expect(table.location).toBeDefined();
        expect(table.method).toBeDefined();
        expect(table.version).toBeDefined();
      }
    });

    it('should calculate levels correctly', () => {
      const levelRange = { min: 5, max: 7 };
      
      // Test deterministic level calculation
      expect(calculateLevel(0, levelRange)).toBe(5);
      expect(calculateLevel(1, levelRange)).toBe(6);
      expect(calculateLevel(2, levelRange)).toBe(7);
      expect(calculateLevel(3, levelRange)).toBe(5); // Wraps around
      
      // Test single level range
      expect(calculateLevel(999, { min: 10, max: 10 })).toBe(10);
    });

    it('should handle encounter slot lookup', () => {
      const table = {
        location: 'Test',
        method: 0,
        version: 'B' as const,
        slots: [
          { speciesId: 1, rate: 50, levelRange: { min: 5, max: 10 } },
          { speciesId: 2, rate: 30, levelRange: { min: 8, max: 12 } },
          { speciesId: 3, rate: 20, levelRange: { min: 10, max: 15 } },
        ]
      };

      expect(getEncounterSlot(table, 0).speciesId).toBe(1);
      expect(getEncounterSlot(table, 1).speciesId).toBe(2);
      expect(getEncounterSlot(table, 2).speciesId).toBe(3);
      
      expect(() => getEncounterSlot(table, 3)).toThrow('Invalid encounter slot');
      expect(() => getEncounterSlot(table, -1)).toThrow('Invalid encounter slot');
    });
  });

  describe('Task #23: Species and Ability Data', () => {
    it('should have valid generated species data', () => {
      const snivy = getGeneratedSpeciesById(495); // Snivy

      expect(snivy).toBeDefined();
      if (snivy) {
        expect(snivy.nationalDex).toBe(495);
        expect(snivy.names.en).toBe('Snivy');
        expect(snivy.baseStats.hp).toBeGreaterThan(0);
        expect(['genderless', 'fixed', 'ratio']).toContain(snivy.gender.type);
        expect(snivy.abilities.ability1).not.toBeNull();
      }
    });

    it('should get species abilities correctly (ability1/ability2 selection)', () => {
      const patrat = getGeneratedSpeciesById(504); // Patrat
      
      expect(patrat).toBeDefined();
      if (patrat) {
        const ability1 = selectAbilityBySlot(0, patrat.abilities);
        const ability2 = selectAbilityBySlot(1, patrat.abilities);
        
        expect(ability1).toBeTruthy();
        expect(ability1?.names.en).toBeDefined();
        
        // ability2 が存在しない種もあるため存在チェックに留める
        if (ability2) {
          expect(ability2.names.en).toBeDefined();
        }
      }
    });

    it('should handle missing species gracefully', () => {
      const unknown = getGeneratedSpeciesById(99999);
      expect(unknown).toBeNull();
    });
  });

  describe('Task #24: WASM Wrapper Service', () => {
    it('should validate generation config', async () => {
      const invalidConfigs = [
        { tid: -1 }, // Invalid TID
        { sid: 70000 }, // Invalid SID
        { syncNatureId: 25 }, // Invalid nature
        { frame: -1 }, // Invalid frame
        { keyInput: 5000 }, // Invalid key input
        { macAddress: [1, 2, 3] }, // Invalid MAC address length
        { macAddress: [1, 2, 3, 4, 5, 256] }, // Invalid MAC address byte
      ];

      for (const invalidConfigPart of invalidConfigs) {
        const config = { 
          ...WasmPokemonService.createDefaultConfig(), 
          ...invalidConfigPart 
        };

        await expect(
          wasmService.generateSnakeRawPokemon({ seed: 1n, config })
        ).rejects.toThrow(WasmServiceError);
      }
    });

    it('should generate batch Pokemon correctly', async () => {
      const config = WasmPokemonService.createDefaultConfig();
      const result = await wasmService.generateSnakeRawBatch({
        seed: 0x123456789ABCDEFn,
        config,
        count: 5,
        offset: 0,
      });

      expect(result.pokemon).toHaveLength(5);
      expect(result.stats.count).toBe(5);
      expect(result.stats.initialSeed).toBe(0x123456789ABCDEFn);

      // Each Pokemon should have different seeds
      const seeds = result.pokemon.map(p => p.seed);
      const uniqueSeeds = new Set(seeds);
      expect(uniqueSeeds.size).toBe(5);
    });

    it('should enforce batch size limits', async () => {
      const config = WasmPokemonService.createDefaultConfig();
      
      await expect(
        wasmService.generateSnakeRawBatch({
          seed: 1n,
          config,
          count: 0,
        })
      ).rejects.toThrow('Invalid count');

      await expect(
        wasmService.generateSnakeRawBatch({
          seed: 1n,
          config,
          count: 15000,
        })
      ).rejects.toThrow('Invalid count');
    });
  });

  describe('Task #25: Data Integration', () => {
    it('should resolve Pokemon data completely (resolver path)', async () => {
      const rawData: RawPokemonData = {
        seed: 0x123456789ABCDEFn,
        pid: 0x12345678,
        nature: 5, // Bold
        sync_applied: false,
        ability_slot: 0,
        gender_value: 100,
        encounter_slot_value: 0,
        encounter_type: 0, // Normal encounter
        level_rand_value: 42,
        shiny_type: 0,
      };

      const ctx: ResolutionContext = buildResolutionContext({
        version: 'B',
        location: 'Route1',
        encounterType: 0 as any,
      });
      let resolved = resolvePokemon(rawData, ctx);
      // Enrich context for gender resolution and re-resolve
      if (resolved.speciesId) {
        enrichForSpecies(ctx, resolved.speciesId);
        resolved = resolvePokemon(rawData, ctx);
      }
      const ui = toUiReadyPokemon(resolved);

      expect(resolved.speciesId).toBeDefined();
      expect(resolved.level).toBeGreaterThan(0);
      expect(['M', 'F', 'N']).toContain(resolved.gender);
      expect(ui.natureName).toBe('Bold');
      expect(ui.shinyStatus).toBe('normal');
    });

    it('should surface synchronize flag (resolver does not mutate nature)', () => {
      const rawData: RawPokemonData = {
        seed: 1n,
        pid: 1,
        nature: 5, // Bold
        sync_applied: true,
        ability_slot: 0,
        gender_value: 100,
        encounter_slot_value: 0,
        encounter_type: 0, // Normal encounter (sync compatible)
        level_rand_value: 42,
        shiny_type: 0,
      };
      const ctx: ResolutionContext = buildResolutionContext({ version: 'B', location: 'Route1', encounterType: 0 as any });
  const resolved = resolvePokemon(rawData, ctx);
  const ui = toUiReadyPokemon(resolved);
  expect(resolved.natureId).toBe(5);
  expect(ui.natureName).toBe('Bold');
  expect(resolved.syncApplied).toBe(true);
    });

    it('should handle missing encounter table gracefully (no species resolution)', () => {
      const invalidRawData: RawPokemonData = {
        seed: 1n,
        pid: 1,
        nature: 5,
        sync_applied: false,
        ability_slot: 0,
        gender_value: 100,
        encounter_slot_value: 0,
        encounter_type: 0,
        level_rand_value: 42,
        shiny_type: 0,
      };

      // Build a context for a non-existent location so table is undefined
      const ctx: ResolutionContext = buildResolutionContext({ version: 'B', location: 'Unknown Location', encounterType: 0 as any });
      const resolved = resolvePokemon(invalidRawData, ctx);
      expect(resolved.speciesId).toBeUndefined();
    });

    it('should validate resolved results against encounter table', () => {
      const ctx: ResolutionContext = buildResolutionContext({ version: 'B', location: 'Route1', encounterType: 0 as any });
      const raw: RawPokemonData = {
        seed: 1n,
        pid: 1,
        nature: 5,
        sync_applied: false,
        ability_slot: 0,
        gender_value: 100,
        encounter_slot_value: 0,
        encounter_type: 0,
        level_rand_value: 42,
        shiny_type: 0,
      };
      const resolved = resolvePokemon(raw, ctx);
      // valid when level is within the encounter slot's range
      const slot = ctx.encounterTable?.species_list[resolved.encounterSlotValue];
      const min = slot?.level_config.min_level ?? 1;
      const max = slot?.level_config.max_level ?? 100;
      expect(resolved.level).toBeGreaterThanOrEqual(min);
      expect(resolved.level).toBeLessThanOrEqual(max);

      // Make invalid by forcing level outside the range and check manually
      const invalidResolved = { ...resolved, level: max + 10 };
      expect(invalidResolved.level! >= min && invalidResolved.level! <= max).toBe(false);
    });
  });

  describe('Task #26: End-to-End Integration', () => {
    it('should complete full WASM to enhanced data pipeline', async () => {
      // Step 1: Generate raw Pokemon data with WASM (snake_case)
      const config = WasmPokemonService.createDefaultConfig();
      const rawData = await wasmService.generateSnakeRawPokemon({
        seed: 0x123456789ABCDEFn,
        config,
      });

      // Step 2: Resolve with encounter tables and species data via context
      const ctx: ResolutionContext = buildResolutionContext({
        version: config.version,
        location: 'Route1',
        encounterType: 0 as any,
      });
      let resolved = resolvePokemon(rawData, ctx);
      if (resolved.speciesId) {
        enrichForSpecies(ctx, resolved.speciesId);
        resolved = resolvePokemon(rawData, ctx);
      }
      const ui = toUiReadyPokemon(resolved);

      // Step 3: Verify complete resolution
      expect(resolved.speciesId).toBeDefined();
      expect(typeof resolved.level).toBe('number');
      expect(resolved.level).toBeGreaterThan(0);
      expect(['M', 'F', 'N']).toContain(resolved.gender);
      expect(ui.natureName).toBeDefined();
      expect(ui.shinyStatus).toBeDefined();
    });

    it('should handle batch processing efficiently', async () => {
      const startTime = performance.now();
      
      // Generate batch of Pokemon
      const config = WasmPokemonService.createDefaultConfig();
      const batchResult = await wasmService.generateSnakeRawBatch({
        seed: 0x123456789ABCDEFn,
        config,
        count: 10,
      });

      // Resolve all Pokemon
  const ctx: ResolutionContext = buildResolutionContext({ version: config.version, location: 'Route1', encounterType: 0 as any });
  // Optionally enrich for all species in the table for gender resolution
  ctx.encounterTable?.species_list.forEach(s => enrichForSpecies(ctx, s.species_id));
  const resolvedAll = batchResult.pokemon.map((r) => resolvePokemon(r, ctx));

      const endTime = performance.now();
      const totalTime = endTime - startTime;

      expect(resolvedAll).toHaveLength(10);
      expect(totalTime).toBeLessThan(1000); // Should complete within 1 second
    });
  });
});