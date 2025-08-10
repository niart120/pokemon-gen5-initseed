/**
 * Pokemon Data Integration Service
 * 
 * This service combines raw WASM Pokemon data with encounter table information
 * and species data to produce complete, enhanced Pokemon information.
 * 
 * Architecture: Integrates WASM output + encounter tables + species data
 */

import type { RawPokemonData, EnhancedPokemonData, EncounterDetails } from '../../types/raw-pokemon-data';
import type { ROMVersion } from '../../types/pokemon';
import { 
  getShinyStatusName, 
  getEncounterTypeName, 
  getNatureName // added: for mapping nature ID to name
  // determineGender // removed: use gender-utils with generated dataset
} from '../../types/raw-pokemon-data';
import { 
  getEncounterTable, 
  getEncounterSlot, 
  calculateLevel, 
  // getDefaultEncounterTable, // removed: JSON datasets only, no fallback
  type EncounterTable 
} from '../../data/encounter-tables';
// Replace TS species dataset with generated JSON adapter
import { getGeneratedSpeciesById, selectAbilityBySlot } from '../../data/species/generated';
import { determineGenderFromSpec } from './gender-utils';

/**
 * Integration configuration
 */
export interface IntegrationConfig {
  /** Game version for encounter table lookup */
  version: ROMVersion;
  /** Default location if encounter table not found */
  defaultLocation?: string;
  /** Whether to apply synchronize effects */
  applySynchronize?: boolean;
  /** Synchronize nature ID (if applicable) */
  synchronizeNature?: number;
}

/**
 * Integration result with metadata
 */
export interface IntegrationResult {
  /** Enhanced Pokemon data */
  pokemon: EnhancedPokemonData;
  /** Integration metadata */
  metadata: {
    /** Whether encounter table was found */
    encounterTableFound: boolean;
    /** Whether species data was found */
    speciesDataFound: boolean;
    /** Whether ability data was found */
    abilityDataFound: boolean;
    /** Integration warnings */
    warnings: string[];
  };
}

/**
 * Integration service error
 */
export class IntegrationError extends Error {
  constructor(
    message: string,
    public code: string,
    public data?: any
  ) {
    super(message);
    this.name = 'IntegrationError';
  }
}

/**
 * Pokemon Data Integration Service
 * 
 * Combines WASM raw data with encounter tables and species information
 * to create complete Pokemon data suitable for display and analysis.
 */
export class PokemonIntegrationService {
  /**
   * Integrate single Pokemon data
   * 
   * @param rawData Raw Pokemon data from WASM
   * @param config Integration configuration
   * @returns Enhanced Pokemon data with metadata
   */
  integratePokemon(
    rawData: RawPokemonData,
    config: IntegrationConfig
  ): IntegrationResult {
    const warnings: string[] = [];
    let encounterTableFound = false;
    let speciesDataFound = false;
    let abilityDataFound = false;

    try {
      // Step 1: Get encounter information
      const encounterInfo = this.resolveEncounterInfo(rawData, config);
      if (encounterInfo.table) {
        encounterTableFound = true;
      }

      // Step 2: Get species information
      const slot = getEncounterSlot(encounterInfo.table, rawData.encounterSlotValue);
      const gSpecies = getGeneratedSpeciesById(slot.speciesId);
      if (gSpecies) {
        speciesDataFound = true;
      } else {
        warnings.push(`Species data not found for ID ${slot.speciesId}`);
        throw new IntegrationError(
          `Cannot integrate Pokemon without species data for ID ${slot.speciesId}`,
          'MISSING_SPECIES_DATA',
          { speciesId: slot.speciesId }
        );
      }

      // Step 3: Get ability information (ability1/2 only)
      const abilityNode = selectAbilityBySlot(rawData.abilitySlot, gSpecies.abilities);
      if (abilityNode) {
        abilityDataFound = true;
      } else {
        warnings.push(`Ability data not found for slot ${rawData.abilitySlot} of species ${gSpecies.names.en}`);
      }

      // Step 4: Calculate derived values
      const gender = determineGenderFromSpec(rawData.genderValue, gSpecies.gender);
      const level = calculateLevel(rawData.levelRandValue, slot.levelRange);
      const shinyStatus = getShinyStatusName(rawData.shinyType);

      // Step 5: Apply synchronize effects if applicable
      const finalNature = this.applySynchronizeIfNeeded(rawData, config);

      // Step 6: Create enhanced Pokemon data
      const enhancedPokemon: EnhancedPokemonData = {
        ...rawData,
        // Map minimal species info expected by EnhancedPokemonData
        species: {
          nationalDex: gSpecies.nationalDex,
          name: gSpecies.names.en,
          baseStats: gSpecies.baseStats,
          // types are not present in generated dataset; provide placeholder single-type for now
          // Consumers relying on types should be updated in a later phase
          types: ['Normal'],
          // genderRatio in old model is percent male or -1; we cannot map accurately here, set -1 for genderless else 0
          genderRatio: gSpecies.gender.type === 'genderless' ? -1 : 0,
          abilities: {
            ability1: gSpecies.abilities.ability1?.names.en || 'Unknown',
            ability2: gSpecies.abilities.ability2?.names.en || undefined,
            hiddenAbility: gSpecies.abilities.hidden?.names.en || undefined,
          },
        },
        ability: abilityNode ? { name: abilityNode.names.en, description: '', isHidden: false } : {
          name: 'Unknown', description: 'Ability data not available', isHidden: false
        },
        gender,
        level,
        encounter: encounterInfo.details,
        natureName: getNatureName(finalNature.applied ? finalNature.natureId : rawData.nature),
        shinyStatus,
      };

      // Update nature if synchronize was applied
      if (finalNature.applied) {
        enhancedPokemon.nature = finalNature.natureId;
        enhancedPokemon.natureName = finalNature.name;
        enhancedPokemon.syncApplied = true;
        warnings.push(`Synchronize applied: nature changed to ${finalNature.name}`);
      }

      return {
        pokemon: enhancedPokemon,
        metadata: {
          encounterTableFound,
          speciesDataFound,
          abilityDataFound,
          warnings,
        },
      };
    } catch (error) {
      if (error instanceof IntegrationError) {
        throw error;
      }
      throw new IntegrationError(
        `Failed to integrate Pokemon data: ${error}`,
        'INTEGRATION_FAILED',
        { rawData, config, error }
      );
    }
  }

  /**
   * Integrate batch of Pokemon data
   * 
   * @param rawDataArray Array of raw Pokemon data
   * @param config Integration configuration
   * @returns Array of integration results
   */
  integratePokemonBatch(
    rawDataArray: RawPokemonData[],
    config: IntegrationConfig
  ): IntegrationResult[] {
    return rawDataArray.map(rawData => this.integratePokemon(rawData, config));
  }

  /**
   * Resolve encounter information for Pokemon
   */
  private resolveEncounterInfo(
    rawData: RawPokemonData,
    config: IntegrationConfig
  ): { table: EncounterTable; details: EncounterDetails } {
    // Lookup encounter table strictly from JSON datasets
    const location = config.defaultLocation || 'Unknown Location';
    const table = getEncounterTable(config.version, location, rawData.encounterType);

    if (!table) {
      // JSON-only policy: do not fallback to any TS defaults
      throw new IntegrationError(
        `Encounter table not found for ${config.version} ${location} type ${rawData.encounterType}`,
        'MISSING_ENCOUNTER_TABLE',
        { version: config.version, location, encounterType: rawData.encounterType }
      );
    }

    // Create encounter details
    const details: EncounterDetails = {
      method: getEncounterTypeName(rawData.encounterType),
      location: table.location,
      rate: table.slots[rawData.encounterSlotValue]?.rate,
      levelRange: table.slots[rawData.encounterSlotValue]?.levelRange || { min: 1, max: 100 },
    };

    return { table, details };
  }

  /**
   * Apply synchronize effects if needed
   */
  private applySynchronizeIfNeeded(
    rawData: RawPokemonData,
    config: IntegrationConfig
  ): { applied: boolean; natureId: number; name: string } {
    // Check if synchronize should be applied
    if (
      config.applySynchronize && 
      config.synchronizeNature !== undefined &&
      rawData.syncApplied &&
      this.isSynchronizeCompatibleEncounter(rawData.encounterType)
    ) {
      return {
        applied: true,
        natureId: config.synchronizeNature,
        name: getNatureName(config.synchronizeNature),
      };
    }

    // Return original nature
    return {
      applied: false,
      natureId: rawData.nature,
      name: getNatureName(rawData.nature),
    };
  }

  /**
   * Check if encounter type supports synchronize
   */
  private isSynchronizeCompatibleEncounter(encounterType: number): boolean {
    // Synchronize works on wild encounters only (exclude static/event/roaming)
    const syncCompatibleTypes = [0, 1, 2, 3, 4, 5, 6, 7];
    return syncCompatibleTypes.includes(encounterType);
  }

  /**
   * Validate integration result
   */
  validateIntegrationResult(result: IntegrationResult): boolean {
    const pokemon = result.pokemon;

    // Basic validation
    if (!pokemon.species || !pokemon.encounter) {
      return false;
    }

    // Validate level is within encounter range
    if (
      pokemon.level < pokemon.encounter.levelRange.min ||
      pokemon.level > pokemon.encounter.levelRange.max
    ) {
      return false;
    }

    // Validate nature ID
    if (pokemon.nature < 0 || pokemon.nature > 24) {
      return false;
    }

    // Validate ability slot
    if (pokemon.abilitySlot < 0 || pokemon.abilitySlot > 1) {
      return false;
    }

    return true;
  }

  /**
   * Get integration statistics
   */
  getIntegrationStats(results: IntegrationResult[]): {
    total: number;
    encounterTablesFound: number;
    speciesDataFound: number;
    abilityDataFound: number;
    totalWarnings: number;
    validResults: number;
  } {
    const stats = {
      total: results.length,
      encounterTablesFound: 0,
      speciesDataFound: 0,
      abilityDataFound: 0,
      totalWarnings: 0,
      validResults: 0,
    };

    for (const result of results) {
      if (result.metadata.encounterTableFound) stats.encounterTablesFound++;
      if (result.metadata.speciesDataFound) stats.speciesDataFound++;
      if (result.metadata.abilityDataFound) stats.abilityDataFound++;
      stats.totalWarnings += result.metadata.warnings.length;
      if (this.validateIntegrationResult(result)) stats.validResults++;
    }

    return stats;
  }
}

/**
 * Global integration service instance
 */
let globalIntegrationService: PokemonIntegrationService | null = null;

/**
 * Get global integration service instance
 */
export function getIntegrationService(): PokemonIntegrationService {
  if (!globalIntegrationService) {
    globalIntegrationService = new PokemonIntegrationService();
  }
  return globalIntegrationService;
}

/**
 * Utility function for quick Pokemon integration
 */
export function integratePokemon(
  rawData: RawPokemonData,
  config: IntegrationConfig
): IntegrationResult {
  const service = getIntegrationService();
  return service.integratePokemon(rawData, config);
}

/**
 * Utility function for quick batch integration
 */
export function integratePokemonBatch(
  rawDataArray: RawPokemonData[],
  config: IntegrationConfig
): IntegrationResult[] {
  const service = getIntegrationService();
  return service.integratePokemonBatch(rawDataArray, config);
}

/**
 * Create default integration config
 */
export function createDefaultIntegrationConfig(): IntegrationConfig {
  return {
    version: 'B',
    defaultLocation: 'Route 1',
    applySynchronize: false,
  };
}