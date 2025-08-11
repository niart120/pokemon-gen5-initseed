/**
 * WASM Wrapper Service for Pokemon Generation
 * 
 * This service provides a high-level interface to the WASM pokemon generation
 * functionality with proper TypeScript integration, validation, and error handling.
 * 
 * Architecture principle: Use direct PokemonGenerator for deterministic generation
 */

import { initWasm, getWasm, isWasmReady } from '../core/wasm-interface';
import type { ROMVersion, ROMRegion, Hardware } from '../../types/rom';
// New resolver-path imports (non-breaking additions)
import { parseFromWasmRaw } from '@/lib/integration/raw-parser';
import type { UnresolvedPokemonData } from '@/types/pokemon-raw';
import {
  resolvePokemon,
  resolveBatch,
  toUiReadyPokemon,
  type ResolutionContext,
  type ResolvedPokemonData,
  type UiReadyPokemonData,
} from '@/lib/integration/pokemon-resolver';
import { buildResolutionContext, type BuildContextOptions } from '@/lib/initialization/build-resolution-context';

/**
 * WASM generation configuration
 */
export interface WasmGenerationConfig {
  /** Game version */
  version: ROMVersion;
  /** Game region */
  region: ROMRegion;
  /** Hardware type */
  hardware: Hardware;
  /** Trainer ID */
  tid: number;
  /** Secret ID */
  sid: number;
  /** Enable synchronize */
  syncEnabled: boolean;
  /** Synchronize nature ID (0-24) */
  syncNatureId: number;
  /** MAC address (6 bytes) */
  macAddress: number[];
  /** Key input value */
  keyInput: number;
  /** Frame number for generation */
  frame: number;
}

/**
 * Pokemon generation request
 */
export interface PokemonGenerationRequest {
  /** Initial seed value */
  seed: bigint;
  /** Generation configuration */
  config: WasmGenerationConfig;
  /** Number of Pokemon to generate (for batch operations) */
  count?: number;
  /** Offset from initial seed (for batch operations) */
  offset?: number;
}

/**
 * Pokemon generation result
 */
// (Removed legacy camelCase UI raw result type)

/**
 * Snake_case raw batch result (new API)
 */
export interface PokemonGenerationResultSnake {
  pokemon: UnresolvedPokemonData[];
  stats: {
    generationTime: number;
    count: number;
    initialSeed: bigint;
  };
}

/**
 * WASM service error types
 */
export class WasmServiceError extends Error {
  constructor(
    message: string,
    public code: string,
    public cause?: Error
  ) {
    super(message);
    this.name = 'WasmServiceError';
  }
}

/**
 * WASM Pokemon Generation Service
 * 
 * Provides high-level TypeScript interface to WASM pokemon generation
 * with proper validation, error handling, and type conversion.
 */
export class WasmPokemonService {
  private isInitialized = false;

  /**
   * Initialize the WASM service
   */
  async initialize(): Promise<void> {
    try {
      await initWasm();
      this.isInitialized = true;
    } catch (error) {
      throw new WasmServiceError(
        'Failed to initialize WASM module',
        'WASM_INIT_FAILED',
        error as Error
      );
    }
  }

  /**
   * Check if service is ready
   */
  isReady(): boolean {
    return this.isInitialized && isWasmReady();
  }

  // (Removed legacy camelCase generation methods)

  // ===================== New resolver-backed APIs (non-breaking) =====================

  /** Generate single Pokemon as snake_case RawPokemonData (domain raw) */
  async generateSnakeRawPokemon(request: PokemonGenerationRequest): Promise<UnresolvedPokemonData> {
    this.validateInitialized();
    this.validateGenerationRequest(request);

    const wasm = getWasm();
    const bwConfig = new wasm.BWGenerationConfig(
      this.toGameVersion(request.config.version),
      wasm.EncounterType.Normal,
      request.config.tid,
      request.config.sid,
      request.config.syncEnabled,
      request.config.syncNatureId
    );
    try {
  const wasmRaw = wasm.PokemonGenerator.generate_single_pokemon_bw(
        BigInt.asUintN(64, request.seed),
        bwConfig
      );
  return parseFromWasmRaw(wasmRaw as unknown as Record<string, unknown>);
    } finally {
      bwConfig.free();
    }
  }

  /** Generate batch as snake_case RawPokemonData[] (domain raw) */
  async generateSnakeRawBatch(request: PokemonGenerationRequest): Promise<PokemonGenerationResultSnake> {
    this.validateInitialized();
    this.validateGenerationRequest(request);

    const wasm = getWasm();
    const count = request.count ?? 1;
    const offset = request.offset ?? 0;
    if (count <= 0 || count > 10000) {
      throw new WasmServiceError(`Invalid count: ${count}. Must be between 1 and 10000`, 'INVALID_COUNT');
    }

    const bwConfig = new wasm.BWGenerationConfig(
      this.toGameVersion(request.config.version),
      wasm.EncounterType.Normal,
      request.config.tid,
      request.config.sid,
      request.config.syncEnabled,
      request.config.syncNatureId
    );
    try {
      const baseSeed = BigInt.asUintN(64, request.seed);
      const offsetBig = BigInt.asUintN(64, BigInt(offset));
      const wasmList = wasm.PokemonGenerator.generate_pokemon_batch_bw(baseSeed, offsetBig, count, bwConfig);
      if (!wasmList || wasmList.length === 0) {
        throw new WasmServiceError('No Pokemon generated from WASM batch operation', 'NO_BATCH_RESULTS');
      }
  const pokemon = wasmList.map((w: unknown) => parseFromWasmRaw(w as Record<string, unknown>));
      return {
        pokemon,
        stats: {
          generationTime: 0, // caller can measure if needed
          count: pokemon.length,
          initialSeed: request.seed,
        },
      };
    } finally {
      bwConfig.free();
    }
  }

  /** Generate and resolve a single Pokemon using ResolutionContext */
  async generateResolvedPokemon(
    request: PokemonGenerationRequest,
    ctx: ResolutionContext
  ): Promise<ResolvedPokemonData> {
    const raw = await this.generateSnakeRawPokemon(request);
    return resolvePokemon(raw, ctx);
  }

  /** Generate and resolve a batch using ResolutionContext */
  async generateResolvedBatch(
    request: PokemonGenerationRequest,
    ctx: ResolutionContext
  ): Promise<{ pokemon: ResolvedPokemonData[]; stats: PokemonGenerationResultSnake['stats'] }> {
    const batch = await this.generateSnakeRawBatch(request);
    return {
      pokemon: resolveBatch(batch.pokemon, ctx),
      stats: batch.stats,
    };
  }

  /** Convenience: generate UI-ready (adds labels only) with context options */
  async generateUiReadyPokemon(
    request: PokemonGenerationRequest,
    ctxOrOpts: ResolutionContext | BuildContextOptions
  ): Promise<UiReadyPokemonData> {
    const ctx = (ctxOrOpts as BuildContextOptions).version
      ? buildResolutionContext(ctxOrOpts as BuildContextOptions)
      : (ctxOrOpts as ResolutionContext);
    const resolved = await this.generateResolvedPokemon(request, ctx);
    return toUiReadyPokemon(resolved);
  }

  /**
   * Validate that service is initialized
   */
  private validateInitialized(): void {
    if (!this.isReady()) {
      throw new WasmServiceError(
        'WASM service not initialized. Call initialize() first.',
        'NOT_INITIALIZED'
      );
    }
  }

  /**
   * Validate generation request
   */
  private validateGenerationRequest(request: PokemonGenerationRequest): void {
    if (!request.config) {
      throw new WasmServiceError('Generation config is required', 'MISSING_CONFIG');
    }

    const config = request.config;

    // Validate TID/SID
    if (config.tid < 0 || config.tid > 65535) {
      throw new WasmServiceError(`Invalid TID: ${config.tid}. Must be 0-65535`, 'INVALID_TID');
    }

    if (config.sid < 0 || config.sid > 65535) {
      throw new WasmServiceError(`Invalid SID: ${config.sid}. Must be 0-65535`, 'INVALID_SID');
    }

    // Validate nature ID
    if (config.syncNatureId < 0 || config.syncNatureId > 24) {
      throw new WasmServiceError(
        `Invalid sync nature ID: ${config.syncNatureId}. Must be 0-24`,
        'INVALID_NATURE'
      );
    }

    // Validate frame
    if (config.frame < 0 || config.frame > 1000000) {
      throw new WasmServiceError(
        `Invalid frame: ${config.frame}. Must be 0-1000000`,
        'INVALID_FRAME'
      );
    }

    // Validate key input
    if (config.keyInput < 0 || config.keyInput > 4095) {
      throw new WasmServiceError(
        `Invalid key input: ${config.keyInput}. Must be 0-4095`,
        'INVALID_KEY_INPUT'
      );
    }

    // MACは現状未使用だが形式チェックは残す
    this.validateAndConvertMacAddress(config.macAddress);
  }

  /**
   * Validate and convert MAC address
   */
  private validateAndConvertMacAddress(macAddress: number[]): Uint8Array {
    if (!Array.isArray(macAddress) || macAddress.length !== 6) {
      throw new WasmServiceError(
        'MAC address must be an array of 6 numbers',
        'INVALID_MAC_ADDRESS'
      );
    }

    for (let i = 0; i < 6; i++) {
      if (macAddress[i] < 0 || macAddress[i] > 255 || !Number.isInteger(macAddress[i])) {
        throw new WasmServiceError(
          `Invalid MAC address byte ${i}: ${macAddress[i]}. Must be 0-255`,
          'INVALID_MAC_BYTE'
        );
      }
    }

    return new Uint8Array(macAddress);
  }

  /**
   * Map ROMVersion ('B' | 'W' | 'B2' | 'W2') to GameVersion enum value
   */
  private toGameVersion(version: ROMVersion): number {
    const wasm = getWasm();
    switch (version) {
      case 'B':
        return wasm.GameVersion.B;
      case 'W':
        return wasm.GameVersion.W;
      case 'B2':
        return wasm.GameVersion.B2;
      case 'W2':
        return wasm.GameVersion.W2;
      default:
        return wasm.GameVersion.B;
    }
  }

  /**
   * Create default generation config for testing
   */
  static createDefaultConfig(): WasmGenerationConfig {
    return {
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
  }
}

/**
 * Global WASM service instance
 */
let globalWasmService: WasmPokemonService | null = null;

/**
 * Get or create global WASM service instance
 */
export async function getWasmPokemonService(): Promise<WasmPokemonService> {
  if (!globalWasmService) {
    globalWasmService = new WasmPokemonService();
    await globalWasmService.initialize();
  }
  return globalWasmService;
}

// (Removed legacy utility wrappers)