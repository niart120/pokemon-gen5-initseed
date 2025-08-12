/**
 * WebAssembly module wrapper for Pokemon BW/BW2 seed calculation
 * This module provides high-performance calculation functions using Rust + WebAssembly
 */

// Type-only imports from generated wasm types
import type {
  BWGenerationConfig as WasmBWGenerationConfig,
  PokemonGenerator as WasmPokemonGenerator,
  EncounterType as WasmEncounterType,
  GameVersion as WasmGameVersion,
} from '../../wasm/wasm_pkg';
// Local type alias for internal interface references
type WasmSearchResult = import('../../wasm/wasm_pkg').SearchResult;

// Init arg for wasm-bindgen init function
type WasmInitArg = { module_or_path: BufferSource | URL };

// WebAssembly module interface - 統合検索とポケモン生成API
export interface WasmModule {
  // 統合検索機能（従来実装）
  IntegratedSeedSearcher: new (
    mac: Uint8Array,
    nazo: Uint32Array,
    hardware: string,
    key_input: number,
    frame: number
  ) => {
    search_seeds_integrated_simd(
      year_start: number,
      month_start: number,
      date_start: number,
      hour_start: number,
      minute_start: number,
      second_start: number,
      range_seconds: number,
      timer0_min: number,
      timer0_max: number,
      vcount_min: number,
      vcount_max: number,
      target_seeds: Uint32Array
    ): WasmSearchResult[];
    free(): void;
  };

  // 追加: ポケモン生成API
  BWGenerationConfig: typeof WasmBWGenerationConfig;
  PokemonGenerator: typeof WasmPokemonGenerator;

  // 追加: 列挙（数値）
  EncounterType: typeof WasmEncounterType;
  GameVersion: typeof WasmGameVersion;
}

let wasmModule: WasmModule | null = null;
let wasmPromise: Promise<WasmModule> | null = null;

/**
 * Initialize WebAssembly module
 */
export async function initWasm(): Promise<WasmModule> {
  if (wasmModule) {
    return wasmModule;
  }

  if (wasmPromise) {
    return wasmPromise;
  }

  wasmPromise = (async (): Promise<WasmModule> => {
    try {
      // Import the WebAssembly module
      const module = await import('../../wasm/wasm_pkg.js');

      // Node(vitest) 環境では fetch が file: URL をサポートしないため、
      // 可能ならバイト列を直接渡す。Web/Worker では URL を渡す。
      let initArg: WasmInitArg;
      const isNode = typeof process !== 'undefined' && !!(process as NodeJS.Process).versions?.node;
      if (isNode) {
        const fs = await import('fs');
        const path = await import('path');
        const wasmPath = path.join(process.cwd(), 'src/wasm/wasm_pkg_bg.wasm');
        const bytes = fs.readFileSync(wasmPath);
        initArg = { module_or_path: bytes };
      } else {
        // ブラウザ/Worker 環境（window が未定義でも WorkerGlobalScope で動作）
        initArg = { module_or_path: new URL('../../wasm/wasm_pkg_bg.wasm', import.meta.url) };
      }

      await module.default(initArg);
      
      wasmModule = {
        IntegratedSeedSearcher: module.IntegratedSeedSearcher,
        BWGenerationConfig: module.BWGenerationConfig,
        PokemonGenerator: module.PokemonGenerator,
        EncounterType: module.EncounterType,
        GameVersion: module.GameVersion,
      } as unknown as WasmModule;
      
      return wasmModule;
    } catch (error) {
      console.error('Failed to load WebAssembly module:', error);
      wasmModule = null;
      wasmPromise = null;
      throw error;
    }
  })();

  return wasmPromise;
}

/**
 * Get WebAssembly module (must be initialized first)
 */
export function getWasm(): WasmModule {
  if (!wasmModule) {
    throw new Error('WebAssembly module not initialized. Call initWasm() first.');
  }
  return wasmModule;
}

/**
 * Check if WebAssembly is available and initialized
 */
export function isWasmReady(): boolean {
  return wasmModule !== null;
}

// Export alias type for consumers without importing wasm_pkg directly
export type { WasmSearchResult };
