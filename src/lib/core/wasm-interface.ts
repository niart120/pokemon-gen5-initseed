/**
 * WebAssembly module wrapper for Pokemon BW/BW2 seed calculation
 * This module provides high-performance calculation functions using Rust + WebAssembly
 */

// Type-only imports from generated wasm types
import type {
  BWGenerationConfig as WasmBWGenerationConfig,
  PokemonGenerator as WasmPokemonGenerator,
  SeedEnumerator as WasmSeedEnumerator,
  EncounterType as WasmEncounterType,
  GameVersion as WasmGameVersion,
  GameMode as WasmGameMode,
  // Egg generation types
  EggSeedEnumeratorJs as WasmEggSeedEnumeratorJs,
  ParentsIVsJs as WasmParentsIVsJs,
  GenerationConditionsJs as WasmGenerationConditionsJs,
  EverstonePlanJs as WasmEverstonePlanJs,
  IndividualFilterJs as WasmIndividualFilterJs,
  TrainerIds as WasmTrainerIds,
  GenderRatio as WasmGenderRatio,
  StatRange as WasmStatRange,
  // Egg boot timing search types
  EggBootTimingSearchResult as WasmEggBootTimingSearchResult,
  EggBootTimingSearchIterator as WasmEggBootTimingSearchIterator,
  // Search common types (for boot timing search parameters)
  DSConfigJs as WasmDSConfigJs,
  SegmentParamsJs as WasmSegmentParamsJs,
  TimeRangeParamsJs as WasmTimeRangeParamsJs,
  SearchRangeParamsJs as WasmSearchRangeParamsJs,
  // MT Seed boot timing search types
  MtSeedBootTimingSearchIterator as WasmMtSeedBootTimingSearchIterator,
  MtSeedBootTimingSearchResult as WasmMtSeedBootTimingSearchResult,
  MtSeedBootTimingSearchResults as WasmMtSeedBootTimingSearchResults,
} from '../../wasm/wasm_pkg';

// Init arg for wasm-bindgen init function
type WasmInitArg = { module_or_path: BufferSource | URL };

// WebAssembly module interface - ポケモン生成API
export interface WasmModule {
  // ポケモン生成API
  BWGenerationConfig: typeof WasmBWGenerationConfig;
  PokemonGenerator: typeof WasmPokemonGenerator;
  SeedEnumerator: typeof WasmSeedEnumerator;

  // 追加: 列挙（数値）
  EncounterType: typeof WasmEncounterType;
  GameVersion: typeof WasmGameVersion;
  GameMode: typeof WasmGameMode;

  // 追加: タマゴ生成API
  EggSeedEnumeratorJs: typeof WasmEggSeedEnumeratorJs;
  ParentsIVsJs: typeof WasmParentsIVsJs;
  GenerationConditionsJs: typeof WasmGenerationConditionsJs;
  EverstonePlanJs: typeof WasmEverstonePlanJs;
  IndividualFilterJs: typeof WasmIndividualFilterJs;
  TrainerIds: typeof WasmTrainerIds;
  GenderRatio: typeof WasmGenderRatio;
  StatRange: typeof WasmStatRange;

  // 追加: 孵化乱数起動時間検索API
  EggBootTimingSearchResult: typeof WasmEggBootTimingSearchResult;
  EggBootTimingSearchIterator: typeof WasmEggBootTimingSearchIterator;

  // 追加: 検索共通パラメータ型 (Boot Timing Search)
  DSConfigJs: typeof WasmDSConfigJs;
  SegmentParamsJs: typeof WasmSegmentParamsJs;
  TimeRangeParamsJs: typeof WasmTimeRangeParamsJs;
  SearchRangeParamsJs: typeof WasmSearchRangeParamsJs;

  // 追加: MT Seed起動時間検索API
  MtSeedBootTimingSearchIterator: typeof WasmMtSeedBootTimingSearchIterator;
  MtSeedBootTimingSearchResult: typeof WasmMtSeedBootTimingSearchResult;
  MtSeedBootTimingSearchResults: typeof WasmMtSeedBootTimingSearchResults;

  calculate_game_offset(initial_seed: bigint, mode: number): number;
  sha1_hash_batch(messages: Uint32Array): Uint32Array;
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
        BWGenerationConfig: module.BWGenerationConfig,
        PokemonGenerator: module.PokemonGenerator,
        SeedEnumerator: module.SeedEnumerator,
        EncounterType: module.EncounterType,
        GameVersion: module.GameVersion,
        GameMode: module.GameMode,
        // タマゴ生成API
        EggSeedEnumeratorJs: module.EggSeedEnumeratorJs,
        ParentsIVsJs: module.ParentsIVsJs,
        GenerationConditionsJs: module.GenerationConditionsJs,
        EverstonePlanJs: module.EverstonePlanJs,
        IndividualFilterJs: module.IndividualFilterJs,
        TrainerIds: module.TrainerIds,
        GenderRatio: module.GenderRatio,
        StatRange: module.StatRange,
        // 孵化乱数起動時間検索API
        EggBootTimingSearchResult: module.EggBootTimingSearchResult,
        EggBootTimingSearchIterator: module.EggBootTimingSearchIterator,
        // 検索共通パラメータ型
        DSConfigJs: module.DSConfigJs,
        SegmentParamsJs: module.SegmentParamsJs,
        TimeRangeParamsJs: module.TimeRangeParamsJs,
        SearchRangeParamsJs: module.SearchRangeParamsJs,
        // MT Seed起動時間検索API
        MtSeedBootTimingSearchIterator: module.MtSeedBootTimingSearchIterator,
        MtSeedBootTimingSearchResult: module.MtSeedBootTimingSearchResult,
        MtSeedBootTimingSearchResults: module.MtSeedBootTimingSearchResults,
        calculate_game_offset: module.calculate_game_offset,
        sha1_hash_batch: module.sha1_hash_batch,
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

// 再利用しやすいよう wasm ctor/instance 型を公開
export type BWGenerationConfigCtor = typeof WasmBWGenerationConfig;
export type SeedEnumeratorCtor = typeof WasmSeedEnumerator;
export type SeedEnumeratorInstance = InstanceType<SeedEnumeratorCtor>;

// Boot Timing Search 用パラメータ型を公開
export type DSConfigJs = WasmDSConfigJs;
export type SegmentParamsJs = WasmSegmentParamsJs;
export type TimeRangeParamsJs = WasmTimeRangeParamsJs;
export type SearchRangeParamsJs = WasmSearchRangeParamsJs;

// MT Seed Boot Timing Search 型を公開
export type MtSeedBootTimingSearchIterator = WasmMtSeedBootTimingSearchIterator;
export type MtSeedBootTimingSearchResult = WasmMtSeedBootTimingSearchResult;
export type MtSeedBootTimingSearchResults = WasmMtSeedBootTimingSearchResults;
