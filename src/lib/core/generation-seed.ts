import { initWasm, getWasm, isWasmReady } from './wasm-interface';

const U64_MAX = 0xffff_ffff_ffff_ffffn;

function assertU64(value: bigint, label: string): void {
  if (value < 0n || value > U64_MAX) {
    throw new RangeError(`${label} out of u64 range`);
  }
}

/**
 * Calculate the seed used for the first generated Pokémon after applying an advance offset.
 */
export async function calculateGenerationStartSeed(initialSeed: bigint, offset: bigint): Promise<bigint> {
  assertU64(initialSeed, 'initialSeed');
  assertU64(offset, 'offset');
  if (!isWasmReady()) {
    await initWasm();
  }
  const wasm = getWasm();
  return wasm.PokemonGenerator.calculate_generation_seed(initialSeed, offset);
}
