import { describe, it } from 'vitest';

// Legacy RawPokemonDataParser and placeholder utilities were removed in favor of a single
// WASM boundary adapter (parseFromWasmRaw) and the resolver-first pipeline.
// These tests are intentionally skipped and kept only as a breadcrumb for migration history.

describe.skip('Legacy raw-parser tests (removed)', () => {
  it('removed in resolver-first architecture', () => {
    // no-op
  });
});

export {};