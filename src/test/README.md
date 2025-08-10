# Test suite overview

This directory contains unit and integration tests for the Gen5 seed searcher. Keep tests practical and integration-oriented; avoid duplicating internal formulas unless necessary.

## Structure
- integration/: End-to-end and cross-module tests
  - pokemon-assembler.test.ts: core assembly logic
  - raw-parser.test.ts: WASM → Raw data parsing
  - wasm-service.test.ts: API adapters, conversions, validation
  - encounter-selection.test.ts: data-driven encounter table checks
  - assembler-sync-rules.integration.test.ts: sync eligibility and special encounters (DustCloud)
- other tests reside next to their subjects under src/test/*

## Conventions
- File name: <feature>.test.ts (integration tests live under integration/)
- Prefer realistic data paths (species/encounters) over re-implemented math
- For sync rules: roaming must never allow sync; static starters/fossils are ineligible; static symbols may allow sync

## WASM fallback (Node)
Node runtime doesn’t fetch WASM directly; tests auto-fallback to TypeScript implementation. Logs may show WASM fetch errors—this is expected and handled.

## How to run
- All tests: npm test
- Rust tests: npm run test:rust (or wasm-pack test)

For more details, see .github/instructions/testing.instructions.md
