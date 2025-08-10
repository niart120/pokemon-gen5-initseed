This folder contains the JSON-driven encounter data loader and schema.

Use these entry points:
- Encounter tables (public): `src/data/encounter-tables.ts`
- JSON loader internals: `src/data/encounters/loader.ts`, `schema.ts`

The legacy sample modules (types.ts, rates.ts, tables.ts) have been removed.
Rely on the domain enums via `src/types/pokemon-enhanced.ts` (re-export) or import directly from `src/types/domain.ts`.
