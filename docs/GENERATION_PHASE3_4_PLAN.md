# Generation Feature Phase3-4 Plan (Draft)

## 1. Purpose
Add Generation UI (Phase3 subset) and Generation Worker + performance pipeline (Phase4) without disrupting existing Search feature.

## 2. Scope (This PR)
- Generation worker scaffolding (no full WASM loop yet)
- Manager + state wiring
- Generation tab + panel skeleton
- Planning types placeholder

## 3. Out of Scope (Future PRs)
- Full WASM PokemonGenerator batching
- Advanced filters, export formats
- Parallel generation / SIMD benchmarking

## 4. Architecture (High Level)
UI -> GenerationManager -> generation-worker -> WASM(PokemonGenerator) -> raw-parser -> resolver -> UI table

## 5. Initial Tasks (Subset Executed in This Branch Start)
1. Branch create ✅
2. WASM API survey (PokemonGenerator exports) ⏳
3. Worker protocol design (START/PAUSE/RESUME/STOP/PROGRESS/RESULT_BATCH/COMPLETE/ERROR)
4. Type definitions file `src/types/generation.ts`
5. Worker & Manager skeletons
6. Store extension + Generation tab skeleton

## 6. Risks / Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| WASM API shape differs from assumption | Rework | Survey early (Task2) |
| Large batch posting blocks main thread | UI jank | Use modest batch size + postMessage transfer of plain arrays |
| Memory growth in results | Crash | Configurable max results + early stop |
| Shiny stop logic complexity | Delay | Implement after basic loop validated |

## 7. Metrics (Targets)
- Throughput (goal later): >=10k results/sec (wild baseline)
- Progress latency: < 500ms

## 8. Next Immediate Action
Complete WASM API survey and update this document with concrete param/return contract.

---
Draft generated on initial scaffold.
