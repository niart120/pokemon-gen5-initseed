/**
 * Shiny reference cases provided externally (verified by separate tool)
 * Each case: seed, tid, sid, expectedPid, expected shiny type label.
 * The test will validate shiny_type computed via WASM ShinyChecker and worker pipeline result mapping.
 */
export interface ShinyReferenceCase {
  seed: bigint; // initial seed (may be used later to reproduce pid path)
  tid: number;
  sid: number;
  pid: number; // expected PID (after any ID correction rules)
  shinyTypeLabel: 'Square' | 'Star' | 'Normal';
  expectedType: number; // 0 Normal / 1 Square / 2 Star (aligns with ShinyType enum values in Rust)
}

export const SHINY_REFERENCE_CASES: ShinyReferenceCase[] = [
  {
    seed: 0x12345678n,
    tid: 12345,
    sid: 33727,
    pid: 0x47B6F430,
  shinyTypeLabel: 'Square', // shiny_value = 0 -> Square (rarer)
    expectedType: 1,
  },
  {
    seed: 0xDEADBEEFn,
    tid: 22224,
    sid: 56673,
    pid: 0x0A7581C3,
  shinyTypeLabel: 'Star', // shiny_value = 7 -> Star (general shiny)
    expectedType: 2,
  },
  {
    seed: 0x11920827n,
    tid: 0,
    sid: 65535,
    pid: 0x0A7581C3,
  shinyTypeLabel: 'Normal', // shiny_value = 29769 -> Normal
    expectedType: 0,
  },
];
