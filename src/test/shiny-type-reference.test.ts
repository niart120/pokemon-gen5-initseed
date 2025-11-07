import { describe, it, expect, beforeAll } from 'vitest';
import { SHINY_REFERENCE_CASES } from '@/test-utils/reference/shiny-cases';
import { initWasmForTesting } from './wasm-loader';

// ShinyChecker は wasm 初期化後に動的取得
let ShinyChecker: any;

/**
 * Validates provided shiny reference cases against WASM ShinyChecker implementation.
 * Note: We don't yet reproduce PID from seed path here; pid is trusted from external tool.
 */

describe('Shiny reference cases (direct checker)', () => {
  beforeAll(async () => {
    await initWasmForTesting();
    // 動的インポートして default 初期化済みバンドルから ShinyChecker を取得
    const wasmModule = await import('@/wasm/wasm_pkg');
    ShinyChecker = wasmModule.ShinyChecker;
    // 念のため存在確認
    if (!ShinyChecker) {
      throw new Error('ShinyChecker not available after WASM init');
    }
    console.log('🦀 WebAssembly module loaded for testing');
  });

  it('should match expected shiny_type for all reference cases', () => {
    for (const c of SHINY_REFERENCE_CASES) {
      const shinyType = ShinyChecker.check_shiny_type(c.tid, c.sid, c.pid);
      expect(shinyType, `Case seed=0x${c.seed.toString(16)} pid=0x${c.pid.toString(16)}`).toBe(c.expectedType);
    }
  });
});
