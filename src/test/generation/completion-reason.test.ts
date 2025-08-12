import { describe, it, expect } from 'vitest';
import { GENERATION_COMPLETION_REASON_LABELS, getGenerationCompletionLabel } from '@/types/generation';

describe('completion reason labels', () => {
  it('has labels for all known reasons', () => {
    const reasons = Object.keys(GENERATION_COMPLETION_REASON_LABELS);
    expect(reasons.sort()).toEqual([
      'error','first-shiny','max-advances','max-results','stopped'
    ].sort());
  });
  it('returns correct label', () => {
    expect(getGenerationCompletionLabel('max-advances')).toContain('上限');
  });
});
