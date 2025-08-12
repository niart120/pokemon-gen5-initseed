import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { GenerationControlCard } from '@/components/generation/GenerationControlCard';
import { GenerationProgressCard } from '@/components/generation/GenerationProgressCard';
import { GenerationParamsCard } from '@/components/generation/GenerationParamsCard';
import { GenerationResultsControlCard } from '@/components/generation/GenerationResultsControlCard';
import { GenerationResultsTableCard } from '@/components/generation/GenerationResultsTableCard';

expect.extend(toHaveNoViolations);
declare module 'vitest' {
  interface Assertion<T = any> {
    toHaveNoViolations(): T;
  }
}

// シンプルなストアモック: UI要求される最小のshapeのみ
const baseState = {
  validateDraft: () => {},
  validationErrors: [] as string[],
  startGeneration: async () => {},
  pauseGeneration: () => {},
  resumeGeneration: () => {},
  stopGeneration: () => {},
  status: 'idle',
  lastCompletion: null as any,
  progress: { processedAdvances: 0, totalAdvances: 0 },
  draftParams: { maxAdvances: 0 } as any,
  results: [] as any[],
  filters: { natureIds: [], shinyOnly: false, sortField: 'advance', sortOrder: 'asc' } as any,
  applyFilters: () => {},
  resetGenerationFilters: () => {},
  clearResults: () => {},
  setDraftParams: () => {},
};
vi.mock('@/store/app-store', () => ({
  useAppStore: (selector?: any) => selector ? selector(baseState) : baseState,
}));

// generation-store セレクタ参照をノーオペ化
vi.mock('@/store/generation-store', () => ({
  selectThroughputEma: () => 0,
  selectEtaFormatted: () => null,
  selectShinyCount: () => 0,
  selectFilteredSortedResults: () => [],
}));

describe('Generation cards accessibility', () => {
  it('has no axe violations (core cards)', async () => {
    const { container } = render(
      <div>
        <GenerationControlCard />
        <GenerationProgressCard />
        <GenerationParamsCard />
        <GenerationResultsControlCard />
        <GenerationResultsTableCard />
      </div>
    );
    const results = await axe(container, { rules: { region: { enabled: true } } });
    expect(results).toHaveNoViolations();
  });
});
