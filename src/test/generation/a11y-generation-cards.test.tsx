import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
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
  validateDraft: vi.fn(),
  validationErrors: [] as string[],
  startGeneration: vi.fn(async () => {}),
  pauseGeneration: vi.fn(),
  resumeGeneration: vi.fn(),
  stopGeneration: vi.fn(),
  status: 'idle' as const,
  lastCompletion: null as any,
  progress: { processedAdvances: 0, totalAdvances: 0 },
  draftParams: { maxAdvances: 0, newGame: false, noSave: false } as any,
  results: [] as any[],
  filters: { natureIds: [], shinyOnly: false, sortField: 'advance', sortOrder: 'asc' } as any,
  applyFilters: vi.fn(),
  resetGenerationFilters: vi.fn(),
  clearResults: vi.fn(),
  setDraftParams: vi.fn((partial: any) => {
    baseState.draftParams = { ...baseState.draftParams, ...partial };
  }),
  encounterField: undefined as string | undefined,
  encounterSpeciesId: undefined as number | undefined,
  encounterTable: undefined as unknown,
  genderRatios: undefined as unknown,
  abilityCatalog: undefined as unknown,
  setEncounterField: vi.fn((value: string | undefined) => {
    baseState.encounterField = value;
  }),
  setEncounterSpeciesId: vi.fn((value: number | undefined) => {
    baseState.encounterSpeciesId = value;
  }),
  setEncounterTable: vi.fn((value: unknown) => {
    baseState.encounterTable = value;
  }),
  setGenderRatios: vi.fn((value: unknown) => {
    baseState.genderRatios = value;
  }),
  setAbilityCatalog: vi.fn((value: unknown) => {
    baseState.abilityCatalog = value;
  }),
};
vi.mock('@/store/app-store', () => {
  const useAppStoreMock: any = (selector?: any) => (selector ? selector(baseState) : baseState);
  useAppStoreMock.getState = () => baseState;
  useAppStoreMock.setState = (updater: any) => {
    const patch = typeof updater === 'function' ? updater(baseState) : updater;
    if (patch && typeof patch === 'object') {
      Object.assign(baseState, patch);
    }
  };
  useAppStoreMock.subscribe = () => () => {};
  useAppStoreMock.destroy = () => {};
  return { useAppStore: useAppStoreMock };
});

// generation-store セレクタ参照をノーオペ化
vi.mock('@/store/generation-store', () => ({
  selectThroughputEma: () => 0,
  selectEtaFormatted: () => null,
  selectShinyCount: () => 0,
  selectFilteredSortedResults: () => [],
  selectUiReadyResults: () => [],
}));

describe('Generation cards accessibility', () => {
  it('has no axe violations (core cards)', async () => {
    const { container } = render(
      <div>
        <GenerationParamsCard />
        <GenerationResultsControlCard />
        <GenerationResultsTableCard />
      </div>
    );
    const results = await axe(container, { rules: { region: { enabled: true } } });
    expect(results).toHaveNoViolations();
  });
});
