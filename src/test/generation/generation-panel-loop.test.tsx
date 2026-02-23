import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { LocaleProvider } from '@/lib/i18n/locale-context';
import { GenerationFilterCard } from '@/components/generation/GenerationFilterCard';
import { GenerationResultsTableCard } from '@/components/generation/GenerationResultsTableCard';
import { useAppStore } from '@/store/app-store';
import { createDefaultGenerationFilters } from '@/store/generation-store';
import { resolveBatch } from '@/lib/generation/pokemon-resolver';
import type { ResolutionContext } from '@/types/pokemon-resolved';

function ensureMatchMedia() {
  if (typeof window.matchMedia !== 'function') {
    window.matchMedia = ((query: string) => {
      return {
        matches: query.includes('max-width: 767px') ? false : true,
        media: query,
        addEventListener: () => void 0,
        removeEventListener: () => void 0,
        addListener: () => void 0,
        removeListener: () => void 0,
        onchange: null,
        dispatchEvent: () => false,
      } as MediaQueryList;
    }) as typeof window.matchMedia;
  }
}

describe('GenerationPanel', () => {
  const sampleResults = [
    { advance: 1, shiny_type: 0, seed: 1n, pid: 0x00000001, nature: 3, sync_applied: false, ability_slot: 0, gender_value: 0, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
    { advance: 5, shiny_type: 2, seed: 5n, pid: 0x00000005, nature: 7, sync_applied: false, ability_slot: 1, gender_value: 32, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
    { advance: 10, shiny_type: 0, seed: 10n, pid: 0x0000000a, nature: 12, sync_applied: false, ability_slot: 2, gender_value: 128, encounter_slot_value: 0, encounter_type: 0, level_rand_value: 0n },
  ];
  const resolutionContext: ResolutionContext = {};

  beforeEach(() => {
    ensureMatchMedia();
    if (typeof window.requestAnimationFrame !== 'function') {
      window.requestAnimationFrame = ((cb: FrameRequestCallback) => {
        return setTimeout(() => cb(performance.now()), 16) as unknown as number;
      }) as typeof window.requestAnimationFrame;
    }
    if (typeof window.cancelAnimationFrame !== 'function') {
      window.cancelAnimationFrame = ((id: number) => {
        clearTimeout(id as unknown as ReturnType<typeof setTimeout>);
      }) as typeof window.cancelAnimationFrame;
    }
    cleanup();
    const resolvedResults = resolveBatch(sampleResults as any, resolutionContext);
    useAppStore.setState({
      results: sampleResults.map((entry) => ({ ...entry })),
      resolvedResults,
      filters: createDefaultGenerationFilters(),
      encounterTable: undefined,
      genderRatios: undefined,
      abilityCatalog: undefined,
    });
  });

  it('renders without triggering react update depth errors', () => {
    try {
      render(
        <LocaleProvider>
          <div>
            <GenerationFilterCard />
            <GenerationResultsTableCard />
          </div>
        </LocaleProvider>
      );
      expect(true).toBe(true);
    } catch (error) {
      if (error instanceof Error) {
        console.error('GenerationPanel render error stack:\n', error.stack);
      }
      throw error;
    }
  });
});
