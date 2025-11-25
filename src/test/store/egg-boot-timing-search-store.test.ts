import { describe, it, expect, beforeEach } from 'vitest';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { createDefaultEggBootTimingSearchParams } from '@/types/egg-boot-timing-search';
import type { DeviceProfile } from '@/types/profile';

describe('egg-boot-timing-search-store', () => {
  beforeEach(() => {
    // Reset store to default state before each test
    useEggBootTimingSearchStore.getState().reset();
  });

  describe('initial state', () => {
    it('should have idle status initially', () => {
      expect(useEggBootTimingSearchStore.getState().status).toBe('idle');
    });

    it('should have empty results initially', () => {
      expect(useEggBootTimingSearchStore.getState().results).toHaveLength(0);
    });

    it('should have empty pending results initially', () => {
      expect(useEggBootTimingSearchStore.getState()._pendingResults).toHaveLength(0);
    });

    it('should have null progress initially', () => {
      expect(useEggBootTimingSearchStore.getState().progress).toBeNull();
    });
  });

  describe('updateDraftParams', () => {
    it('should update dateRange', () => {
      const { updateDateRange } = useEggBootTimingSearchStore.getState();
      updateDateRange({ startYear: 2025, startMonth: 6, startDay: 15 });
      const { dateRange } = useEggBootTimingSearchStore.getState().draftParams;
      expect(dateRange.startYear).toBe(2025);
      expect(dateRange.startMonth).toBe(6);
      expect(dateRange.startDay).toBe(15);
    });

    it('should update keyInputMask', () => {
      const { updateDraftParams } = useEggBootTimingSearchStore.getState();
      updateDraftParams({ keyInputMask: 0x0100 });
      expect(useEggBootTimingSearchStore.getState().draftParams.keyInputMask).toBe(0x0100);
    });

    it('should update advanceCount', () => {
      const { updateDraftParams } = useEggBootTimingSearchStore.getState();
      updateDraftParams({ advanceCount: 500 });
      expect(useEggBootTimingSearchStore.getState().draftParams.advanceCount).toBe(500);
    });

    it('should update filterDisabled', () => {
      const { updateDraftParams } = useEggBootTimingSearchStore.getState();
      updateDraftParams({ filterDisabled: true });
      expect(useEggBootTimingSearchStore.getState().draftParams.filterDisabled).toBe(true);
    });
  });

  describe('updateDraftConditions', () => {
    it('should update tid', () => {
      const { updateDraftConditions } = useEggBootTimingSearchStore.getState();
      updateDraftConditions({ tid: 12345 });
      expect(useEggBootTimingSearchStore.getState().draftParams.conditions.tid).toBe(12345);
    });

    it('should update sid', () => {
      const { updateDraftConditions } = useEggBootTimingSearchStore.getState();
      updateDraftConditions({ sid: 54321 });
      expect(useEggBootTimingSearchStore.getState().draftParams.conditions.sid).toBe(54321);
    });

    it('should update masudaMethod', () => {
      const { updateDraftConditions } = useEggBootTimingSearchStore.getState();
      updateDraftConditions({ masudaMethod: true });
      expect(useEggBootTimingSearchStore.getState().draftParams.conditions.masudaMethod).toBe(true);
    });
  });

  describe('updateDraftParents', () => {
    it('should update male parent IVs', () => {
      const { updateDraftParents } = useEggBootTimingSearchStore.getState();
      updateDraftParents({ male: [0, 1, 2, 3, 4, 5] });
      expect(useEggBootTimingSearchStore.getState().draftParams.parents.male).toEqual([0, 1, 2, 3, 4, 5]);
    });

    it('should update female parent IVs', () => {
      const { updateDraftParents } = useEggBootTimingSearchStore.getState();
      updateDraftParents({ female: [10, 20, 30, 31, 31, 31] });
      expect(useEggBootTimingSearchStore.getState().draftParams.parents.female).toEqual([10, 20, 30, 31, 31, 31]);
    });
  });

  describe('applyProfile', () => {
    it('should apply profile settings including timer0Range and vcountRange', () => {
      const mockProfile: DeviceProfile = {
        id: 'test-profile',
        name: 'Test Profile',
        romVersion: 'W',
        romRegion: 'USA',
        hardware: 'DS_LITE',
        timer0Auto: false,
        macAddress: [0x00, 0x11, 0x22, 0x33, 0x44, 0x55],
        timer0Range: { min: 0x0C80, max: 0x0C82 },
        vcountRange: { min: 0x61, max: 0x61 },
        tid: 11111,
        sid: 22222,
        frame: 8,
        shinyCharm: false,
        newGame: false,
        withSave: true,
        memoryLink: false,
        createdAt: new Date().toISOString(),
      };

      const { applyProfile } = useEggBootTimingSearchStore.getState();
      applyProfile(mockProfile);

      const state = useEggBootTimingSearchStore.getState();
      expect(state.draftParams.romVersion).toBe('W');
      expect(state.draftParams.romRegion).toBe('USA');
      expect(state.draftParams.hardware).toBe('DS_LITE');
      expect(state.draftParams.macAddress).toEqual([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
      expect(state.draftParams.timer0Range).toEqual({ min: 0x0C80, max: 0x0C82 });
      expect(state.draftParams.vcountRange).toEqual({ min: 0x61, max: 0x61 });
      expect(state.draftParams.conditions.tid).toBe(11111);
      expect(state.draftParams.conditions.sid).toBe(22222);
    });
  });

  describe('validateDraft', () => {
    it('should pass validation for default params', () => {
      const { validateDraft } = useEggBootTimingSearchStore.getState();
      const valid = validateDraft();
      expect(valid).toBe(true);
      expect(useEggBootTimingSearchStore.getState().validationErrors).toHaveLength(0);
      expect(useEggBootTimingSearchStore.getState().params).not.toBeNull();
    });
  });

  describe('updateResultFilters', () => {
    it('should update shinyOnly filter', () => {
      const { updateResultFilters } = useEggBootTimingSearchStore.getState();
      updateResultFilters({ shinyOnly: true });
      expect(useEggBootTimingSearchStore.getState().resultFilters.shinyOnly).toBe(true);
    });

    it('should update natures filter', () => {
      const { updateResultFilters } = useEggBootTimingSearchStore.getState();
      updateResultFilters({ natures: [1, 5, 10] });
      expect(useEggBootTimingSearchStore.getState().resultFilters.natures).toEqual([1, 5, 10]);
    });
  });

  describe('getFilteredResults', () => {
    const mockResult = (shiny: 0 | 1 | 2, nature: number) => ({
      boot: {
        datetime: new Date(),
        timer0: 0x0C79,
        vcount: 0x60,
        keyCode: 0,
        keyInputNames: [],
        macAddress: [0, 0, 0, 0, 0, 0] as const,
      },
      lcgSeedHex: '1234567890ABCDEF',
      egg: {
        advance: 100,
        egg: {
          lcgSeedHex: '1234567890ABCDEF',
          nature,
          shiny,
          ivs: [31, 31, 31, 31, 31, 31] as [number, number, number, number, number, number],
          ability: 0 as 0 | 1 | 2,
          gender: 'male' as 'male' | 'female' | 'genderless',
          hiddenPower: { type: 'known' as const, hpType: 0, power: 70 },
          pid: 0x12345678,
        },
        isStable: true,
      },
      isStable: true,
    });

    beforeEach(() => {
      // Add mock results
      useEggBootTimingSearchStore.setState({
        results: [
          mockResult(0, 0),  // not shiny, hardy
          mockResult(2, 1),  // square shiny, lonely
          mockResult(1, 2),  // star shiny, brave
          mockResult(0, 3),  // not shiny, adamant
        ],
      });
    });

    it('should return all results when no filter is active', () => {
      const { getFilteredResults } = useEggBootTimingSearchStore.getState();
      expect(getFilteredResults()).toHaveLength(4);
    });

    it('should filter shiny only when shinyOnly is true', () => {
      const { updateResultFilters, getFilteredResults } = useEggBootTimingSearchStore.getState();
      updateResultFilters({ shinyOnly: true });
      const filtered = getFilteredResults();
      expect(filtered).toHaveLength(2);
      expect(filtered.every(r => r.egg.egg.shiny !== 0)).toBe(true);
    });

    it('should filter by natures', () => {
      const { updateResultFilters, getFilteredResults } = useEggBootTimingSearchStore.getState();
      updateResultFilters({ natures: [0, 1] });
      const filtered = getFilteredResults();
      expect(filtered).toHaveLength(2);
      expect(filtered.every(r => [0, 1].includes(r.egg.egg.nature))).toBe(true);
    });

    it('should apply both shinyOnly and natures filters', () => {
      const { updateResultFilters, getFilteredResults } = useEggBootTimingSearchStore.getState();
      updateResultFilters({ shinyOnly: true, natures: [1] });
      const filtered = getFilteredResults();
      expect(filtered).toHaveLength(1);
      expect(filtered[0].egg.egg.shiny).toBe(2);
      expect(filtered[0].egg.egg.nature).toBe(1);
    });
  });

  describe('clearResults', () => {
    it('should clear results and pending results', () => {
      useEggBootTimingSearchStore.setState({
        results: [{ boot: {}, lcgSeedHex: 'test', egg: {}, isStable: true } as any],
        _pendingResults: [{ boot: {}, lcgSeedHex: 'test2', egg: {}, isStable: true } as any],
        progress: { processedCombinations: 100, totalCombinations: 1000, foundCount: 1, progressPercent: 10, elapsedMs: 1000 },
        errorMessage: 'test error',
        lastElapsedMs: 5000,
      });

      const { clearResults } = useEggBootTimingSearchStore.getState();
      clearResults();

      const state = useEggBootTimingSearchStore.getState();
      expect(state.results).toHaveLength(0);
      expect(state._pendingResults).toHaveLength(0);
      expect(state.progress).toBeNull();
      expect(state.errorMessage).toBeNull();
      expect(state.lastElapsedMs).toBeNull();
    });
  });

  describe('reset', () => {
    it('should reset all state to defaults', () => {
      useEggBootTimingSearchStore.setState({
        draftParams: { ...createDefaultEggBootTimingSearchParams(), userOffset: 999 },
        validationErrors: ['error'],
        status: 'running',
        results: [{ boot: {}, lcgSeedHex: 'test', egg: {}, isStable: true } as any],
        _pendingResults: [{ boot: {}, lcgSeedHex: 'test2', egg: {}, isStable: true } as any],
        resultFilters: { shinyOnly: true },
        errorMessage: 'test error',
        lastElapsedMs: 5000,
      });

      const { reset } = useEggBootTimingSearchStore.getState();
      reset();

      const state = useEggBootTimingSearchStore.getState();
      expect(state.draftParams.userOffset).toBe(0);  // default value
      expect(state.validationErrors).toHaveLength(0);
      expect(state.status).toBe('idle');
      expect(state.results).toHaveLength(0);
      expect(state._pendingResults).toHaveLength(0);
      expect(state.resultFilters).toEqual({});
      expect(state.errorMessage).toBeNull();
      expect(state.lastElapsedMs).toBeNull();
    });
  });

  describe('internal result handling', () => {
    it('should add pending result', () => {
      const mockResult = {
        boot: {
          datetime: new Date(),
          timer0: 0x0C79,
          vcount: 0x60,
          keyCode: 0,
          keyInputNames: [],
          macAddress: [0, 0, 0, 0, 0, 0] as const,
        },
        lcgSeedHex: '1234567890ABCDEF',
        egg: {
          advance: 100,
          egg: {
            nature: 0,
            shiny: 0,
            ivs: [31, 31, 31, 31, 31, 31] as const,
            ability: 0,
            gender: 0,
            hiddenPower: { type: 0, power: 70 },
            pid: 0x12345678,
          },
        },
        isStable: true,
      };

      const { _addPendingResult } = useEggBootTimingSearchStore.getState();
      const canContinue = _addPendingResult(mockResult as any);

      expect(canContinue).toBe(true);
      expect(useEggBootTimingSearchStore.getState()._pendingResults).toHaveLength(1);
    });

    it('should transfer pending results to results on complete', () => {
      const mockResult = {
        boot: { datetime: new Date(), timer0: 0x0C79, vcount: 0x60, keyCode: 0, keyInputNames: [], macAddress: [0, 0, 0, 0, 0, 0] as const },
        lcgSeedHex: '1234567890ABCDEF',
        egg: { advance: 100, egg: { nature: 0, shiny: 0, ivs: [31, 31, 31, 31, 31, 31] as const, ability: 0, gender: 0, hiddenPower: { type: 0, power: 70 }, pid: 0x12345678 }, isStable: true },
        isStable: true,
      };

      useEggBootTimingSearchStore.setState({
        _pendingResults: [mockResult as any],
        progress: { processedCombinations: 100, totalCombinations: 100, foundCount: 1, progressPercent: 100, elapsedMs: 1000 },
      });

      const { _onComplete } = useEggBootTimingSearchStore.getState();
      _onComplete({ reason: 'completed', processedCombinations: 100, totalCombinations: 100, resultsCount: 1, elapsedMs: 1000 });

      const state = useEggBootTimingSearchStore.getState();
      expect(state.results).toHaveLength(1);
      expect(state._pendingResults).toHaveLength(0);
      expect(state.status).toBe('completed');
    });

    it('should transfer pending results to results on stop', () => {
      const mockResult = {
        boot: { datetime: new Date(), timer0: 0x0C79, vcount: 0x60, keyCode: 0, keyInputNames: [], macAddress: [0, 0, 0, 0, 0, 0] as const },
        lcgSeedHex: '1234567890ABCDEF',
        egg: { advance: 100, egg: { nature: 0, shiny: 0, ivs: [31, 31, 31, 31, 31, 31] as const, ability: 0, gender: 0, hiddenPower: { type: 0, power: 70 }, pid: 0x12345678 }, isStable: true },
        isStable: true,
      };

      useEggBootTimingSearchStore.setState({
        _pendingResults: [mockResult as any],
      });

      const { _onStopped } = useEggBootTimingSearchStore.getState();
      _onStopped();

      const state = useEggBootTimingSearchStore.getState();
      expect(state.results).toHaveLength(1);
      expect(state._pendingResults).toHaveLength(0);
      expect(state.status).toBe('idle');
    });
  });
});
