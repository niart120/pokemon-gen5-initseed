import { describe, it, expect, beforeEach } from 'vitest';
import { useEggStore } from '@/store/egg-store';
import { EggGameMode, createDefaultEggParamsHex } from '@/types/egg';

describe('egg-store', () => {
  beforeEach(() => {
    // Reset store to default state before each test
    useEggStore.getState().reset();
  });

  describe('updateDraftParams', () => {
    it('should update draft params', () => {
      const { updateDraftParams } = useEggStore.getState();
      updateDraftParams({ count: 200 });
      expect(useEggStore.getState().draftParams.count).toBe(200);
    });

    it('should update baseSeedHex', () => {
      const { updateDraftParams } = useEggStore.getState();
      updateDraftParams({ baseSeedHex: 'ABCDEF' });
      expect(useEggStore.getState().draftParams.baseSeedHex).toBe('ABCDEF');
    });

    it('should update gameMode', () => {
      const { updateDraftParams } = useEggStore.getState();
      updateDraftParams({ gameMode: EggGameMode.Bw2Continue });
      expect(useEggStore.getState().draftParams.gameMode).toBe(EggGameMode.Bw2Continue);
    });
  });

  describe('updateDraftConditions', () => {
    it('should update conditions', () => {
      const { updateDraftConditions } = useEggStore.getState();
      updateDraftConditions({ tid: 12345 });
      expect(useEggStore.getState().draftParams.conditions.tid).toBe(12345);
    });

    it('should update everstone', () => {
      const { updateDraftConditions } = useEggStore.getState();
      updateDraftConditions({ everstone: { type: 'fixed', nature: 3 } });
      expect(useEggStore.getState().draftParams.conditions.everstone).toEqual({ type: 'fixed', nature: 3 });
    });

    it('should update multiple conditions', () => {
      const { updateDraftConditions } = useEggStore.getState();
      updateDraftConditions({ usesDitto: true, rerollCount: 5 });
      const state = useEggStore.getState();
      expect(state.draftParams.conditions.usesDitto).toBe(true);
      expect(state.draftParams.conditions.rerollCount).toBe(5);
    });
  });

  describe('updateDraftParentsMale', () => {
    it('should update male parent IVs', () => {
      const { updateDraftParentsMale } = useEggStore.getState();
      updateDraftParentsMale([0, 1, 2, 3, 4, 5]);
      expect(useEggStore.getState().draftParams.parents.male).toEqual([0, 1, 2, 3, 4, 5]);
    });
  });

  describe('updateDraftParentsFemale', () => {
    it('should update female parent IVs', () => {
      const { updateDraftParentsFemale } = useEggStore.getState();
      updateDraftParentsFemale([10, 20, 30, 31, 31, 31]);
      expect(useEggStore.getState().draftParams.parents.female).toEqual([10, 20, 30, 31, 31, 31]);
    });
  });

  describe('validateDraft', () => {
    it('should pass validation for default params', () => {
      const { validateDraft } = useEggStore.getState();
      const valid = validateDraft();
      expect(valid).toBe(true);
      expect(useEggStore.getState().validationErrors).toHaveLength(0);
      expect(useEggStore.getState().params).not.toBeNull();
    });

    it('should fail validation for invalid count', () => {
      const { updateDraftParams, validateDraft } = useEggStore.getState();
      updateDraftParams({ count: -1 });
      const valid = validateDraft();
      expect(valid).toBe(false);
      expect(useEggStore.getState().validationErrors.length).toBeGreaterThan(0);
      expect(useEggStore.getState().params).toBeNull();
    });

    it('should fail validation for invalid rerollCount', () => {
      const { updateDraftConditions, validateDraft } = useEggStore.getState();
      updateDraftConditions({ rerollCount: 10 });
      const valid = validateDraft();
      expect(valid).toBe(false);
      expect(useEggStore.getState().validationErrors.length).toBeGreaterThan(0);
    });

    it('should fail validation for invalid parent IV', () => {
      const { updateDraftParentsMale, validateDraft } = useEggStore.getState();
      updateDraftParentsMale([100, 31, 31, 31, 31, 31]);
      const valid = validateDraft();
      expect(valid).toBe(false);
      expect(useEggStore.getState().validationErrors.length).toBeGreaterThan(0);
    });
  });

  describe('clearResults', () => {
    it('should clear results and completion', () => {
      // Set some state manually
      useEggStore.setState({
        results: [{ advance: 0, egg: {} as any, isStable: false }],
        lastCompletion: { reason: 'max-count', processedCount: 1, filteredCount: 1, elapsedMs: 10 },
        errorMessage: 'test',
      });

      const { clearResults } = useEggStore.getState();
      clearResults();

      const state = useEggStore.getState();
      expect(state.results).toHaveLength(0);
      expect(state.lastCompletion).toBeNull();
      expect(state.errorMessage).toBeNull();
    });
  });

  describe('reset', () => {
    it('should reset all state to defaults', () => {
      // Modify state
      useEggStore.setState({
        draftParams: { ...createDefaultEggParamsHex(), count: 999 },
        validationErrors: ['error'],
        status: 'running',
        results: [{ advance: 0, egg: {} as any, isStable: false }],
        lastCompletion: { reason: 'max-count', processedCount: 1, filteredCount: 1, elapsedMs: 10 },
        errorMessage: 'test',
      });

      const { reset } = useEggStore.getState();
      reset();

      const state = useEggStore.getState();
      expect(state.draftParams.count).toBe(100);
      expect(state.validationErrors).toHaveLength(0);
      expect(state.status).toBe('idle');
      expect(state.results).toHaveLength(0);
      expect(state.lastCompletion).toBeNull();
      expect(state.errorMessage).toBeNull();
    });
  });

  describe('status transitions', () => {
    it('should have idle status initially', () => {
      expect(useEggStore.getState().status).toBe('idle');
    });
  });
});
