import { describe, it, expect } from 'vitest';
import {
  validateEggParams,
  hexParamsToEggParams,
  eggParamsToHex,
  parentIvInputsToIvSet,
  filterIvRangeInputToStatRange,
  createDefaultEggParamsHex,
  createDefaultEggConditions,
  createDefaultParentsIVs,
  createDefaultEggFilter,
  isEggWorkerResponse,
  EggGameMode,
  IV_UNKNOWN,
  type EggGenerationParamsHex,
  type ParentIvInputState,
  type FilterIvRangeInputState,
} from '@/types/egg';

describe('egg types', () => {
  describe('hexParamsToEggParams', () => {
    it('converts hex params correctly', () => {
      const hex: EggGenerationParamsHex = {
        baseSeedHex: 'FFFFFFFFFFFFFFFF',
        userOffsetHex: '0',
        count: 100,
        conditions: createDefaultEggConditions(),
        parents: createDefaultParentsIVs(),
        filter: null,
        considerNpcConsumption: false,
        gameMode: EggGameMode.BwContinue,
      };
      const params = hexParamsToEggParams(hex);
      expect(params.baseSeed).toBe(BigInt('0xFFFFFFFFFFFFFFFF'));
      expect(params.userOffset).toBe(BigInt(0));
      expect(params.count).toBe(100);
      expect(params.gameMode).toBe(EggGameMode.BwContinue);
    });

    it('handles lowercase hex input', () => {
      const hex: EggGenerationParamsHex = {
        baseSeedHex: 'abcdef',
        userOffsetHex: 'ff',
        count: 50,
        conditions: createDefaultEggConditions(),
        parents: createDefaultParentsIVs(),
        filter: null,
        considerNpcConsumption: false,
        gameMode: EggGameMode.BwNew,
      };
      const params = hexParamsToEggParams(hex);
      expect(params.baseSeed).toBe(BigInt('0xabcdef'));
      expect(params.userOffset).toBe(BigInt('0xff'));
    });

    it('handles 0x prefix', () => {
      const hex: EggGenerationParamsHex = {
        baseSeedHex: '0x1234',
        userOffsetHex: '0xAB',
        count: 10,
        conditions: createDefaultEggConditions(),
        parents: createDefaultParentsIVs(),
        filter: null,
        considerNpcConsumption: false,
        gameMode: EggGameMode.BwContinue,
      };
      const params = hexParamsToEggParams(hex);
      expect(params.baseSeed).toBe(BigInt('0x1234'));
      expect(params.userOffset).toBe(BigInt('0xAB'));
    });
  });

  describe('eggParamsToHex', () => {
    it('converts params to hex correctly', () => {
      const params = hexParamsToEggParams(createDefaultEggParamsHex());
      const hex = eggParamsToHex(params);
      expect(hex.baseSeedHex).toBe('0');
      expect(hex.userOffsetHex).toBe('0');
      expect(hex.count).toBe(100);
    });
  });

  describe('validateEggParams', () => {
    it('passes valid params', () => {
      const hex = createDefaultEggParamsHex();
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors).toHaveLength(0);
    });

    it('detects invalid count (too high)', () => {
      const hex = createDefaultEggParamsHex();
      hex.count = 100001;
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0]).toContain('count');
    });

    it('detects invalid count (too low)', () => {
      const hex = createDefaultEggParamsHex();
      hex.count = 0;
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0]).toContain('count');
    });

    it('accepts valid femaleParentAbility values', () => {
      const hex = createDefaultEggParamsHex();
      hex.conditions.femaleParentAbility = 2;
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBe(0);
    });

    it('detects invalid TID', () => {
      const hex = createDefaultEggParamsHex();
      hex.conditions.tid = 70000;
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors.some(e => e.includes('tid'))).toBe(true);
    });

    it('detects invalid SID', () => {
      const hex = createDefaultEggParamsHex();
      hex.conditions.sid = -1;
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors.some(e => e.includes('sid'))).toBe(true);
    });

    it('detects invalid parent IV', () => {
      const hex = createDefaultEggParamsHex();
      hex.parents.male[0] = 33;
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors.some(e => e.includes('parents.male[0]'))).toBe(true);
    });

    it('detects invalid filter IV range', () => {
      const hex = createDefaultEggParamsHex();
      hex.filter = createDefaultEggFilter();
      hex.filter.ivRanges[0] = { min: 20, max: 10 }; // min > max
      const params = hexParamsToEggParams(hex);
      const errors = validateEggParams(params);
      expect(errors.length).toBeGreaterThan(0);
      expect(errors.some(e => e.includes('ivRanges[0]'))).toBe(true);
    });
  });

  describe('parentIvInputsToIvSet', () => {
    it('converts normal values', () => {
      const inputs: ParentIvInputState[] = [
        { value: 31, isUnknown: false },
        { value: 25, isUnknown: false },
        { value: 0, isUnknown: false },
        { value: 15, isUnknown: false },
        { value: 31, isUnknown: false },
        { value: 10, isUnknown: false },
      ];
      const ivSet = parentIvInputsToIvSet(inputs);
      expect(ivSet).toEqual([31, 25, 0, 15, 31, 10]);
    });

    it('converts unknown values to 32', () => {
      const inputs: ParentIvInputState[] = [
        { value: 31, isUnknown: true },
        { value: 25, isUnknown: false },
        { value: 0, isUnknown: true },
        { value: 15, isUnknown: false },
        { value: 31, isUnknown: false },
        { value: 10, isUnknown: true },
      ];
      const ivSet = parentIvInputsToIvSet(inputs);
      expect(ivSet).toEqual([IV_UNKNOWN, 25, IV_UNKNOWN, 15, 31, IV_UNKNOWN]);
    });

    it('throws on invalid length', () => {
      const inputs: ParentIvInputState[] = [
        { value: 31, isUnknown: false },
      ];
      expect(() => parentIvInputsToIvSet(inputs)).toThrow();
    });
  });

  describe('filterIvRangeInputToStatRange', () => {
    it('converts normal range', () => {
      const input: FilterIvRangeInputState = {
        min: 5,
        max: 25,
        includeUnknown: false,
      };
      const range = filterIvRangeInputToStatRange(input);
      expect(range).toEqual({ min: 5, max: 25 });
    });

    it('includes unknown when flag is set', () => {
      const input: FilterIvRangeInputState = {
        min: 5,
        max: 25,
        includeUnknown: true,
      };
      const range = filterIvRangeInputToStatRange(input);
      expect(range).toEqual({ min: 5, max: IV_UNKNOWN });
    });
  });

  describe('isEggWorkerResponse', () => {
    it('returns true for valid READY response', () => {
      expect(isEggWorkerResponse({ type: 'READY', version: '1' })).toBe(true);
    });

    it('returns true for valid RESULTS response', () => {
      expect(isEggWorkerResponse({ type: 'RESULTS', payload: { results: [] } })).toBe(true);
    });

    it('returns true for valid COMPLETE response', () => {
      expect(isEggWorkerResponse({
        type: 'COMPLETE',
        payload: { reason: 'max-count', processedCount: 100, filteredCount: 10, elapsedMs: 50 }
      })).toBe(true);
    });

    it('returns true for valid ERROR response', () => {
      expect(isEggWorkerResponse({
        type: 'ERROR',
        message: 'test',
        category: 'RUNTIME',
        fatal: false
      })).toBe(true);
    });

    it('returns false for invalid type', () => {
      expect(isEggWorkerResponse({ type: 'INVALID' })).toBe(false);
    });

    it('returns false for null', () => {
      expect(isEggWorkerResponse(null)).toBe(false);
    });

    it('returns false for undefined', () => {
      expect(isEggWorkerResponse(undefined)).toBe(false);
    });

    it('returns false for non-object', () => {
      expect(isEggWorkerResponse('string')).toBe(false);
    });
  });

  describe('default creators', () => {
    it('createDefaultEggConditions returns valid conditions', () => {
      const conditions = createDefaultEggConditions();
      expect(conditions.hasNidoranFlag).toBe(false);
      expect(conditions.everstone).toEqual({ type: 'none' });
      expect(conditions.usesDitto).toBe(false);
      expect(conditions.masudaMethod).toBe(false);
      expect(conditions.femaleParentAbility).toBe(0);
      expect(conditions.tid).toBe(0);
      expect(conditions.sid).toBe(0);
      expect(conditions.genderRatio.threshold).toBe(127);
    });

    it('createDefaultParentsIVs returns 6x31 for both parents', () => {
      const parents = createDefaultParentsIVs();
      expect(parents.male).toEqual([31, 31, 31, 31, 31, 31]);
      expect(parents.female).toEqual([31, 31, 31, 31, 31, 31]);
    });

    it('createDefaultEggFilter returns 0-32 (任意) ranges', () => {
      const filter = createDefaultEggFilter();
      expect(filter.ivRanges.length).toBe(6);
      filter.ivRanges.forEach(range => {
        expect(range.min).toBe(0);
        expect(range.max).toBe(32);
      });
    });
  });
});
