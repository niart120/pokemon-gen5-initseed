/**
 * Tests for egg-result-filter module
 */

import { describe, it, expect } from 'vitest';
import {
  applyBootTimingFilters,
  isValidHexFilter,
  formatBootTimingFilterStatus,
  type EggBootTimingFilters,
} from '@/lib/egg/egg-result-filter';
import type { EnumeratedEggDataWithBootTiming, ResolvedEgg } from '@/types/egg';

const createMockEgg = (): ResolvedEgg => ({
  ivs: [31, 31, 31, 31, 31, 31],
  nature: 0,
  gender: 'male',
  ability: 0,
  shiny: 0,
  pid: 0x12345678,
  hiddenPower: { type: 'unknown' },
});

const createMockResult = (overrides?: Partial<EnumeratedEggDataWithBootTiming>): EnumeratedEggDataWithBootTiming => ({
  advance: 100,
  egg: createMockEgg(),
  isStable: true,
  seedSourceMode: 'boot-timing',
  timer0: 0x10A0,
  vcount: 0x5C,
  ...overrides,
});

describe('egg-result-filter', () => {
  describe('applyBootTimingFilters', () => {
    it('should return all results when seedSourceMode is lcg', () => {
      const results = [
        createMockResult({ timer0: 0x1000, vcount: 0x50 }),
        createMockResult({ timer0: 0x2000, vcount: 0x60 }),
      ];
      const filters: EggBootTimingFilters = {
        timer0Filter: '1000',
        vcountFilter: '50',
      };

      const filtered = applyBootTimingFilters(results, filters, 'lcg');
      expect(filtered).toHaveLength(2); // No filtering in LCG mode
    });

    it('should return all results when no filters are set', () => {
      const results = [
        createMockResult({ timer0: 0x1000, vcount: 0x50 }),
        createMockResult({ timer0: 0x2000, vcount: 0x60 }),
      ];
      const filters: EggBootTimingFilters = {};

      const filtered = applyBootTimingFilters(results, filters, 'boot-timing');
      expect(filtered).toHaveLength(2);
    });

    it('should filter by timer0', () => {
      const results = [
        createMockResult({ timer0: 0x10A0, vcount: 0x50 }),
        createMockResult({ timer0: 0x10A1, vcount: 0x50 }),
        createMockResult({ timer0: 0x10A2, vcount: 0x50 }),
      ];
      const filters: EggBootTimingFilters = {
        timer0Filter: '10A0',
      };

      const filtered = applyBootTimingFilters(results, filters, 'boot-timing');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].timer0).toBe(0x10A0);
    });

    it('should filter by vcount', () => {
      const results = [
        createMockResult({ timer0: 0x1000, vcount: 0x5C }),
        createMockResult({ timer0: 0x1000, vcount: 0x5D }),
        createMockResult({ timer0: 0x1000, vcount: 0x5E }),
      ];
      const filters: EggBootTimingFilters = {
        vcountFilter: '5C',
      };

      const filtered = applyBootTimingFilters(results, filters, 'boot-timing');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].vcount).toBe(0x5C);
    });

    it('should filter by both timer0 and vcount', () => {
      const results = [
        createMockResult({ timer0: 0x10A0, vcount: 0x5C }),
        createMockResult({ timer0: 0x10A0, vcount: 0x5D }),
        createMockResult({ timer0: 0x10A1, vcount: 0x5C }),
        createMockResult({ timer0: 0x10A1, vcount: 0x5D }),
      ];
      const filters: EggBootTimingFilters = {
        timer0Filter: '10A0',
        vcountFilter: '5C',
      };

      const filtered = applyBootTimingFilters(results, filters, 'boot-timing');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].timer0).toBe(0x10A0);
      expect(filtered[0].vcount).toBe(0x5C);
    });

    it('should handle results without timer0/vcount', () => {
      const results = [
        createMockResult({ timer0: undefined, vcount: undefined }),
        createMockResult({ timer0: 0x10A0, vcount: 0x5C }),
      ];
      const filters: EggBootTimingFilters = {
        timer0Filter: '10A0',
      };

      const filtered = applyBootTimingFilters(results, filters, 'boot-timing');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].timer0).toBe(0x10A0);
    });

    it('should match all when filter is invalid hex', () => {
      const results = [
        createMockResult({ timer0: 0x1000, vcount: 0x50 }),
        createMockResult({ timer0: 0x2000, vcount: 0x60 }),
      ];
      const filters: EggBootTimingFilters = {
        timer0Filter: 'XYZ', // Invalid hex
      };

      const filtered = applyBootTimingFilters(results, filters, 'boot-timing');
      expect(filtered).toHaveLength(2); // All match when filter is invalid
    });

    it('should be case-insensitive for hex filter', () => {
      const results = [
        createMockResult({ timer0: 0x10A0, vcount: 0x5C }),
      ];

      const filterLower: EggBootTimingFilters = { timer0Filter: '10a0' };
      const filterUpper: EggBootTimingFilters = { timer0Filter: '10A0' };

      const filteredLower = applyBootTimingFilters(results, filterLower, 'boot-timing');
      const filteredUpper = applyBootTimingFilters(results, filterUpper, 'boot-timing');

      expect(filteredLower).toHaveLength(1);
      expect(filteredUpper).toHaveLength(1);
    });
  });

  describe('isValidHexFilter', () => {
    it('should return true for empty string', () => {
      expect(isValidHexFilter('')).toBe(true);
    });

    it('should return true for valid hex string', () => {
      expect(isValidHexFilter('10A0')).toBe(true);
      expect(isValidHexFilter('5c')).toBe(true);
      expect(isValidHexFilter('0123456789abcdefABCDEF')).toBe(true);
    });

    it('should return false for invalid hex string', () => {
      expect(isValidHexFilter('XYZ')).toBe(false);
      expect(isValidHexFilter('10G0')).toBe(false);
      expect(isValidHexFilter('hello')).toBe(false);
    });

    it('should handle whitespace', () => {
      expect(isValidHexFilter('  ')).toBe(true); // Empty after trim
      expect(isValidHexFilter(' 10A0 ')).toBe(true);
    });
  });

  describe('formatBootTimingFilterStatus', () => {
    it('should return empty string when no filters set', () => {
      expect(formatBootTimingFilterStatus({})).toBe('');
      expect(formatBootTimingFilterStatus({ timer0Filter: '', vcountFilter: '' })).toBe('');
    });

    it('should format timer0 filter only', () => {
      expect(formatBootTimingFilterStatus({ timer0Filter: '10A0' })).toBe('Timer0: 0x10A0');
    });

    it('should format vcount filter only', () => {
      expect(formatBootTimingFilterStatus({ vcountFilter: '5c' })).toBe('VCount: 0x5C');
    });

    it('should format both filters', () => {
      expect(formatBootTimingFilterStatus({ timer0Filter: '10A0', vcountFilter: '5c' }))
        .toBe('Timer0: 0x10A0, VCount: 0x5C');
    });
  });
});
