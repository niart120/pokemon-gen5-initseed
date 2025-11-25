/**
 * Tests for boot-timing-egg-derivation module
 */

import { describe, it, expect } from 'vitest';
import {
  deriveBootTimingEggSeedJobs,
  validateEggBootTimingInputs,
  EGG_BOOT_TIMING_PAIR_LIMIT,
} from '@/lib/egg/boot-timing-egg-derivation';
import {
  createDefaultEggParamsHex,
  createDefaultEggBootTimingDraft,
} from '@/types/egg';

describe('boot-timing-egg-derivation', () => {
  describe('validateEggBootTimingInputs', () => {
    it('should return error when timestamp is missing', () => {
      const draft = createDefaultEggBootTimingDraft();
      const errors = validateEggBootTimingInputs(draft);
      expect(errors).toContain('boot-timing timestamp required');
    });

    it('should return error when timestamp is invalid', () => {
      const draft = createDefaultEggBootTimingDraft();
      draft.timestampIso = 'invalid-date';
      const errors = validateEggBootTimingInputs(draft);
      expect(errors).toContain('boot-timing timestamp invalid');
    });

    it('should return error when timer0 range is out of bounds', () => {
      const draft = createDefaultEggBootTimingDraft();
      draft.timestampIso = new Date().toISOString();
      draft.timer0Range = { min: -1, max: 0xFFFF };
      const errors = validateEggBootTimingInputs(draft);
      expect(errors).toContain('timer0 range out of bounds');
    });

    it('should return error when timer0 range is invalid (min > max)', () => {
      const draft = createDefaultEggBootTimingDraft();
      draft.timestampIso = new Date().toISOString();
      draft.timer0Range = { min: 100, max: 50 };
      draft.vcountRange = { min: 0, max: 10 };
      const errors = validateEggBootTimingInputs(draft);
      expect(errors).toContain('timer0 range invalid');
    });

    it('should return error when vcount range is out of bounds', () => {
      const draft = createDefaultEggBootTimingDraft();
      draft.timestampIso = new Date().toISOString();
      draft.timer0Range = { min: 0, max: 10 };
      draft.vcountRange = { min: 0, max: 0x100 }; // exceeds 0xFF
      const errors = validateEggBootTimingInputs(draft);
      expect(errors).toContain('vcount range out of bounds');
    });

    it('should return error when combination count exceeds limit', () => {
      const draft = createDefaultEggBootTimingDraft();
      draft.timestampIso = new Date().toISOString();
      draft.timer0Range = { min: 0, max: 100 }; // 101 values
      draft.vcountRange = { min: 0, max: 10 };  // 11 values = 1111 total > 512
      const errors = validateEggBootTimingInputs(draft);
      expect(errors.some(e => e.includes('exceed limit'))).toBe(true);
    });

    it('should return empty array for valid inputs', () => {
      const draft = createDefaultEggBootTimingDraft();
      draft.timestampIso = new Date().toISOString();
      draft.timer0Range = { min: 0x1000, max: 0x1010 }; // 17 values
      draft.vcountRange = { min: 0x50, max: 0x60 };      // 17 values = 289 total
      const errors = validateEggBootTimingInputs(draft);
      expect(errors).toHaveLength(0);
    });
  });

  describe('deriveBootTimingEggSeedJobs', () => {
    it('should return error when timestamp is missing', () => {
      const params = createDefaultEggParamsHex();
      params.seedSourceMode = 'boot-timing';
      // bootTiming.timestampIso is undefined by default

      const result = deriveBootTimingEggSeedJobs(params);
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('timestamp');
      }
    });

    it('should derive jobs for valid inputs', () => {
      const params = createDefaultEggParamsHex();
      params.seedSourceMode = 'boot-timing';
      params.bootTiming = {
        ...createDefaultEggBootTimingDraft(),
        timestampIso: '2025-01-15T12:00:00.000Z',
        timer0Range: { min: 0x1000, max: 0x1002 }, // 3 values
        vcountRange: { min: 0x50, max: 0x51 },      // 2 values = 6 jobs
        romVersion: 'B',
        romRegion: 'JPN',
        hardware: 'DS',
        macAddress: [0x00, 0x09, 0xBF, 0x12, 0x34, 0x56],
      };

      const result = deriveBootTimingEggSeedJobs(params);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.jobs.length).toBe(6); // 3 * 2 = 6
        
        // Check first job metadata
        const firstJob = result.jobs[0];
        expect(firstJob.metadata.seedSourceMode).toBe('boot-timing');
        expect(firstJob.metadata.derivedSeedIndex).toBe(0);
        expect(firstJob.metadata.timer0).toBe(0x1000);
        expect(firstJob.metadata.vcount).toBe(0x50);
        expect(firstJob.metadata.bootTimestampIso).toBe('2025-01-15T12:00:00.000Z');
        
        // Check last job metadata
        const lastJob = result.jobs[5];
        expect(lastJob.metadata.derivedSeedIndex).toBe(5);
        expect(lastJob.metadata.timer0).toBe(0x1002);
        expect(lastJob.metadata.vcount).toBe(0x51);
      }
    });

    it('should respect maxPairs option', () => {
      const params = createDefaultEggParamsHex();
      params.seedSourceMode = 'boot-timing';
      params.bootTiming = {
        ...createDefaultEggBootTimingDraft(),
        timestampIso: '2025-01-15T12:00:00.000Z',
        timer0Range: { min: 0x1000, max: 0x1009 }, // 10 values
        vcountRange: { min: 0x50, max: 0x59 },      // 10 values = 100 jobs
        romVersion: 'B',
        romRegion: 'JPN',
        hardware: 'DS',
        macAddress: [0x00, 0x09, 0xBF, 0x12, 0x34, 0x56],
      };

      // With default limit (512), should succeed
      const result1 = deriveBootTimingEggSeedJobs(params);
      expect(result1.ok).toBe(true);
      if (result1.ok) {
        expect(result1.jobs.length).toBe(100);
      }

      // With custom limit (50), should fail
      const result2 = deriveBootTimingEggSeedJobs(params, { maxPairs: 50 });
      expect(result2.ok).toBe(false);
    });
  });

  describe('EGG_BOOT_TIMING_PAIR_LIMIT', () => {
    it('should be 512', () => {
      expect(EGG_BOOT_TIMING_PAIR_LIMIT).toBe(512);
    });
  });
});
