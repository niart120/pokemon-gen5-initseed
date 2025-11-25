/**
 * Tests for egg-boot-timing-runner module
 */

import { describe, it, expect } from 'vitest';
import {
  createDerivedEggSeedState,
  shouldAppendDerivedEggResults,
  currentDerivedEggSeedJob,
  markDerivedEggSeedAbort,
  advanceDerivedEggSeedState,
  getDerivedEggSeedProgress,
} from '@/store/modules/egg-boot-timing-runner';
import type { DerivedEggSeedJob, EggCompletion } from '@/types/egg';

const createMockJob = (index: number): DerivedEggSeedJob => ({
  params: {
    baseSeed: BigInt(0x12345678 + index),
    userOffset: BigInt(0),
    count: 100,
    conditions: {
      hasNidoranFlag: false,
      everstone: { type: 'none' },
      usesDitto: false,
      femaleParentAbility: 0,
      masudaMethod: false,
      tid: 0,
      sid: 0,
      genderRatio: { threshold: 127, genderless: false },
    },
    parents: {
      male: [31, 31, 31, 31, 31, 31],
      female: [31, 31, 31, 31, 31, 31],
    },
    filter: null,
    considerNpcConsumption: false,
    gameMode: 0,
  },
  metadata: {
    seedSourceMode: 'boot-timing',
    derivedSeedIndex: index,
    timer0: 0x1000 + index,
    vcount: 0x50 + index,
    keyMask: 0,
    keyCode: 0,
    bootTimestampIso: '2025-01-15T12:00:00.000Z',
    macAddress: [0x00, 0x09, 0xBF, 0x12, 0x34, 0x56],
    seedSourceSeedHex: (0x12345678 + index).toString(16).toUpperCase(),
  },
});

const createMockCompletion = (processedCount: number = 100): EggCompletion => ({
  reason: 'max-count',
  processedCount,
  filteredCount: processedCount,
  elapsedMs: 50,
});

describe('egg-boot-timing-runner', () => {
  describe('createDerivedEggSeedState', () => {
    it('should create state with correct initial values', () => {
      const jobs = [createMockJob(0), createMockJob(1), createMockJob(2)];
      const state = createDerivedEggSeedState(jobs);

      expect(state.jobs).toHaveLength(3);
      expect(state.cursor).toBe(0);
      expect(state.total).toBe(3);
      expect(state.aggregate.processedCount).toBe(0);
      expect(state.aggregate.filteredCount).toBe(0);
      expect(state.aggregate.elapsedMs).toBe(0);
      expect(state.abortRequested).toBe(false);
    });

    it('should handle empty jobs array', () => {
      const state = createDerivedEggSeedState([]);

      expect(state.jobs).toHaveLength(0);
      expect(state.cursor).toBe(0);
      expect(state.total).toBe(0);
    });
  });

  describe('shouldAppendDerivedEggResults', () => {
    it('should return false for null state', () => {
      expect(shouldAppendDerivedEggResults(null)).toBe(false);
    });

    it('should return false when cursor is 0', () => {
      const state = createDerivedEggSeedState([createMockJob(0)]);
      expect(shouldAppendDerivedEggResults(state)).toBe(false);
    });

    it('should return true when cursor > 0', () => {
      const state = createDerivedEggSeedState([createMockJob(0), createMockJob(1)]);
      const advanced = advanceDerivedEggSeedState(state, createMockCompletion());
      expect(shouldAppendDerivedEggResults(advanced.nextState)).toBe(true);
    });
  });

  describe('currentDerivedEggSeedJob', () => {
    it('should return null for null state', () => {
      expect(currentDerivedEggSeedJob(null)).toBeNull();
    });

    it('should return current job', () => {
      const jobs = [createMockJob(0), createMockJob(1)];
      const state = createDerivedEggSeedState(jobs);
      
      expect(currentDerivedEggSeedJob(state)).toBe(jobs[0]);
    });

    it('should return null when cursor exceeds jobs', () => {
      const state = createDerivedEggSeedState([createMockJob(0)]);
      const advanced = advanceDerivedEggSeedState(state, createMockCompletion());
      
      expect(currentDerivedEggSeedJob(advanced.nextState)).toBeNull();
    });
  });

  describe('markDerivedEggSeedAbort', () => {
    it('should return null for null state', () => {
      expect(markDerivedEggSeedAbort(null)).toBeNull();
    });

    it('should set abortRequested to true', () => {
      const state = createDerivedEggSeedState([createMockJob(0)]);
      const aborted = markDerivedEggSeedAbort(state);

      expect(aborted?.abortRequested).toBe(true);
    });

    it('should return same state if already aborted', () => {
      const state = createDerivedEggSeedState([createMockJob(0)]);
      const aborted1 = markDerivedEggSeedAbort(state);
      const aborted2 = markDerivedEggSeedAbort(aborted1);

      expect(aborted1).toBe(aborted2);
    });
  });

  describe('advanceDerivedEggSeedState', () => {
    it('should accumulate aggregate values', () => {
      const state = createDerivedEggSeedState([createMockJob(0), createMockJob(1), createMockJob(2)]);
      const completion: EggCompletion = {
        reason: 'max-count',
        processedCount: 100,
        filteredCount: 50,
        elapsedMs: 25,
      };

      const result = advanceDerivedEggSeedState(state, completion);

      expect(result.aggregate.processedCount).toBe(100);
      expect(result.aggregate.filteredCount).toBe(50);
      expect(result.aggregate.elapsedMs).toBe(25);
    });

    it('should advance cursor and return next job', () => {
      const jobs = [createMockJob(0), createMockJob(1), createMockJob(2)];
      const state = createDerivedEggSeedState(jobs);

      const result = advanceDerivedEggSeedState(state, createMockCompletion());

      expect(result.nextState?.cursor).toBe(1);
      expect(result.nextJob).toBe(jobs[1]);
      expect(result.finalCompletion).toBeNull();
    });

    it('should return finalCompletion when all jobs complete', () => {
      const state = createDerivedEggSeedState([createMockJob(0)]);
      const completion = createMockCompletion();

      const result = advanceDerivedEggSeedState(state, completion);

      expect(result.nextState).toBeNull();
      expect(result.nextJob).toBeNull();
      expect(result.finalCompletion).not.toBeNull();
      expect(result.finalCompletion?.processedCount).toBe(100);
    });

    it('should stop early when abortRequested is true', () => {
      const state = createDerivedEggSeedState([createMockJob(0), createMockJob(1)]);
      const abortedState = markDerivedEggSeedAbort(state)!;

      const result = advanceDerivedEggSeedState(abortedState, createMockCompletion());

      expect(result.nextState).toBeNull();
      expect(result.finalCompletion).not.toBeNull();
    });
  });

  describe('getDerivedEggSeedProgress', () => {
    it('should return null for null state', () => {
      expect(getDerivedEggSeedProgress(null)).toBeNull();
    });

    it('should calculate progress correctly', () => {
      const state = createDerivedEggSeedState([createMockJob(0), createMockJob(1), createMockJob(2), createMockJob(3)]);
      
      // Initial state
      let progress = getDerivedEggSeedProgress(state);
      expect(progress?.currentJob).toBe(1);
      expect(progress?.totalJobs).toBe(4);
      expect(progress?.progressPercent).toBe(0);
      expect(progress?.isComplete).toBe(false);

      // After first job
      const advanced = advanceDerivedEggSeedState(state, createMockCompletion());
      progress = getDerivedEggSeedProgress(advanced.nextState);
      expect(progress?.currentJob).toBe(2);
      expect(progress?.progressPercent).toBe(25);
    });

    it('should handle empty jobs', () => {
      const state = createDerivedEggSeedState([]);
      const progress = getDerivedEggSeedProgress(state);

      expect(progress?.totalJobs).toBe(0);
      expect(progress?.progressPercent).toBe(0);
      expect(progress?.isComplete).toBe(true);
    });
  });
});
