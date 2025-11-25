/**
 * Boot-Timing Runner State Management for Egg Generation
 * 
 * Manages multi-job execution state for boot-timing enumeration mode.
 * Based on: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md §10.5
 */

import type { EggCompletion } from '@/types/egg';
import type {
  DerivedEggSeedJob,
  DerivedEggSeedAggregate,
  DerivedEggSeedRunState,
} from '@/types/egg';

// === State Factory ===

export function createDerivedEggSeedState(jobs: DerivedEggSeedJob[]): DerivedEggSeedRunState {
  return {
    jobs,
    cursor: 0,
    total: jobs.length,
    aggregate: {
      processedCount: 0,
      filteredCount: 0,
      elapsedMs: 0,
    },
    abortRequested: false,
  };
}

// === State Queries ===

export function shouldAppendDerivedEggResults(state: DerivedEggSeedRunState | null): boolean {
  return Boolean(state && state.cursor > 0);
}

export function currentDerivedEggSeedJob(state: DerivedEggSeedRunState | null): DerivedEggSeedJob | null {
  if (!state) return null;
  return state.jobs[state.cursor] ?? null;
}

export function markDerivedEggSeedAbort(state: DerivedEggSeedRunState | null): DerivedEggSeedRunState | null {
  if (!state) return null;
  if (state.abortRequested) return state;
  return { ...state, abortRequested: true };
}

// === State Advancement ===

export interface DerivedEggSeedAdvanceResult {
  nextState: DerivedEggSeedRunState | null;
  nextJob: DerivedEggSeedJob | null;
  finalCompletion: EggCompletion | null;
  aggregate: DerivedEggSeedAggregate;
}

export function advanceDerivedEggSeedState(
  state: DerivedEggSeedRunState,
  completion: EggCompletion,
): DerivedEggSeedAdvanceResult {
  const aggregate: DerivedEggSeedAggregate = {
    processedCount: state.aggregate.processedCount + completion.processedCount,
    filteredCount: state.aggregate.filteredCount + completion.filteredCount,
    elapsedMs: state.aggregate.elapsedMs + completion.elapsedMs,
  };

  const nextCursor = state.cursor + 1;
  const hasMore = nextCursor < state.total;

  if (!hasMore || state.abortRequested) {
    const finalCompletion: EggCompletion = {
      ...completion,
      processedCount: aggregate.processedCount,
      filteredCount: aggregate.filteredCount,
      elapsedMs: aggregate.elapsedMs,
    };
    return {
      nextState: null,
      nextJob: null,
      finalCompletion,
      aggregate,
    };
  }

  return {
    nextState: {
      ...state,
      cursor: nextCursor,
      aggregate,
    },
    nextJob: state.jobs[nextCursor],
    finalCompletion: null,
    aggregate,
  };
}

// === Progress Calculation ===

export interface DerivedEggSeedProgress {
  currentJob: number;
  totalJobs: number;
  progressPercent: number;
  isComplete: boolean;
}

export function getDerivedEggSeedProgress(state: DerivedEggSeedRunState | null): DerivedEggSeedProgress | null {
  if (!state) return null;
  const isComplete = state.cursor >= state.total;
  return {
    currentJob: state.cursor + 1,
    totalJobs: state.total,
    progressPercent: state.total > 0 ? (state.cursor / state.total) * 100 : 0,
    isComplete,
  };
}
