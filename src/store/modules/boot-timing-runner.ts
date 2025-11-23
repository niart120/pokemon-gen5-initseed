import type { GenerationCompletion } from '@/types/generation';
import type { DerivedSeedJob } from '@/lib/generation/boot-timing-derivation';

export interface DerivedSeedAggregate {
  processedAdvances: number;
  resultsCount: number;
  elapsedMs: number;
  shinyFound: boolean;
}

export interface DerivedSeedRunState {
  readonly jobs: DerivedSeedJob[];
  readonly cursor: number;
  readonly total: number;
  readonly aggregate: DerivedSeedAggregate;
  readonly abortRequested: boolean;
}

export function createDerivedSeedState(jobs: DerivedSeedJob[]): DerivedSeedRunState {
  return {
    jobs,
    cursor: 0,
    total: jobs.length,
    aggregate: {
      processedAdvances: 0,
      resultsCount: 0,
      elapsedMs: 0,
      shinyFound: false,
    },
    abortRequested: false,
  };
}

export function shouldAppendDerivedResults(state: DerivedSeedRunState | null): boolean {
  return Boolean(state && state.cursor > 0);
}

export function currentDerivedSeedJob(state: DerivedSeedRunState | null): DerivedSeedJob | null {
  if (!state) return null;
  return state.jobs[state.cursor] ?? null;
}

export function markDerivedSeedAbort(state: DerivedSeedRunState | null): DerivedSeedRunState | null {
  if (!state) return null;
  if (state.abortRequested) return state;
  return { ...state, abortRequested: true };
}

export interface DerivedSeedAdvanceResult {
  nextState: DerivedSeedRunState | null;
  nextJob: DerivedSeedJob | null;
  finalCompletion: GenerationCompletion | null;
  aggregate: DerivedSeedAggregate;
}

export function advanceDerivedSeedState(
  state: DerivedSeedRunState,
  completion: GenerationCompletion,
): DerivedSeedAdvanceResult {
  const aggregate: DerivedSeedAggregate = {
    processedAdvances: state.aggregate.processedAdvances + completion.processedAdvances,
    resultsCount: state.aggregate.resultsCount + completion.resultsCount,
    elapsedMs: state.aggregate.elapsedMs + completion.elapsedMs,
    shinyFound: state.aggregate.shinyFound || completion.shinyFound,
  };
  const nextCursor = state.cursor + 1;
  const hasMore = nextCursor < state.total;
  if (!hasMore) {
    const finalCompletion: GenerationCompletion = {
      ...completion,
      processedAdvances: aggregate.processedAdvances,
      resultsCount: aggregate.resultsCount,
      elapsedMs: aggregate.elapsedMs,
      shinyFound: aggregate.shinyFound,
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
