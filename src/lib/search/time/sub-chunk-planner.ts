import { advanceByAllowedSeconds, type SearchTimePlan } from './time-plan';

const MS_PER_SECOND = 1000;

export interface SubChunkWindow {
  startTimestampMs: number;
  durationSeconds: number;
  countedSeconds: number;
  lastAllowedTimestampMs?: number;
}

export interface IterateAllowedSubChunksOptions {
  plan: SearchTimePlan;
  chunkStartMs: number;
  chunkEndExclusiveMs: number;
  desiredAllowedSeconds: number;
  maxSecondsPerChunk?: number;
}

export function* iterateAllowedSubChunks(
  options: IterateAllowedSubChunksOptions
): Generator<SubChunkWindow> {
  const {
    plan,
    chunkStartMs,
    chunkEndExclusiveMs,
    desiredAllowedSeconds,
    maxSecondsPerChunk,
  } = options;

  if (chunkEndExclusiveMs <= chunkStartMs) {
    return;
  }

  const desiredSeconds = Math.max(1, Math.floor(desiredAllowedSeconds));
  const maxTimelineSeconds = resolveMaxTimelineSeconds(maxSecondsPerChunk);
  const perChunkTarget = desiredSeconds;

  let cursor = chunkStartMs;

  while (cursor < chunkEndExclusiveMs) {
    const timelineLimitMs = Math.min(
      chunkEndExclusiveMs,
      cursor + maxTimelineSeconds * MS_PER_SECOND
    );

    const {
      endTimestampMs,
      countedSeconds,
      firstAllowedTimestampMs,
      lastAllowedTimestampMs,
    } = advanceByAllowedSeconds(plan, cursor, timelineLimitMs, perChunkTarget);

    if (countedSeconds <= 0 || firstAllowedTimestampMs === undefined) {
      if (endTimestampMs <= cursor) {
        break;
      }
      cursor = endTimestampMs;
      continue;
    }

    const timelineMs = Math.max(0, endTimestampMs - firstAllowedTimestampMs);
    const durationSeconds = Math.max(
      countedSeconds,
      Math.ceil(timelineMs / MS_PER_SECOND)
    );

    yield {
      startTimestampMs: firstAllowedTimestampMs,
      durationSeconds,
      countedSeconds,
      lastAllowedTimestampMs,
    };

    if (endTimestampMs <= cursor) {
      break;
    }

    cursor = endTimestampMs;
  }
}

function resolveMaxTimelineSeconds(maxSecondsPerChunk?: number): number {
  if (maxSecondsPerChunk === undefined || !Number.isFinite(maxSecondsPerChunk)) {
    return Number.MAX_SAFE_INTEGER;
  }
  return Math.max(1, Math.floor(maxSecondsPerChunk));
}
