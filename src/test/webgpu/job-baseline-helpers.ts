import { SeedCalculator } from '@/lib/core/seed-calculator';
import { getDateFromTimePlan } from '@/lib/search/time/time-plan';
import type { SeedSearchJob } from '@/lib/webgpu/seed-search/types';
import type { SearchConditions } from '@/types/search';

export interface CpuBaselineEntry {
  seed: number;
  timer0: number;
  vcount: number;
  datetime: Date;
  keyCode: number;
}

const calculator = new SeedCalculator();

export function enumerateJobCpuBaseline(
  conditions: SearchConditions,
  job: SeedSearchJob,
  limit?: number
): CpuBaselineEntry[] {
  const entries: CpuBaselineEntry[] = [];

  for (const segment of job.segments) {
    const messagesPerVcount = Math.max(1, segment.rangeSeconds);
    const messagesPerTimer0 = messagesPerVcount * Math.max(1, segment.vcountCount);
    const baseOffset =
      segment.baseTimer0Index * messagesPerTimer0 +
      segment.baseVcountIndex * messagesPerVcount +
      segment.baseSecondOffset;

    for (let localIndex = 0; localIndex < segment.messageCount; localIndex += 1) {
      const absoluteOffset = baseOffset + localIndex;
      const timer0Index = Math.floor(absoluteOffset / messagesPerTimer0);
      const remainderAfterTimer0 = absoluteOffset - timer0Index * messagesPerTimer0;
      const vcountIndex = Math.floor(remainderAfterTimer0 / messagesPerVcount);
      const timeCombinationOffset = remainderAfterTimer0 - vcountIndex * messagesPerVcount;

      const timer0 = segment.timer0Min + timer0Index;
      const vcount = segment.vcountMin + vcountIndex;
      const datetime = getDateFromTimePlan(job.timePlan, timeCombinationOffset);
      const message = calculator.generateMessage(conditions, timer0, vcount, datetime, segment.keyCode);
      const { seed } = calculator.calculateSeed(message);

      entries.push({ seed, timer0, vcount, datetime, keyCode: segment.keyCode });

      if (typeof limit === 'number' && limit > 0 && entries.length >= limit) {
        return entries;
      }
    }
  }

  return entries;
}

export function pickUniqueEntries(entries: readonly CpuBaselineEntry[]): CpuBaselineEntry[] {
  const seen = new Set<number>();
  const unique: CpuBaselineEntry[] = [];
  for (const entry of entries) {
    if (seen.has(entry.seed)) {
      continue;
    }
    seen.add(entry.seed);
    unique.push(entry);
  }
  return unique;
}
