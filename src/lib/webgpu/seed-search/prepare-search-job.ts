import romParameters from '@/data/rom-parameters';
import { hasImpossibleKeyCombination, KEY_CODE_BASE } from '@/lib/utils/key-input';
import type { Hardware } from '@/types/rom';
import type { SearchConditions } from '@/types/search';
import { resolveTimePlan, type ResolvedTimePlan } from '@/lib/search/time/time-plan';
import type {
  SeedSearchJob,
  SeedSearchJobLimits,
  SeedSearchJobOptions,
  SeedSearchJobSegment,
} from './types';

const GX_STAT = 0x06000000;
const MAX_U32 = 0xffffffff;

interface Timer0SegmentDescriptor {
  timer0Min: number;
  timer0Max: number;
  vcount: number;
}

interface KernelContext {
  rangeSeconds: number;
  timer0Segments: Timer0SegmentDescriptor[];
  keyCodes: number[];
  nazoSwapped: Uint32Array;
  macLower: number;
  data7Swapped: number;
  hardwareType: number;
  startYear: number;
  startDayOfYear: number;
  startSecondOfDay: number;
  startDayOfWeek: number;
  dayCount: number;
  hourRangeStart: number;
  hourRangeCount: number;
  minuteRangeStart: number;
  minuteRangeCount: number;
  secondRangeStart: number;
  secondRangeCount: number;
}

export function prepareSearchJob(
  conditions: SearchConditions,
  targetSeedsInput: readonly number[] = [],
  options?: SeedSearchJobOptions
): SeedSearchJob {
  const { plan, firstCombinationDate } = resolveTimePlan(conditions);
  const limits = resolveRequiredDispatchLimits(options);
  const targetSeeds = buildTargetSeedArray(targetSeedsInput);
  const kernelContext = buildKernelContext(conditions, plan, firstCombinationDate);
  const segments = buildSegments(kernelContext, limits);

  const totalMessages = segments.reduce((sum, segment) => sum + segment.messageCount, 0);

  return {
    segments,
    targetSeeds,
    timePlan: plan,
    summary: {
      totalMessages,
      totalSegments: segments.length,
      targetSeedCount: targetSeeds.length,
      rangeSeconds: kernelContext.rangeSeconds,
    },
    limits,
    conditions,
  };
}

function buildSegments(context: KernelContext, limits: SeedSearchJobLimits): SeedSearchJobSegment[] {
  const segments: SeedSearchJobSegment[] = [];
  if (context.rangeSeconds <= 0) {
    return segments;
  }

  const maxMessagesByWorkgroups = Math.max(1, limits.workgroupSize * limits.maxWorkgroupsPerDispatch);
  const chunkSizeLimit = Math.min(limits.maxMessagesPerDispatch, maxMessagesByWorkgroups);

  let globalOffset = 0;
  let segmentCounter = 0;

  for (const keyCode of context.keyCodes) {
    const keyInputSwapped = swap32(keyCode >>> 0);

    for (const timerSegment of context.timer0Segments) {
      const timer0Count = timerSegment.timer0Max - timerSegment.timer0Min + 1;
      if (timer0Count <= 0) {
        continue;
      }

      const vcountCount = 1;
      const totalMessagesForSegment = context.rangeSeconds * timer0Count;
      let remaining = totalMessagesForSegment;
      let localOffset = 0;

      while (remaining > 0) {
        const messageCount = Math.min(remaining, chunkSizeLimit);
        const workgroupCount = computeWorkgroupCount(messageCount, limits);
        const baseIndices = computeBaseIndices(context.rangeSeconds, vcountCount, localOffset);
        const configWords = encodeConfigWords({
          messageCount,
          baseTimer0Index: baseIndices.baseTimer0Index,
          baseVcountIndex: baseIndices.baseVcountIndex,
          baseSecondOffset: baseIndices.baseSecondOffset,
          rangeSeconds: context.rangeSeconds,
          timer0Min: timerSegment.timer0Min,
          timer0Count,
          vcountMin: timerSegment.vcount,
          vcountCount,
          startSecondOfDay: context.startSecondOfDay,
          startDayOfWeek: context.startDayOfWeek,
          macLower: context.macLower,
          data7Swapped: context.data7Swapped,
          keyInputSwapped,
          hardwareType: context.hardwareType,
          nazoSwapped: context.nazoSwapped,
          startYear: context.startYear,
          startDayOfYear: context.startDayOfYear,
          dayCount: context.dayCount,
          hourRangeStart: context.hourRangeStart,
          hourRangeCount: context.hourRangeCount,
          minuteRangeStart: context.minuteRangeStart,
          minuteRangeCount: context.minuteRangeCount,
          secondRangeStart: context.secondRangeStart,
          secondRangeCount: context.secondRangeCount,
          groupsPerDispatch: workgroupCount,
          workgroupSize: limits.workgroupSize,
          candidateCapacity: limits.candidateCapacityPerDispatch,
        });

        segments.push({
          id: `seg-${segmentCounter}`,
          keyCode,
          timer0Min: timerSegment.timer0Min,
          timer0Max: timerSegment.timer0Max,
          timer0Count,
          vcountMin: timerSegment.vcount,
          vcountCount,
          rangeSeconds: context.rangeSeconds,
          messageCount,
          localMessageOffset: localOffset,
          globalMessageOffset: globalOffset,
          baseTimer0Index: baseIndices.baseTimer0Index,
          baseVcountIndex: baseIndices.baseVcountIndex,
          baseSecondOffset: baseIndices.baseSecondOffset,
          workgroupCount,
          configWords,
        });

        remaining -= messageCount;
        localOffset += messageCount;
        globalOffset += messageCount;
        segmentCounter += 1;
      }
    }
  }

  return segments;
}

function computeWorkgroupCount(messageCount: number, limits: SeedSearchJobLimits): number {
  const totalWorkgroupsNeeded = Math.max(1, Math.ceil(messageCount / limits.workgroupSize));
  const maxWorkgroups = Math.max(1, limits.maxWorkgroupsPerDispatch);
  return Math.min(totalWorkgroupsNeeded, maxWorkgroups);
}

function resolveRequiredDispatchLimits(options?: SeedSearchJobOptions): SeedSearchJobLimits {
  if (!options?.limits) {
    throw new Error('Seed search job limits are required for WebGPU execution');
  }
  return sanitizeDispatchLimits(options.limits);
}

function sanitizeDispatchLimits(limits: SeedSearchJobLimits): SeedSearchJobLimits {
  const workgroupSize = clampPositiveInteger(limits.workgroupSize, 'workgroupSize');
  const maxWorkgroupsPerDispatch = clampPositiveInteger(
    limits.maxWorkgroupsPerDispatch,
    'maxWorkgroupsPerDispatch'
  );
  const candidateCapacityPerDispatch = clampPositiveInteger(
    limits.candidateCapacityPerDispatch,
    'candidateCapacityPerDispatch'
  );
  const requestedMaxMessages = clampPositiveInteger(
    limits.maxMessagesPerDispatch,
    'maxMessagesPerDispatch'
  );
  const maxWorkgroupsByMessages = Math.max(1, Math.floor(MAX_U32 / Math.max(1, workgroupSize)));
  const safeWorkgroupsPerDispatch = Math.min(maxWorkgroupsPerDispatch, maxWorkgroupsByMessages);
  const maxMessagesByWorkgroups = Math.max(1, workgroupSize * safeWorkgroupsPerDispatch);
  const maxMessagesPerDispatch = Math.min(requestedMaxMessages, maxMessagesByWorkgroups);
  return {
    workgroupSize,
    maxWorkgroupsPerDispatch: safeWorkgroupsPerDispatch,
    candidateCapacityPerDispatch,
    maxMessagesPerDispatch,
  };
}

function buildKernelContext(
  conditions: SearchConditions,
  plan: ResolvedTimePlan['plan'],
  startDate: Date
): KernelContext {
  const params = resolveRomParameters(conditions);
  const keyCodes = generateKeyCodes(conditions.keyInput);
  if (keyCodes.length === 0) {
    throw new Error('入力されたキー条件から生成できる組み合わせがありません');
  }

  const timer0Segments = resolveTimer0Segments(conditions, params);
  if (timer0Segments.length === 0) {
    throw new Error('timer0の範囲が正しく設定されていません');
  }

  const nazoSwapped = createNazoSwapped(params.nazo);
  const { macLower, data7Swapped } = computeMacWords(conditions.macAddress, HARDWARE_FRAME_VALUES[conditions.hardware]);

  const rangeSeconds = plan.dayCount * plan.combosPerDay;
  if (rangeSeconds <= 0) {
    throw new Error('探索対象の秒数が0以下です');
  }

  return {
    rangeSeconds,
    timer0Segments,
    keyCodes,
    nazoSwapped,
    macLower,
    data7Swapped,
    hardwareType: mapHardwareToId(conditions.hardware),
    startYear: startDate.getFullYear(),
    startDayOfYear: calculateLocalDayOfYear(startDate),
    startSecondOfDay: calculateSecondOfDay(startDate),
    startDayOfWeek: startDate.getDay(),
    dayCount: plan.dayCount,
    hourRangeStart: plan.hourRangeStart,
    hourRangeCount: plan.hourRangeCount,
    minuteRangeStart: plan.minuteRangeStart,
    minuteRangeCount: plan.minuteRangeCount,
    secondRangeStart: plan.secondRangeStart,
    secondRangeCount: plan.secondRangeCount,
  };
}

function buildTargetSeedArray(values: readonly number[]): Uint32Array {
  if (!values || values.length === 0) {
    return new Uint32Array(0);
  }

  const sanitized: number[] = [];
  for (const candidate of values) {
    if (typeof candidate !== 'number' || !Number.isFinite(candidate)) {
      continue;
    }
    sanitized.push(candidate >>> 0);
  }

  return Uint32Array.from(sanitized);
}

function clampPositiveInteger(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number`);
  }
  return Math.floor(value);
}

function computeBaseIndices(rangeSeconds: number, vcountCount: number, baseOffset: number) {
  const safeRangeSeconds = Math.max(rangeSeconds, 1);
  const safeVcountCount = Math.max(vcountCount, 1);
  const messagesPerVcount = safeRangeSeconds;
  const messagesPerTimer0 = messagesPerVcount * safeVcountCount;
  const baseTimer0Index = Math.floor(baseOffset / messagesPerTimer0);
  const remainderAfterTimer0 = baseOffset - baseTimer0Index * messagesPerTimer0;
  const baseVcountIndex = Math.floor(remainderAfterTimer0 / messagesPerVcount);
  const baseSecondOffset = remainderAfterTimer0 - baseVcountIndex * messagesPerVcount;
  return { baseTimer0Index, baseVcountIndex, baseSecondOffset };
}

interface EncodeConfigWordsParams {
  messageCount: number;
  baseTimer0Index: number;
  baseVcountIndex: number;
  baseSecondOffset: number;
  rangeSeconds: number;
  timer0Min: number;
  timer0Count: number;
  vcountMin: number;
  vcountCount: number;
  startSecondOfDay: number;
  startDayOfWeek: number;
  macLower: number;
  data7Swapped: number;
  keyInputSwapped: number;
  hardwareType: number;
  nazoSwapped: Uint32Array;
  startYear: number;
  startDayOfYear: number;
  dayCount: number;
  hourRangeStart: number;
  hourRangeCount: number;
  minuteRangeStart: number;
  minuteRangeCount: number;
  secondRangeStart: number;
  secondRangeCount: number;
  groupsPerDispatch: number;
  workgroupSize: number;
  candidateCapacity: number;
}

function encodeConfigWords(params: EncodeConfigWordsParams): Uint32Array {
  const data = new Uint32Array(32);
  data[0] = params.messageCount >>> 0;
  data[1] = params.baseTimer0Index >>> 0;
  data[2] = params.baseVcountIndex >>> 0;
  data[3] = params.baseSecondOffset >>> 0;
  data[4] = params.rangeSeconds >>> 0;
  data[5] = params.timer0Min >>> 0;
  data[6] = params.timer0Count >>> 0;
  data[7] = params.vcountMin >>> 0;
  data[8] = params.vcountCount >>> 0;
  data[9] = params.startSecondOfDay >>> 0;
  data[10] = params.startDayOfWeek >>> 0;
  data[11] = params.macLower >>> 0;
  data[12] = params.data7Swapped >>> 0;
  data[13] = params.keyInputSwapped >>> 0;
  data[14] = params.hardwareType >>> 0;
  for (let i = 0; i < params.nazoSwapped.length && (15 + i) < 20; i += 1) {
    data[15 + i] = params.nazoSwapped[i] >>> 0;
  }
  data[20] = params.startYear >>> 0;
  data[21] = params.startDayOfYear >>> 0;
  data[22] = params.groupsPerDispatch >>> 0;
  data[23] = params.workgroupSize >>> 0;
  data[24] = params.candidateCapacity >>> 0;
  data[25] = params.dayCount >>> 0;
  data[26] = params.hourRangeStart >>> 0;
  data[27] = params.hourRangeCount >>> 0;
  data[28] = params.minuteRangeStart >>> 0;
  data[29] = params.minuteRangeCount >>> 0;
  data[30] = params.secondRangeStart >>> 0;
  data[31] = params.secondRangeCount >>> 0;
  return data;
}

const HARDWARE_FRAME_VALUES: Record<Hardware, number> = {
  DS: 8,
  DS_LITE: 6,
  '3DS': 9,
};

function resolveRomParameters(conditions: SearchConditions) {
  const versionData = romParameters[conditions.romVersion as keyof typeof romParameters];
  if (!versionData) {
    throw new Error(`ROMバージョン ${conditions.romVersion} は未対応です`);
  }

  const regionData = versionData[conditions.romRegion as keyof typeof versionData];
  if (!regionData) {
    throw new Error(`ROMリージョン ${conditions.romRegion} は未対応です`);
  }

  return {
    nazo: [...regionData.nazo] as [number, number, number, number, number],
    vcountTimerRanges: regionData.vcountTimerRanges.map((entry) => [...entry] as [number, number, number]),
  };
}

function generateKeyCodes(keyInputMask: number): number[] {
  const enabledBits: number[] = [];
  for (let bit = 0; bit < 12; bit += 1) {
    if ((keyInputMask & (1 << bit)) !== 0) {
      enabledBits.push(bit);
    }
  }

  const keyCodes: number[] = [];
  const totalCombinations = 1 << enabledBits.length;
  for (let combo = 0; combo < totalCombinations; combo += 1) {
    let pressedMask = 0;
    for (let bitIndex = 0; bitIndex < enabledBits.length; bitIndex += 1) {
      if ((combo & (1 << bitIndex)) !== 0) {
        pressedMask |= 1 << enabledBits[bitIndex];
      }
    }
    if (hasImpossibleKeyCombination(pressedMask)) {
      continue;
    }
    keyCodes.push((pressedMask ^ KEY_CODE_BASE) >>> 0);
  }

  return keyCodes;
}

function resolveTimer0Segments(
  conditions: SearchConditions,
  params: { vcountTimerRanges: readonly [number, number, number][] }
): Timer0SegmentDescriptor[] {
  const segments: Timer0SegmentDescriptor[] = [];
  const timer0Min = conditions.timer0VCountConfig.timer0Range.min;
  const timer0Max = conditions.timer0VCountConfig.timer0Range.max;

  let current: Timer0SegmentDescriptor | null = null;

  for (let timer0 = timer0Min; timer0 <= timer0Max; timer0 += 1) {
    const vcount = getVCountForTimer0(params, timer0);
    if (current && current.vcount === vcount && timer0 === current.timer0Max + 1) {
      current.timer0Max = timer0;
    } else {
      if (current) {
        segments.push(current);
      }
      current = { timer0Min: timer0, timer0Max: timer0, vcount };
    }
  }

  if (current) {
    segments.push(current);
  }

  return segments;
}

function getVCountForTimer0(params: { vcountTimerRanges: readonly [number, number, number][] }, timer0: number): number {
  for (const [vcount, min, max] of params.vcountTimerRanges) {
    if (timer0 >= min && timer0 <= max) {
      return vcount;
    }
  }
  return params.vcountTimerRanges.length > 0 ? params.vcountTimerRanges[0][0] : 0x60;
}

function computeMacWords(mac: number[], frame: number): { macLower: number; data7Swapped: number } {
  const safeMac = normalizeMac(mac);
  const macLower = ((safeMac[4] & 0xff) << 8) | (safeMac[5] & 0xff);
  const macUpper =
    (safeMac[0] & 0xff) |
    ((safeMac[1] & 0xff) << 8) |
    ((safeMac[2] & 0xff) << 16) |
    ((safeMac[3] & 0xff) << 24);
  const data7 = (macUpper ^ GX_STAT ^ frame) >>> 0;
  return { macLower, data7Swapped: swap32(data7) };
}

function normalizeMac(mac: number[]): number[] {
  const result = new Array(6).fill(0);
  for (let i = 0; i < 6; i += 1) {
    const value = mac[i] ?? 0;
    result[i] = (Number(value) & 0xff) >>> 0;
  }
  return result;
}

function mapHardwareToId(hardware: Hardware): number {
  switch (hardware) {
    case 'DS':
      return 0;
    case 'DS_LITE':
      return 1;
    case '3DS':
      return 2;
    default:
      return 0;
  }
}

function swap32(value: number): number {
  return (
    ((value & 0xff) << 24) |
    (((value >>> 8) & 0xff) << 16) |
    (((value >>> 16) & 0xff) << 8) |
    ((value >>> 24) & 0xff)
  ) >>> 0;
}

function createNazoSwapped(nazo: readonly number[]): Uint32Array {
  const array = new Uint32Array(nazo.length);
  for (let i = 0; i < nazo.length; i += 1) {
    array[i] = swap32(nazo[i] >>> 0);
  }
  return array;
}

function calculateLocalDayOfYear(date: Date): number {
  const startOfYear = new Date(date.getFullYear(), 0, 1);
  const diffMs = date.getTime() - startOfYear.getTime();
  return Math.floor(diffMs / (24 * 60 * 60 * 1000)) + 1;
}

function calculateSecondOfDay(date: Date): number {
  return date.getHours() * 3600 + date.getMinutes() * 60 + date.getSeconds();
}
