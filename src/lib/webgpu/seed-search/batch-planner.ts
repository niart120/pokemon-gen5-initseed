import { DOUBLE_BUFFER_SET_COUNT, MATCH_RECORD_BYTES } from './constants';
import { estimateHostMemoryLimitBytes } from './environment';
import type { WebGpuBatchPlan } from './types';
import type { WebGpuDeviceContext } from './device-context';

export interface WebGpuBatchPlannerOptions {
  hostMemoryLimitBytes?: number;
  bufferSetCount?: number;
  workgroupSize?: number;
  maxMessagesOverride?: number | null;
  hostMemoryLimitPerSlot?: number;
}

export interface WebGpuBatchPlanner {
  computePlan(totalMessages: number): WebGpuBatchPlan;
}

export function createWebGpuBatchPlanner(
  context: WebGpuDeviceContext,
  options?: WebGpuBatchPlannerOptions
): WebGpuBatchPlanner {
  const hostMemoryLimitBytes = (() => {
    if (
      typeof options?.hostMemoryLimitBytes === 'number' &&
      Number.isFinite(options.hostMemoryLimitBytes) &&
      options.hostMemoryLimitBytes > 0
    ) {
      return Math.max(MATCH_RECORD_BYTES, Math.floor(options.hostMemoryLimitBytes));
    }
    return estimateHostMemoryLimitBytes();
  })();
  const bufferSetCount = options?.bufferSetCount ?? DOUBLE_BUFFER_SET_COUNT;
  const hostMemoryLimitPerSlot = (() => {
    const provided = options?.hostMemoryLimitPerSlot;
    if (typeof provided === 'number' && Number.isFinite(provided) && provided > 0) {
      return Math.max(MATCH_RECORD_BYTES, Math.floor(provided));
    }
    const calculated = Math.floor(hostMemoryLimitBytes / bufferSetCount);
    return Math.max(MATCH_RECORD_BYTES, calculated);
  })();
  const workgroupSize = (() => {
    if (typeof options?.workgroupSize === 'number' && Number.isFinite(options.workgroupSize) && options.workgroupSize > 0) {
      return Math.floor(options.workgroupSize);
    }
    return context.getSupportedWorkgroupSize();
  })();
  const maxMessagesOverride = typeof options?.maxMessagesOverride === 'number'
    ? Math.max(1, Math.floor(options.maxMessagesOverride))
    : null;

  if (hostMemoryLimitBytes <= 0) {
    throw new Error('host memory limit must be positive');
  }
  if (bufferSetCount <= 0) {
    throw new Error('buffer set count must be positive');
  }

  const resolveMessagesPerDispatch = (totalMessages: number): number => {
    if (maxMessagesOverride !== null) {
      return Math.min(maxMessagesOverride, totalMessages || 1);
    }

    const device = context.getDevice();
    const limits = device.limits;

    const storageLimit = Math.max(1, limits.maxStorageBufferBindingSize ?? MATCH_RECORD_BYTES);
    const maxByStorage = Math.max(1, Math.floor(storageLimit / MATCH_RECORD_BYTES));

    const maxByHost = Math.max(
      1,
      Math.floor(hostMemoryLimitPerSlot / MATCH_RECORD_BYTES)
    );

    const supportedWorkgroupSize = context.getSupportedWorkgroupSize(workgroupSize);
    const maxWorkgroups = limits.maxComputeWorkgroupsPerDimension ?? 65535;
    const maxByWorkload = Math.max(1, supportedWorkgroupSize * maxWorkgroups);

    const rawMax = Math.min(maxByStorage, maxByHost, maxByWorkload);

    if (totalMessages <= rawMax) {
      if (totalMessages <= 1) {
        return 1;
      }
      return Math.max(1, Math.min(rawMax, Math.ceil(totalMessages / 2)));
    }

    return rawMax;
  };

  const computePlan = (totalMessages: number): WebGpuBatchPlan => {
    if (!Number.isFinite(totalMessages) || totalMessages < 0) {
      throw new Error('totalMessages must be a non-negative finite value');
    }

    if (totalMessages === 0) {
      return {
        maxMessagesPerDispatch: 0,
        dispatches: [],
      };
    }

    const maxMessagesPerDispatch = resolveMessagesPerDispatch(totalMessages);

    const dispatches = [] as WebGpuBatchPlan['dispatches'];
    let remaining = totalMessages;
    let baseOffset = 0;
    while (remaining > 0) {
      const count = Math.min(maxMessagesPerDispatch, remaining);
      dispatches.push({ baseOffset, messageCount: count });
      baseOffset += count;
      remaining -= count;
    }

    if (dispatches.length === 1 && totalMessages > 1) {
      const first = dispatches[0];
      const firstHalf = Math.ceil(first.messageCount / 2);
      const secondHalf = first.messageCount - firstHalf;
      if (secondHalf > 0) {
        dispatches[0] = { baseOffset: first.baseOffset, messageCount: firstHalf };
        dispatches.push({ baseOffset: first.baseOffset + firstHalf, messageCount: secondHalf });
      }
    }

    return {
      maxMessagesPerDispatch,
      dispatches,
    };
  };

  return { computePlan };
}
