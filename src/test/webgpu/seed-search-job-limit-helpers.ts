import type { SeedSearchJobLimits } from '@/lib/webgpu/seed-search/types';

const BASE_TEST_LIMITS: SeedSearchJobLimits = {
  workgroupSize: 32,
  maxWorkgroupsPerDispatch: 256,
  candidateCapacityPerDispatch: 512,
  maxMessagesPerDispatch: 32 * 256,
  maxDispatchesInFlight: 2,
};

export function createTestSeedSearchJobLimits(
  overrides?: Partial<SeedSearchJobLimits>
): SeedSearchJobLimits {
  const workgroupSize = overrides?.workgroupSize ?? BASE_TEST_LIMITS.workgroupSize;
  const maxWorkgroupsPerDispatch =
    overrides?.maxWorkgroupsPerDispatch ?? BASE_TEST_LIMITS.maxWorkgroupsPerDispatch;
  const candidateCapacityPerDispatch =
    overrides?.candidateCapacityPerDispatch ?? BASE_TEST_LIMITS.candidateCapacityPerDispatch;
  const requestedMaxMessages = overrides?.maxMessagesPerDispatch ?? BASE_TEST_LIMITS.maxMessagesPerDispatch;
  const maxMessagesPerDispatch = Math.min(
    requestedMaxMessages,
    Math.max(1, workgroupSize * maxWorkgroupsPerDispatch)
  );
  const maxDispatchesInFlight = overrides?.maxDispatchesInFlight ?? BASE_TEST_LIMITS.maxDispatchesInFlight;

  return {
    workgroupSize,
    maxWorkgroupsPerDispatch,
    candidateCapacityPerDispatch,
    maxMessagesPerDispatch,
    maxDispatchesInFlight,
  };
}
