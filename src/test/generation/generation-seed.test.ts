import { beforeAll, describe, expect, it } from 'vitest';
import { calculateGenerationStartSeed } from '@/lib/core/generation-seed';
import { initWasm, getWasm } from '@/lib/core/wasm-interface';
import { domainEncounterTypeToWasm } from '@/lib/core/mapping/encounter-type';
import { DomainEncounterType } from '@/types/domain';

describe('calculateGenerationStartSeed', () => {
  beforeAll(async () => {
    await initWasm();
  });

  it('matches the first enumerated seed', async () => {
    const baseSeed = 0x1234_5678_9abc_def0n;
    const offset = 1024n;
    const startSeed = await calculateGenerationStartSeed(baseSeed, offset);

    const { BWGenerationConfig, GameVersion, SeedEnumerator } = getWasm();
    const config = new BWGenerationConfig(
      GameVersion.B2,
      domainEncounterTypeToWasm(DomainEncounterType.Normal),
      0,
      0,
      false,
      0
    );
    const enumerator = new SeedEnumerator(baseSeed, offset, 1, config);
    const first = enumerator.next_pokemon();
    expect(first).toBeDefined();
    expect(first!.get_seed).toBe(startSeed);
  });
});
