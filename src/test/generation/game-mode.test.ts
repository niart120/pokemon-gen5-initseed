import { describe, it, expect } from 'vitest';
import { deriveDomainGameMode, type GenerationParams } from '@/types/generation';
import { DomainGameMode } from '@/types/domain';

function params(partial: Partial<GenerationParams>) {
  return {
    version: 'B' as GenerationParams['version'],
    newGame: true,
    withSave: true,
    memoryLink: false,
    ...partial,
  } satisfies Pick<GenerationParams, 'version' | 'newGame' | 'withSave' | 'memoryLink'>;
}

describe('deriveDomainGameMode', () => {
  it('maps BW new game with save', () => {
    expect(deriveDomainGameMode(params({ version: 'B', newGame: true, withSave: true }))).toBe(DomainGameMode.BwNewGameWithSave);
  });

  it('maps BW new game without save', () => {
    expect(deriveDomainGameMode(params({ version: 'W', newGame: true, withSave: false }))).toBe(DomainGameMode.BwNewGameNoSave);
  });

  it('maps BW continue', () => {
    expect(deriveDomainGameMode(params({ version: 'B', newGame: false }))).toBe(DomainGameMode.BwContinue);
  });

  it('maps BW2 continue with memory link', () => {
    expect(deriveDomainGameMode(params({ version: 'B2', newGame: false, memoryLink: true }))).toBe(DomainGameMode.Bw2ContinueWithMemoryLink);
  });

  it('maps BW2 new game with save and no memory link', () => {
    expect(deriveDomainGameMode(params({ version: 'W2', newGame: true, withSave: true, memoryLink: false }))).toBe(DomainGameMode.Bw2NewGameNoMemoryLinkSave);
  });

  it('maps BW2 new game without save', () => {
    expect(deriveDomainGameMode(params({ version: 'B2', newGame: true, withSave: false }))).toBe(DomainGameMode.Bw2NewGameNoSave);
  });

  it('throws when memory link is set without save', () => {
    expect(() => deriveDomainGameMode(params({ version: 'W2', newGame: true, withSave: false, memoryLink: true })) ).toThrowError();
  });
});
