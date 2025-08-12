import { DomainNatureNames, DomainShinyType } from '@/types/domain';
import { formatHexDisplay } from '@/lib/utils/hex-parser';
import type { GenerationResult } from '@/types/generation';

export function pidHex(pid: number): string {
  return formatHexDisplay(pid >>> 0, 8, true);
}

export function seedHex(seed: bigint | number): string {
  return formatHexDisplay(typeof seed === 'bigint' ? seed : BigInt(seed >>> 0), 16, true);
}

export function natureName(id: number): string {
  if (id < 0 || id >= DomainNatureNames.length) return 'Unknown';
  return DomainNatureNames[id];
}

export function shinyLabel(t: number): 'No' | 'Square' | 'Star' | 'Unknown' {
  switch (t) {
    case DomainShinyType.Normal: return 'No';
    case DomainShinyType.Square: return 'Square';
    case DomainShinyType.Star: return 'Star';
    default: return 'Unknown';
  }
}

export function shinyDomainStatus(t: number): 'normal' | 'square' | 'star' {
  switch (t) {
    case DomainShinyType.Square: return 'square';
    case DomainShinyType.Star: return 'star';
    case DomainShinyType.Normal:
    default: return 'normal';
  }
}

export function adaptGenerationResultDisplay(r: GenerationResult) {
  return {
    advance: r.advance,
    pidHex: pidHex(r.pid),
    natureName: natureName(r.nature),
    shinyLabel: shinyLabel(r.shiny_type),
  };
}
