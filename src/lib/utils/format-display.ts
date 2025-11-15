import { DomainNatureNames, DomainShinyType } from '@/types/domain';
import { formatHexDisplay } from '@/lib/utils/hex-parser';
import type { GenerationResult } from '@/types/generation';
import type { SupportedLocale } from '@/types/i18n';
import {
  resolveDisplayUnknownLabel,
  resolveShinyLabel,
  type ShinyLabelKey,
} from '@/lib/i18n/strings/display-common';

export function pidHex(pid: number): string {
  return formatHexDisplay(pid >>> 0, 8, true);
}

export function seedHex(seed: bigint | number): string {
  return formatHexDisplay(typeof seed === 'bigint' ? seed : BigInt(seed >>> 0), 16, true);
}

export function natureName(id: number, locale: SupportedLocale = 'en'): string {
  const bucket = DomainNatureNames[locale] ?? DomainNatureNames.en;
  if (id < 0 || id >= bucket.length) {
    return resolveDisplayUnknownLabel(locale);
  }
  return bucket[id];
}

export function shinyLabel(t: number, locale: SupportedLocale = 'en'): string {
  let key: ShinyLabelKey;
  switch (t) {
    case DomainShinyType.Normal:
      key = 'normal';
      break;
    case DomainShinyType.Square:
      key = 'square';
      break;
    case DomainShinyType.Star:
      key = 'star';
      break;
    default:
      key = 'unknown';
      break;
  }
  return resolveShinyLabel(key, locale);
}

export function shinyDomainStatus(t: number): 'normal' | 'square' | 'star' {
  switch (t) {
    case DomainShinyType.Square: return 'square';
    case DomainShinyType.Star: return 'star';
    case DomainShinyType.Normal:
    default: return 'normal';
  }
}

/**
 * Calculate needle direction value from LCG seed
 * Formula: ((seed >>> 32n) * 8n) >> 32n
 * Returns value 0-7 representing 8 directions
 */
export function calculateNeedleDirection(seed: bigint): number {
  const direction = ((seed >> 32n) * 8n) >> 32n;
  return Number(direction & 7n); // Ensure 0-7 range
}

/**
 * Map needle direction value to arrow character
 * 0: ↑, 1: ↗, 2: →, 3: ↘, 4: ↓, 5: ↙, 6: ←, 7: ↖
 */
export function needleDirectionArrow(direction: number): string {
  const arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'];
  return arrows[direction] ?? '?';
}

/**
 * Get formatted needle display with both arrow and value
 * e.g., "↑(0)", "↗(1)", etc.
 */
export function needleDisplay(seed: bigint): string {
  const direction = calculateNeedleDirection(seed);
  const arrow = needleDirectionArrow(direction);
  return `${arrow}(${direction})`;
}

export function adaptGenerationResultDisplay(r: GenerationResult, locale: SupportedLocale = 'en') {
  return {
    advance: r.advance,
    pidHex: pidHex(r.pid),
    natureName: natureName(r.nature, locale),
    shinyLabel: shinyLabel(r.shiny_type, locale),
  };
}
