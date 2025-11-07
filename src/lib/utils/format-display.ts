import { DomainNatureNames, DomainShinyType } from '@/types/domain';
import { formatHexDisplay } from '@/lib/utils/hex-parser';
import type { GenerationResult } from '@/types/generation';
import type { SupportedLocale } from '@/types/i18n';

export function pidHex(pid: number): string {
  return formatHexDisplay(pid >>> 0, 8, true);
}

export function seedHex(seed: bigint | number): string {
  return formatHexDisplay(typeof seed === 'bigint' ? seed : BigInt(seed >>> 0), 16, true);
}

const UNKNOWN_LABEL: Record<SupportedLocale, string> = {
  en: 'Unknown',
  ja: '不明',
};

export function natureName(id: number, locale: SupportedLocale = 'en'): string {
  const bucket = DomainNatureNames[locale] ?? DomainNatureNames.en;
  if (id < 0 || id >= bucket.length) {
    return UNKNOWN_LABEL[locale] ?? UNKNOWN_LABEL.en;
  }
  return bucket[id];
}

const SHINY_LABELS: Record<SupportedLocale, { normal: string; square: string; star: string; unknown: string }> = {
  en: { normal: '-', square: '◇', star: '☆', unknown: 'Unknown' },
  ja: { normal: '-', square: '◇', star: '☆', unknown: 'Unknown' },
};

export function shinyLabel(t: number, locale: SupportedLocale = 'en'): string {
  const labels = SHINY_LABELS[locale] ?? SHINY_LABELS.en;
  switch (t) {
    case DomainShinyType.Normal: return labels.normal;
    case DomainShinyType.Square: return labels.square;
    case DomainShinyType.Star: return labels.star;
    default: return labels.unknown;
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

export function adaptGenerationResultDisplay(r: GenerationResult, locale: SupportedLocale = 'en') {
  return {
    advance: r.advance,
    pidHex: pidHex(r.pid),
    natureName: natureName(r.nature, locale),
    shinyLabel: shinyLabel(r.shiny_type, locale),
  };
}
