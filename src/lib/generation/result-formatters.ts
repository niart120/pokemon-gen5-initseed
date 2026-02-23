import { formatResultDateTime } from '@/lib/i18n/strings/search-results';

const DEFAULT_TIMER0_FALLBACK = '--';
const DEFAULT_VCOUNT_FALLBACK = '--';

interface HexFormatterOptions {
  fallback?: string;
}

function toHex(value: number, width: number): string {
  return `0x${(value >>> 0).toString(16).toUpperCase().padStart(width, '0')}`;
}

export function formatTimer0Hex(value: number | null | undefined, options?: HexFormatterOptions): string {
  const fallback = options?.fallback ?? DEFAULT_TIMER0_FALLBACK;
  if (value == null) {
    return fallback;
  }
  return toHex(value, 4);
}

export function formatVCountHex(value: number | null | undefined, options?: HexFormatterOptions): string {
  const fallback = options?.fallback ?? DEFAULT_VCOUNT_FALLBACK;
  if (value == null) {
    return fallback;
  }
  return toHex(value, 2);
}

type BootTimestampInput = string | Date | null | undefined;

export function formatBootTimestampDisplay(
  source: BootTimestampInput,
  locale: 'ja' | 'en',
): string {
  if (!source) return '';
  const dt = source instanceof Date ? new Date(source.getTime()) : new Date(source);
  if (Number.isNaN(dt.getTime())) {
    return '';
  }
  return formatResultDateTime(dt, locale);
}

export function buildGenerationResultRowKey(
  advance: number,
  timer0: number | null | undefined,
  vcount: number | null | undefined,
): string {
  const timer0Key = timer0 ?? 'na';
  const vcountKey = vcount ?? 'na';
  return `${advance}-${timer0Key}-${vcountKey}`;
}
