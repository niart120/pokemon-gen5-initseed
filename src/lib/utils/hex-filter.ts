export interface HexFilterParseOptions {
  maxValue: number;
}

function clampToRange(value: number, maxValue: number): number {
  if (Number.isNaN(value) || value < 0) return 0;
  if (value > maxValue) return maxValue;
  return value;
}

export function normalizeHexFilterInput(value?: string): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  return trimmed.toUpperCase();
}

export function parseHexFilterValue(value: string | undefined, { maxValue }: HexFilterParseOptions): number | null {
  const normalized = normalizeHexFilterInput(value);
  if (!normalized) {
    return null;
  }

  let token = normalized.replace(/\s+/g, '');
  if (!token) {
    return null;
  }

  let base = 10;
  if (token.startsWith('0X')) {
    base = 16;
    token = token.slice(2);
  } else if (/[A-F]/.test(token)) {
    base = 16;
  }

  const pattern = base === 16 ? /^[0-9A-F]+$/ : /^[0-9]+$/;
  if (!pattern.test(token)) {
    return null;
  }

  const parsed = parseInt(token, base);
  if (Number.isNaN(parsed)) {
    return null;
  }

  return clampToRange(parsed, maxValue);
}
