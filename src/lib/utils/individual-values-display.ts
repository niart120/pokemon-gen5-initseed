import { generateIndividualValues, type IndividualValues } from './individual-values';
import type { SupportedLocale } from '@/types/i18n';
import { resolveIvTooltipLabel, type IvTooltipContextKey } from '@/lib/i18n/strings/individual-values';

export interface IvTooltipEntry {
  label: string;
  spread: string;
  pattern: string;
}

// オフセット値は BW/BW2 で検証済みの IV 消費と一致させる。
const CONTEXTS: Array<{ key: IvTooltipContextKey; offset: number; roamer: boolean }> = [
  { key: 'wild', offset: 0, roamer: false },
  { key: 'roamer', offset: 1, roamer: true },
  { key: 'bw2', offset: 2, roamer: false },
];

const PATTERN_DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

function formatSpread(values: IndividualValues): string {
  return `H${values.h} A${values.a} B${values.b} C${values.c} D${values.d} S${values.s}`;
}

function formatPattern(values: IndividualValues): string {
  return [values.h, values.a, values.b, values.c, values.d, values.s]
    .map(value => PATTERN_DIGITS[value] ?? '?')
    .join('');
}

export function getIvTooltipEntries(seed: number, locale: SupportedLocale): IvTooltipEntry[] {
  const normalizedSeed = seed >>> 0;
  return CONTEXTS.map(ctx => {
    const values = generateIndividualValues(normalizedSeed, ctx.offset, ctx.roamer);
    return {
      label: resolveIvTooltipLabel(ctx.key, locale),
      spread: formatSpread(values),
      pattern: formatPattern(values),
    };
  });
}
