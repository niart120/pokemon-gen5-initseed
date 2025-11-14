import { generateIndividualValues, type IndividualValues } from './individual-values';
import type { SupportedLocale } from '@/types/i18n';

export interface IvTooltipEntry {
  label: string;
  spread: string;
  pattern: string;
}

const LABEL_MAP: Record<SupportedLocale, Record<'wild' | 'roamer' | 'bw2', string>> = {
  ja: {
    wild: 'BW/BW2 野生 (消費0)',
    roamer: 'BW 徘徊 (消費1)',
    bw2: 'BW2 野生 (消費2)',
  },
  en: {
    wild: 'BW/BW2 Wild (offset 0)',
    roamer: 'BW Roamer (offset 1)',
    bw2: 'BW2 Wild (offset 2)',
  },
};

// オフセット値は BW/BW2 で検証済みの IV 消費と一致させる。
const CONTEXTS: Array<{ key: 'wild' | 'roamer' | 'bw2'; offset: number; roamer: boolean }> = [
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
  const labels = LABEL_MAP[locale] ?? LABEL_MAP.ja;
  return CONTEXTS.map(ctx => {
    const values = generateIndividualValues(normalizedSeed, ctx.offset, ctx.roamer);
    return {
      label: labels[ctx.key],
      spread: formatSpread(values),
      pattern: formatPattern(values),
    };
  });
}
