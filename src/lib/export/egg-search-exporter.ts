/**
 * Export functionality for egg boot timing search results
 * Supports CSV, JSON, and text formats
 */

import type { EggBootTimingSearchResult } from '@/types/egg-boot-timing-search';
import type { IvSet, HiddenPowerInfo } from '@/types/egg';
import { natureName } from '@/lib/utils/format-display';
import { keyCodeToNames } from '@/lib/utils/key-input';
import { resolveKeyInputDisplay } from '@/lib/generation/result-formatters';
import type { ExportFormat } from './file-utils';
import type { SupportedLocale } from '@/types/i18n';
import {
  resolveAbilityLabel,
  resolveGenderLabel,
  resolveShinyLabel,
  resolveHiddenPowerTypeName,
} from '@/lib/i18n/strings/export-common';

export interface EggSearchExportOptions {
  format: ExportFormat;
}

export interface EggSearchExportContext {
  locale: SupportedLocale;
}

interface AdaptedEggSearchResult {
  // Boot information
  bootTimestamp: string;
  timer0Hex: string;
  vcountHex: string;
  keyInputDisplay: string;
  lcgSeedHex: string;
  mtSeedHex: string;
  // Egg information
  advance: number;
  ability: 0 | 1 | 2;
  abilityLabel: string;
  gender: 'male' | 'female' | 'genderless';
  genderLabel: string;
  nature: number;
  natureName: string;
  shiny: 0 | 1 | 2;
  shinyLabel: string;
  ivs: IvSet;
  hiddenPower: HiddenPowerInfo;
  hiddenPowerLabel: string;
  pidHex: string;
  pidDec: number;
  isStable: boolean;
}



function formatIv(iv: number): string {
  return iv === 32 ? '?' : String(iv);
}

function formatHiddenPower(hp: HiddenPowerInfo, locale: SupportedLocale): string {
  if (hp.type === 'unknown') {
    return locale === 'ja' ? '不明' : 'Unknown';
  }
  const typeName = resolveHiddenPowerTypeName(hp.hpType, locale);
  return `${typeName}/${hp.power}`;
}

function formatPidHex(pid: number): string {
  return '0x' + (pid >>> 0).toString(16).toUpperCase().padStart(8, '0');
}

function formatTimer0Hex(timer0: number): string {
  return '0x' + timer0.toString(16).toUpperCase().padStart(4, '0');
}

function formatVcountHex(vcount: number): string {
  return '0x' + vcount.toString(16).toUpperCase().padStart(2, '0');
}

function formatDatetime(date: Date): string {
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function adaptEggSearchResults(
  results: EggBootTimingSearchResult[],
  context: EggSearchExportContext
): AdaptedEggSearchResult[] {
  const { locale } = context;

  return results.map((result) => {
    const egg = result.egg.egg;

    // Format key input
    const keyNames = keyCodeToNames(result.boot.keyCode);
    const keyInputDisplay = resolveKeyInputDisplay(keyNames, locale) || '-';

    return {
      bootTimestamp: formatDatetime(result.boot.datetime),
      timer0Hex: formatTimer0Hex(result.boot.timer0),
      vcountHex: formatVcountHex(result.boot.vcount),
      keyInputDisplay,
      lcgSeedHex: result.lcgSeedHex,
      mtSeedHex: egg.mtSeedHex,
      advance: result.egg.advance,
      ability: egg.ability,
      abilityLabel: resolveAbilityLabel(egg.ability, locale),
      gender: egg.gender,
      genderLabel: resolveGenderLabel(egg.gender, locale),
      nature: egg.nature,
      natureName: natureName(egg.nature, locale),
      shiny: egg.shiny,
      shinyLabel: resolveShinyLabel(egg.shiny, locale),
      ivs: egg.ivs,
      hiddenPower: egg.hiddenPower,
      hiddenPowerLabel: formatHiddenPower(egg.hiddenPower, locale),
      pidHex: formatPidHex(egg.pid),
      pidDec: egg.pid >>> 0,
      isStable: result.isStable,
    };
  });
}

export function exportEggSearchResults(
  results: EggBootTimingSearchResult[],
  options: EggSearchExportOptions,
  context: EggSearchExportContext
): string {
  const adapted = adaptEggSearchResults(results, context);

  switch (options.format) {
    case 'csv':
      return formatCsvExport(adapted);
    case 'json':
      return formatJsonExport(adapted, context);
    case 'txt':
      return formatTxtExport(adapted, context);
    default:
      throw new Error(`Unsupported format: ${options.format}`);
  }
}

const CSV_HEADERS = [
  'BootTimestamp',
  'Timer0',
  'VCount',
  'KeyInput',
  'LCGSeed',
  'MTSeed',
  'Advance',
  'Ability',
  'Gender',
  'Nature',
  'Shiny',
  'H',
  'A',
  'B',
  'C',
  'D',
  'S',
  'HiddenPower',
  'PID',
  'Stable',
];

function formatCsvExport(adapted: AdaptedEggSearchResult[]): string {
  const lines = [CSV_HEADERS.join(',')];

  for (const row of adapted) {
    const csvRow = [
      row.bootTimestamp,
      row.timer0Hex,
      row.vcountHex,
      `"${row.keyInputDisplay}"`,
      row.lcgSeedHex,
      row.mtSeedHex,
      String(row.advance),
      row.abilityLabel,
      row.genderLabel,
      row.natureName,
      row.shinyLabel,
      formatIv(row.ivs[0]),
      formatIv(row.ivs[1]),
      formatIv(row.ivs[2]),
      formatIv(row.ivs[3]),
      formatIv(row.ivs[4]),
      formatIv(row.ivs[5]),
      row.hiddenPowerLabel,
      row.pidHex,
      row.isStable ? 'Yes' : 'No',
    ];
    lines.push(csvRow.join(','));
  }

  return lines.join('\n');
}

function formatJsonExport(
  adapted: AdaptedEggSearchResult[],
  context: EggSearchExportContext
): string {
  const data = {
    exportDate: new Date().toISOString(),
    format: 'egg-search-v1',
    locale: context.locale,
    totalResults: adapted.length,
    results: adapted.map((row) => ({
      boot: {
        timestamp: row.bootTimestamp,
        timer0: row.timer0Hex,
        vcount: row.vcountHex,
        keyInput: row.keyInputDisplay,
      },
      lcgSeed: row.lcgSeedHex,
      mtSeed: row.mtSeedHex,
      advance: row.advance,
      ability: row.ability,
      abilityLabel: row.abilityLabel,
      gender: row.gender,
      genderLabel: row.genderLabel,
      nature: row.nature,
      natureName: row.natureName,
      shiny: row.shiny,
      shinyLabel: row.shinyLabel,
      ivs: {
        hp: row.ivs[0],
        atk: row.ivs[1],
        def: row.ivs[2],
        spa: row.ivs[3],
        spd: row.ivs[4],
        spe: row.ivs[5],
      },
      hiddenPower: row.hiddenPower,
      hiddenPowerLabel: row.hiddenPowerLabel,
      pid: row.pidHex,
      pidDecimal: row.pidDec,
      isStable: row.isStable,
    })),
  };

  return JSON.stringify(data, null, 2);
}

function formatTxtExport(
  adapted: AdaptedEggSearchResult[],
  context: EggSearchExportContext
): string {
  const lines: string[] = [];
  const isJa = context.locale === 'ja';

  lines.push(isJa ? 'タマゴ起動時間検索結果 エクスポート' : 'Egg Boot Timing Search Results Export');
  lines.push(`Export Date: ${new Date().toISOString()}`);
  lines.push(`Total Results: ${adapted.length}`);
  lines.push('');

  for (let i = 0; i < adapted.length; i++) {
    const row = adapted[i];
    lines.push(`Result #${i + 1}`);
    lines.push(`  Boot Timestamp: ${row.bootTimestamp}`);
    lines.push(`  Timer0: ${row.timer0Hex}`);
    lines.push(`  VCount: ${row.vcountHex}`);
    lines.push(`  Key Input: ${row.keyInputDisplay}`);
    lines.push(`  LCG Seed: ${row.lcgSeedHex}`);
    lines.push(`  MT Seed: ${row.mtSeedHex}`);
    lines.push(`  Advance: ${row.advance}`);
    lines.push(`  Ability: ${row.abilityLabel}`);
    lines.push(`  Gender: ${row.genderLabel}`);
    lines.push(`  Nature: ${row.natureName}`);
    lines.push(`  Shiny: ${row.shinyLabel}`);
    lines.push(`  IVs: ${row.ivs.map(formatIv).join('-')}`);
    lines.push(`  Hidden Power: ${row.hiddenPowerLabel}`);
    lines.push(`  PID: ${row.pidHex}`);
    lines.push(`  Stable: ${row.isStable ? 'Yes' : 'No'}`);
    lines.push('');
  }

  return lines.join('\n');
}
