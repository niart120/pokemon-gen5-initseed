/**
 * Export functionality for egg generation results
 * Supports CSV, JSON, and text formats
 */

import type { EnumeratedEggDataWithBootTiming, IvSet, HiddenPowerInfo } from '@/types/egg';
import { natureName, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import type { ExportFormat } from './file-utils';
import type { SupportedLocale } from '@/types/i18n';
import {
  resolveAbilityLabel,
  resolveGenderLabel,
  resolveShinyLabel,
  resolveHiddenPowerTypeName,
} from '@/lib/i18n/strings/export-common';

export interface EggGenerationExportOptions {
  format: ExportFormat;
  includeBootTiming?: boolean;
}

export interface EggGenerationExportContext {
  locale: SupportedLocale;
  isBootTimingMode: boolean;
}

interface AdaptedEggResult {
  advance: number;
  directionArrow: string;
  directionValue: number;
  lcgSeedHex: string;
  mtSeedHex: string;
  pidHex: string;
  pidDec: number;
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
  isStable: boolean;
  // Boot-Timing fields (optional)
  timer0Hex?: string;
  vcountHex?: string;
  bootTimestampIso?: string;
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

function adaptEggResults(
  results: EnumeratedEggDataWithBootTiming[],
  context: EggGenerationExportContext
): AdaptedEggResult[] {
  const { locale, isBootTimingMode } = context;

  return results.map((row) => {
    let directionValue = -1;
    let directionArrow = '?';

    // Calculate needle direction from LCG seed
    const seedHex = row.egg.lcgSeedHex;
    if (seedHex) {
      try {
        const seed = BigInt(seedHex);
        directionValue = calculateNeedleDirection(seed);
        directionArrow = needleDirectionArrow(directionValue);
      } catch {
        directionValue = -1;
        directionArrow = '?';
      }
    }

    const adapted: AdaptedEggResult = {
      advance: row.advance,
      directionArrow,
      directionValue,
      lcgSeedHex: row.egg.lcgSeedHex,
      mtSeedHex: row.egg.mtSeedHex,
      pidHex: formatPidHex(row.egg.pid),
      pidDec: row.egg.pid >>> 0,
      ability: row.egg.ability,
      abilityLabel: resolveAbilityLabel(row.egg.ability, locale),
      gender: row.egg.gender,
      genderLabel: resolveGenderLabel(row.egg.gender, locale),
      nature: row.egg.nature,
      natureName: natureName(row.egg.nature, locale),
      shiny: row.egg.shiny,
      shinyLabel: resolveShinyLabel(row.egg.shiny, locale),
      ivs: row.egg.ivs,
      hiddenPower: row.egg.hiddenPower,
      hiddenPowerLabel: formatHiddenPower(row.egg.hiddenPower, locale),
      isStable: row.isStable,
    };

    // Add boot timing fields if available
    if (isBootTimingMode) {
      if (row.timer0 !== undefined) {
        adapted.timer0Hex = formatTimer0Hex(row.timer0);
      }
      if (row.vcount !== undefined) {
        adapted.vcountHex = formatVcountHex(row.vcount);
      }
      if (row.bootTimestampIso) {
        adapted.bootTimestampIso = row.bootTimestampIso;
      }
    }

    return adapted;
  });
}

export function exportEggGenerationResults(
  results: EnumeratedEggDataWithBootTiming[],
  options: EggGenerationExportOptions,
  context: EggGenerationExportContext
): string {
  const adapted = adaptEggResults(results, context);
  const includeBootTiming = Boolean(options.includeBootTiming) && context.isBootTimingMode;

  switch (options.format) {
    case 'csv':
      return formatCsvExport(adapted, includeBootTiming);
    case 'json':
      return formatJsonExport(adapted, includeBootTiming, context);
    case 'txt':
      return formatTxtExport(adapted, includeBootTiming, context);
    default:
      throw new Error(`Unsupported format: ${options.format}`);
  }
}

const CSV_HEADERS_BASE = [
  'Advance',
  'Direction',
  'DirectionValue',
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
  'LCGSeed',
  'MTSeed',
  'Stable',
];

const CSV_HEADERS_BOOT_TIMING = ['Timer0', 'VCount', 'BootTimestamp'];

function formatCsvExport(adapted: AdaptedEggResult[], includeBootTiming: boolean): string {
  const headers = includeBootTiming
    ? [...CSV_HEADERS_BASE, ...CSV_HEADERS_BOOT_TIMING]
    : CSV_HEADERS_BASE;

  const lines = [headers.join(',')];

  for (const row of adapted) {
    const baseRow = [
      String(row.advance),
      row.directionArrow,
      row.directionValue >= 0 ? String(row.directionValue) : '',
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
      row.lcgSeedHex,
      row.mtSeedHex,
      row.isStable ? 'Yes' : 'No',
    ];

    if (includeBootTiming) {
      baseRow.push(
        row.timer0Hex ?? '',
        row.vcountHex ?? '',
        row.bootTimestampIso ?? ''
      );
    }

    lines.push(baseRow.join(','));
  }

  return lines.join('\n');
}

function formatJsonExport(
  adapted: AdaptedEggResult[],
  includeBootTiming: boolean,
  context: EggGenerationExportContext
): string {
  const data = {
    exportDate: new Date().toISOString(),
    format: 'egg-generation-v1',
    locale: context.locale,
    totalResults: adapted.length,
    isBootTimingMode: context.isBootTimingMode,
    results: adapted.map((row) => {
      const base: Record<string, unknown> = {
        advance: row.advance,
        direction: row.directionArrow,
        directionValue: row.directionValue >= 0 ? row.directionValue : null,
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
        lcgSeed: row.lcgSeedHex,
        mtSeed: row.mtSeedHex,
        isStable: row.isStable,
      };

      if (includeBootTiming) {
        base.timer0 = row.timer0Hex ?? null;
        base.vcount = row.vcountHex ?? null;
        base.bootTimestamp = row.bootTimestampIso ?? null;
      }

      return base;
    }),
  };

  return JSON.stringify(data, null, 2);
}

function formatTxtExport(
  adapted: AdaptedEggResult[],
  includeBootTiming: boolean,
  context: EggGenerationExportContext
): string {
  const lines: string[] = [];
  const isJa = context.locale === 'ja';

  lines.push(isJa ? 'タマゴ生成結果 エクスポート' : 'Egg Generation Results Export');
  lines.push(`Export Date: ${new Date().toISOString()}`);
  lines.push(`Total Results: ${adapted.length}`);
  lines.push('');

  for (let i = 0; i < adapted.length; i++) {
    const row = adapted[i];
    lines.push(`Result #${i + 1}`);
    lines.push(`  Advance: ${row.advance}`);
    if (row.directionValue >= 0) {
      lines.push(`  Direction: ${row.directionArrow} (${row.directionValue})`);
    }
    lines.push(`  Ability: ${row.abilityLabel}`);
    lines.push(`  Gender: ${row.genderLabel}`);
    lines.push(`  Nature: ${row.natureName}`);
    lines.push(`  Shiny: ${row.shinyLabel}`);
    lines.push(`  IVs: ${row.ivs.map(formatIv).join('-')}`);
    lines.push(`  Hidden Power: ${row.hiddenPowerLabel}`);
    lines.push(`  PID: ${row.pidHex}`);
    lines.push(`  LCG Seed: ${row.lcgSeedHex}`);
    lines.push(`  MT Seed: ${row.mtSeedHex}`);
    lines.push(`  Stable: ${row.isStable ? 'Yes' : 'No'}`);

    if (includeBootTiming) {
      if (row.timer0Hex) lines.push(`  Timer0: ${row.timer0Hex}`);
      if (row.vcountHex) lines.push(`  VCount: ${row.vcountHex}`);
      if (row.bootTimestampIso) lines.push(`  Boot Timestamp: ${row.bootTimestampIso}`);
    }

    lines.push('');
  }

  return lines.join('\n');
}
