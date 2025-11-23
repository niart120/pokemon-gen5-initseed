import type { GenerationResult, SeedSourceMode } from '@/types/generation';
import {
  pidHex,
  seedHex,
  natureName,
  shinyLabel,
  calculateNeedleDirection,
  needleDirectionArrow,
} from '@/lib/utils/format-display';
import { resolveBatch, toUiReadyPokemon, type ResolutionContext } from '@/lib/generation/pokemon-resolver';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import { formatResultDateTime, formatKeyInputDisplay } from '@/lib/i18n/strings/search-results';
import type { SupportedLocale } from '@/types/i18n';

export interface GenerationExportOptions {
  format: 'csv' | 'json' | 'txt';
  includeAdvancedFields?: boolean;
}

interface AdaptedGenerationResult {
  advance: number;
  seedHex: string;
  seedDec: string; // BigInt -> string 保持 (安全)
  pidHex: string;
  pidDec: number;
  seedSourceMode?: SeedSourceMode;
  derivedSeedIndex?: number;
  seedSourceSeedHex?: string;
  timer0Hex?: string;
  timer0Value?: number;
  vcountHex?: string;
  vcountValue?: number;
  bootTimestampIso?: string;
  bootTimestampDisplay?: string;
  keyInputDisplay?: string;
  macAddress?: string;
  natureId: number;
  natureName: string;
  shinyType: number;
  shinyLabel: string;
  needleDirectionValue: number;
  needleDirectionArrow: string;
  abilitySlot: number;
  encounterType: number;
  encounterSlotValue: number;
  syncApplied: boolean;
  genderValue: number;
  levelRandHex: string;
  levelRandDec: string; // BigInt -> string
  // --- resolved additions (optional if context provided) ---
  speciesName?: string;
  abilityName?: string;
  genderResolved?: string; // 'M','F','-','?'
  level?: number;
  stats?: {
    hp?: number;
    attack?: number;
    defense?: number;
    specialAttack?: number;
    specialDefense?: number;
    speed?: number;
  };
}

interface BootTimingMetaSeedEntry {
  derivedSeedIndex: number | null;
  timer0Hex: string | null;
  vcountHex: string | null;
  seedHex: string | null;
}

interface BootTimingMeta {
  bootTimestamp: string | null;
  keyInput: string | null;
  macAddress: string | null;
  seeds: BootTimingMetaSeedEntry[];
}

// 既存toHex系は format-display へ統合。seedは16桁幅固定表示。
function toHexBigInt(v: bigint): string { return '0x' + seedHex(v).toLowerCase(); }
function toHex32(v: number): string { return '0x' + pidHex(v).toLowerCase(); }

function formatMacAddress(bytes: readonly number[] | undefined): string | undefined {
  if (!bytes || bytes.length < 6) {
    return undefined;
  }
  return bytes
    .slice(0, 6)
    .map((b) => b.toString(16).toUpperCase().padStart(2, '0'))
    .join(':');
}

export function adaptGenerationResults(results: GenerationResult[], opts?: {
  encounterTable?: EncounterTable;
  genderRatios?: Map<number, GenderRatio>;
  abilityCatalog?: Map<number, string[]>;
  locale?: SupportedLocale;
  version?: 'B' | 'W' | 'B2' | 'W2';
  baseSeed?: bigint;
}): AdaptedGenerationResult[] {
  const {
    encounterTable,
    genderRatios,
    abilityCatalog,
    locale = 'ja',
    version = 'B',
    baseSeed,
  } = opts || {};
  let resolvedUi: ReturnType<typeof toUiReadyPokemon>[] | null = null;
  if (encounterTable || genderRatios || abilityCatalog) {
    try {
      const context: ResolutionContext = { encounterTable, genderRatios, abilityCatalog };
      const resolved = resolveBatch(results, context);
      resolvedUi = resolved.map(r => toUiReadyPokemon(r, { locale, version, baseSeed }));
    } catch { /* fail soft; keep legacy fields */ }
  }
  return results.map((r, idx) => {
    let needleValue = -1;
    let needleArrow = '?';
    try {
      needleValue = calculateNeedleDirection(r.seed);
      needleArrow = needleDirectionArrow(needleValue);
    } catch {
      needleValue = -1;
      needleArrow = '?';
    }
    const uiEntry = resolvedUi?.[idx];
    const stats = uiEntry?.stats;
    let timer0Hex: string | undefined;
    let timer0Value: number | undefined;
    if (typeof r.timer0 === 'number') {
      timer0Value = r.timer0 >>> 0;
      timer0Hex = '0x' + timer0Value.toString(16).toUpperCase().padStart(4, '0');
    }
    let vcountHex: string | undefined;
    let vcountValue: number | undefined;
    if (typeof r.vcount === 'number') {
      vcountValue = r.vcount >>> 0;
      vcountHex = '0x' + vcountValue.toString(16).toUpperCase().padStart(2, '0');
    }
    let bootTimestampDisplay: string | undefined;
    if (r.bootTimestampIso) {
      const dt = new Date(r.bootTimestampIso);
      if (!Number.isNaN(dt.getTime())) {
        bootTimestampDisplay = formatResultDateTime(dt, locale);
      }
    }
    const keyInputDisplay = r.keyInputNames && r.keyInputNames.length
      ? formatKeyInputDisplay(r.keyInputNames, locale)
      : undefined;
    const macAddressDisplay = formatMacAddress(r.macAddress);
    return {
    advance: r.advance,
    seedHex: toHexBigInt(r.seed),
    seedDec: r.seed.toString(),
    pidHex: toHex32(r.pid >>> 0),
    pidDec: r.pid >>> 0,
    seedSourceMode: r.seedSourceMode,
    derivedSeedIndex: r.derivedSeedIndex,
    seedSourceSeedHex: r.seedSourceSeedHex,
    timer0Hex,
    timer0Value,
    vcountHex,
    vcountValue,
    bootTimestampIso: r.bootTimestampIso,
    bootTimestampDisplay,
    keyInputDisplay,
    macAddress: macAddressDisplay,
    natureId: r.nature,
    natureName: natureName(r.nature),
    shinyType: r.shiny_type,
    shinyLabel: shinyLabel(r.shiny_type),
      needleDirectionValue: needleValue,
      needleDirectionArrow: needleArrow,
    abilitySlot: r.ability_slot,
    encounterType: r.encounter_type,
    encounterSlotValue: r.encounter_slot_value,
    syncApplied: r.sync_applied,
    genderValue: r.gender_value,
    levelRandHex: toHexBigInt(r.level_rand_value),
    levelRandDec: r.level_rand_value.toString(),
      speciesName: uiEntry?.speciesName,
      abilityName: uiEntry?.abilityName,
      genderResolved: uiEntry?.gender,
      level: uiEntry?.level,
      stats: stats
        ? {
            hp: stats.hp,
            attack: stats.attack,
            defense: stats.defense,
            specialAttack: stats.specialAttack,
            specialDefense: stats.specialDefense,
            speed: stats.speed,
          }
        : undefined,
    } satisfies AdaptedGenerationResult;
  });
}

export type GenerationExportContext = Parameters<typeof adaptGenerationResults>[1];

function buildBootTimingMeta(adapted: AdaptedGenerationResult[]): BootTimingMeta | null {
  const bootRows = adapted.filter(row => row.seedSourceMode === 'boot-timing');
  if (!bootRows.length) {
    return null;
  }
  const seedMap = new Map<string, BootTimingMetaSeedEntry>();
  for (const row of bootRows) {
    const key = row.derivedSeedIndex != null
      ? `idx-${row.derivedSeedIndex}`
      : `seed-${row.seedHex}-${row.timer0Hex ?? ''}-${row.vcountHex ?? ''}`;
    if (!seedMap.has(key)) {
      seedMap.set(key, {
        derivedSeedIndex: row.derivedSeedIndex ?? null,
        timer0Hex: row.timer0Hex ?? null,
        vcountHex: row.vcountHex ?? null,
        seedHex: row.seedSourceSeedHex ?? row.seedHex,
      });
    }
  }
  return {
    bootTimestamp: bootRows[0]?.bootTimestampIso ?? null,
    keyInput: bootRows[0]?.keyInputDisplay ?? null,
    macAddress: bootRows[0]?.macAddress ?? null,
    seeds: Array.from(seedMap.values()),
  };
}

export function exportGenerationResults(
  results: GenerationResult[],
  options: GenerationExportOptions,
  context?: GenerationExportContext,
): string {
  switch (options.format) {
    case 'csv':
      return exportCsv(results, options, context);
    case 'json':
      return exportJson(results, options, context);
    case 'txt':
      return exportTxt(results, options, context);
    default:
      throw new Error('Unsupported format');
  }
}

const CSV_HEADERS = [
  'Advance',
  'NeedleDirection',
  'NeedleDirectionValue',
  'SpeciesName',
  'AbilityName',
  'Gender',
  'NatureName',
  'ShinyLabel',
  'Level',
  'HP',
  'Attack',
  'Defense',
  'SpecialAttack',
  'SpecialDefense',
  'Speed',
  'SeedHex',
  'PIDHex',
  'Timer0Hex',
  'VCountHex',
  'BootTimestamp',
  'KeyInput',
  'SeedSourceMode',
  'DerivedSeedIndex',
  'SeedSourceSeedHex',
  'MacAddress',
  'SeedDec',
  'PIDDec',
  'NatureId',
  'ShinyType',
  'AbilitySlot',
  'EncounterType',
  'EncounterSlotValue',
  'SyncApplied',
  'GenderValue',
  'LevelRandHex',
  'LevelRandDec',
];

const DISPLAY_COLUMN_COUNT = 25;

function exportCsv(
  results: GenerationResult[],
  options: GenerationExportOptions,
  ctx?: GenerationExportContext,
): string {
  const adapted = adaptGenerationResults(results, ctx);
  const includeAdvanced = Boolean(options.includeAdvancedFields);
  const headers = includeAdvanced
    ? CSV_HEADERS
    : CSV_HEADERS.slice(0, DISPLAY_COLUMN_COUNT);
  const lines = [headers.join(',')];
  for (const a of adapted) {
    const stats = a.stats;
      const baseRow = [
      String(a.advance),
      a.needleDirectionArrow,
      a.needleDirectionValue >= 0 ? String(a.needleDirectionValue) : '',
      a.speciesName ?? '',
      a.abilityName ?? '',
      a.genderResolved ?? '',
      a.natureName,
      a.shinyLabel,
      a.level != null ? String(a.level) : '',
      stats?.hp != null ? String(stats.hp) : '',
      stats?.attack != null ? String(stats.attack) : '',
      stats?.defense != null ? String(stats.defense) : '',
      stats?.specialAttack != null ? String(stats.specialAttack) : '',
      stats?.specialDefense != null ? String(stats.specialDefense) : '',
      stats?.speed != null ? String(stats.speed) : '',
      a.seedHex,
      a.pidHex,
      a.timer0Hex ?? '',
      a.vcountHex ?? '',
      a.bootTimestampDisplay ?? a.bootTimestampIso ?? '',
      a.keyInputDisplay ?? '',
      a.seedSourceMode ?? '',
      a.derivedSeedIndex != null ? String(a.derivedSeedIndex) : '',
      a.seedSourceSeedHex ?? '',
      a.macAddress ?? '',
    ];
    if (includeAdvanced) {
      baseRow.push(
        a.seedDec,
        String(a.pidDec),
        String(a.natureId),
        String(a.shinyType),
        String(a.abilitySlot),
        String(a.encounterType),
        String(a.encounterSlotValue),
        String(a.syncApplied),
        String(a.genderValue),
        a.levelRandHex,
        a.levelRandDec,
      );
    }
    lines.push(baseRow.join(','));
  }
  return lines.join('\n');
}

function exportJson(
  results: GenerationResult[],
  options: GenerationExportOptions,
  ctx?: GenerationExportContext,
): string {
  const adapted = adaptGenerationResults(results, ctx);
  const includeAdvanced = Boolean(options.includeAdvancedFields);
  const bootTimingMeta = buildBootTimingMeta(adapted);
  const data = {
    exportDate: new Date().toISOString(),
    format: 'generation-v2',
    totalResults: adapted.length,
    meta: {
      seedSourceMode: bootTimingMeta ? 'boot-timing' : 'lcg',
      bootTiming: bootTimingMeta,
    },
    results: adapted.map((a) => {
      const base: Record<string, unknown> = {
        advance: a.advance,
        needleDirection: a.needleDirectionArrow,
        needleDirectionValue: a.needleDirectionValue >= 0 ? a.needleDirectionValue : null,
        seedHex: a.seedHex,
        pidHex: a.pidHex,
        speciesName: a.speciesName ?? null,
        abilityName: a.abilityName ?? null,
        gender: a.genderResolved ?? null,
        natureName: a.natureName,
        shinyLabel: a.shinyLabel,
        level: a.level ?? null,
        stats: a.stats ?? null,
      };
      if (includeAdvanced) {
        base.seedDec = a.seedDec;
        base.pidDec = a.pidDec;
        base.natureId = a.natureId;
        base.shinyType = a.shinyType;
        base.abilitySlot = a.abilitySlot;
        base.encounterType = a.encounterType;
        base.encounterSlotValue = a.encounterSlotValue;
        base.syncApplied = a.syncApplied;
        base.genderValue = a.genderValue;
        base.levelRandHex = a.levelRandHex;
        base.levelRandDec = a.levelRandDec;
      }
      base.bootTimestamp = a.bootTimestampDisplay ?? a.bootTimestampIso ?? null;
      base.timer0Hex = a.timer0Hex ?? null;
      base.vcountHex = a.vcountHex ?? null;
      base.keyInput = a.keyInputDisplay ?? null;
      base.seedSourceMode = a.seedSourceMode ?? null;
      base.derivedSeedIndex = a.derivedSeedIndex ?? null;
      base.seedSourceSeedHex = a.seedSourceSeedHex ?? null;
      base.macAddress = a.macAddress ?? null;
      return base;
    }),
  };
  return JSON.stringify(data, null, 2);
}

function exportTxt(
  results: GenerationResult[],
  options: GenerationExportOptions,
  ctx?: GenerationExportContext,
): string {
  const adapted = adaptGenerationResults(results, ctx);
  const includeAdvanced = Boolean(options.includeAdvancedFields);
  const out: string[] = [];
  out.push('Generation Results Export (v2)');
  out.push(`Export Date: ${new Date().toISOString()}`);
  out.push(`Total Results: ${adapted.length}`);
  out.push('');
  adapted.forEach((a, idx) => {
    out.push(`Result #${idx+1}`);
    out.push(`  Advance: ${a.advance}`);
    if (a.needleDirectionValue >= 0) {
      out.push(`  NeedleDirection: ${a.needleDirectionArrow} (${a.needleDirectionValue})`);
    }
    out.push(`  Seed: ${a.seedHex}${includeAdvanced ? ` (${a.seedDec})` : ''}`);
    out.push(`  PID: ${a.pidHex}${includeAdvanced ? ` (${a.pidDec})` : ''}`);
    out.push(`  NatureName: ${a.natureName}`);
    if (includeAdvanced) {
      out.push(`  NatureId: ${a.natureId}`);
    }
    out.push(`  Shiny: ${a.shinyLabel}${includeAdvanced ? ` (${a.shinyType})` : ''}`);
    if (a.bootTimestampDisplay || a.bootTimestampIso) {
      out.push(`  BootTime: ${a.bootTimestampDisplay ?? a.bootTimestampIso}`);
    }
    if (a.timer0Hex) {
      out.push(`  Timer0: ${a.timer0Hex}${includeAdvanced && a.timer0Value != null ? ` (${a.timer0Value})` : ''}`);
    }
    if (a.vcountHex) {
      out.push(`  VCount: ${a.vcountHex}${includeAdvanced && a.vcountValue != null ? ` (${a.vcountValue})` : ''}`);
    }
    if (a.keyInputDisplay) {
      out.push(`  KeyInput: ${a.keyInputDisplay}`);
    }
    if (a.seedSourceMode) {
      out.push(`  SeedSourceMode: ${a.seedSourceMode}`);
    }
    if (a.derivedSeedIndex != null) {
      out.push(`  DerivedSeedIndex: ${a.derivedSeedIndex}`);
    }
    if (a.seedSourceSeedHex) {
      out.push(`  SeedSourceSeedHex: ${a.seedSourceSeedHex}`);
    }
    if (a.macAddress) {
      out.push(`  MacAddress: ${a.macAddress}`);
    }
    if (includeAdvanced) {
      out.push(`  AbilitySlot: ${a.abilitySlot}`);
      out.push(`  Encounter: type=${a.encounterType} slotVal=${a.encounterSlotValue}`);
      out.push(`  SyncApplied: ${a.syncApplied}`);
      out.push(`  GenderValue: ${a.genderValue}`);
      out.push(`  LevelRand: ${a.levelRandHex} (${a.levelRandDec})`);
    }
    if (a.speciesName) out.push(`  Species: ${a.speciesName}`);
    if (a.abilityName) out.push(`  Ability: ${a.abilityName}`);
    if (a.genderResolved) out.push(`  GenderResolved: ${a.genderResolved}`);
    if (a.level != null) out.push(`  Level: ${a.level}`);
    if (a.stats) {
      const { hp, attack, defense, specialAttack, specialDefense, speed } = a.stats;
      const statEntries: Array<[string, number | undefined]> = [
        ['HP', hp],
        ['Attack', attack],
        ['Defense', defense],
        ['SpecialAttack', specialAttack],
        ['SpecialDefense', specialDefense],
        ['Speed', speed],
      ];
      statEntries.forEach(([label, value]) => {
        if (value != null) {
          out.push(`  ${label}: ${value}`);
        }
      });
    }
    out.push('');
  });
  return out.join('\n');
}
