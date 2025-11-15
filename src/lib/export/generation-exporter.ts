import type { GenerationResult } from '@/types/generation';
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

// 既存toHex系は format-display へ統合。seedは16桁幅固定表示。
function toHexBigInt(v: bigint): string { return '0x' + seedHex(v).toLowerCase(); }
function toHex32(v: number): string { return '0x' + pidHex(v).toLowerCase(); }

export function adaptGenerationResults(results: GenerationResult[], opts?: {
  encounterTable?: EncounterTable;
  genderRatios?: Map<number, GenderRatio>;
  abilityCatalog?: Map<number, string[]>;
  locale?: 'ja' | 'en';
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
    return {
    advance: r.advance,
    seedHex: toHexBigInt(r.seed),
    seedDec: r.seed.toString(),
    pidHex: toHex32(r.pid >>> 0),
    pidDec: r.pid >>> 0,
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

const DISPLAY_COLUMN_COUNT = 17;

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
  const data = {
    exportDate: new Date().toISOString(),
    format: 'generation-v2',
    totalResults: adapted.length,
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
