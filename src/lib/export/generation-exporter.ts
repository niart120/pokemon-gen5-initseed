import type { GenerationResult } from '@/types/generation';
import { pidHex, seedHex, natureName, shinyLabel } from '@/lib/utils/format-display';
import { resolveBatch, toUiReadyPokemon, type ResolutionContext } from '@/lib/generation/pokemon-resolver';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';

export interface GenerationExportOptions {
  format: 'csv' | 'json' | 'txt';
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
}

// 既存toHex系は format-display へ統合。seedは16桁幅固定表示。
function toHexBigInt(v: bigint): string { return '0x' + seedHex(v).toLowerCase(); }
function toHex32(v: number): string { return '0x' + pidHex(v).toLowerCase(); }

export function adaptGenerationResults(results: GenerationResult[], opts?: {
  encounterTable?: EncounterTable;
  genderRatios?: Map<number, GenderRatio>;
  abilityCatalog?: Map<number, string[]>;
  locale?: 'ja' | 'en';
}): AdaptedGenerationResult[] {
  const { encounterTable, genderRatios, abilityCatalog, locale = 'ja' } = opts || {};
  let resolvedUi: ReturnType<typeof toUiReadyPokemon>[] | null = null;
  if (encounterTable || genderRatios || abilityCatalog) {
    try {
      const context: ResolutionContext = { encounterTable, genderRatios, abilityCatalog };
      const resolved = resolveBatch(results, context);
      resolvedUi = resolved.map(r => toUiReadyPokemon(r, { locale }));
    } catch { /* fail soft; keep legacy fields */ }
  }
  return results.map((r, idx) => ({
    advance: r.advance,
    seedHex: toHexBigInt(r.seed),
    seedDec: r.seed.toString(),
    pidHex: toHex32(r.pid >>> 0),
    pidDec: r.pid >>> 0,
    natureId: r.nature,
    natureName: natureName(r.nature),
    shinyType: r.shiny_type,
    shinyLabel: shinyLabel(r.shiny_type),
    abilitySlot: r.ability_slot,
    encounterType: r.encounter_type,
    encounterSlotValue: r.encounter_slot_value,
    syncApplied: r.sync_applied,
    genderValue: r.gender_value,
    levelRandHex: toHexBigInt(r.level_rand_value),
    levelRandDec: r.level_rand_value.toString(),
    speciesName: resolvedUi?.[idx]?.speciesName,
    abilityName: resolvedUi?.[idx]?.abilityName,
    genderResolved: resolvedUi?.[idx]?.gender,
    level: resolvedUi?.[idx]?.level,
  }));
}

export function exportGenerationResults(results: GenerationResult[], options: GenerationExportOptions): string {
  switch (options.format) {
    case 'csv':
      return exportCsv(results);
    case 'json':
      return exportJson(results);
    case 'txt':
      return exportTxt(results);
    default:
      throw new Error('Unsupported format');
  }
}

const CSV_HEADERS = [
  'Advance','SeedHex','SeedDec','PIDHex','PIDDec','NatureId','NatureName','ShinyType','ShinyLabel','AbilitySlot','EncounterType','EncounterSlotValue','SyncApplied','GenderValue','LevelRandHex','LevelRandDec',
  'SpeciesName','AbilityName','Gender','Level'
];

function exportCsv(results: GenerationResult[], ctx?: Parameters<typeof adaptGenerationResults>[1]): string {
  const adapted = adaptGenerationResults(results, ctx);
  const lines = [CSV_HEADERS.join(',')];
  for (const a of adapted) {
    lines.push([
      a.advance,
      a.seedHex,
      a.seedDec,
      a.pidHex,
      a.pidDec,
      a.natureId,
      a.natureName,
      a.shinyType,
      a.shinyLabel,
      a.abilitySlot,
      a.encounterType,
      a.encounterSlotValue,
      a.syncApplied,
      a.genderValue,
      a.levelRandHex,
      a.levelRandDec,
      a.speciesName ?? '',
      a.abilityName ?? '',
      a.genderResolved ?? '',
      a.level ?? '',
    ].join(','));
  }
  return lines.join('\n');
}

function exportJson(results: GenerationResult[], ctx?: Parameters<typeof adaptGenerationResults>[1]): string {
  const adapted = adaptGenerationResults(results, ctx);
  const data = {
    exportDate: new Date().toISOString(),
    format: 'generation-v2',
    totalResults: adapted.length,
    results: adapted,
  };
  return JSON.stringify(data, null, 2);
}

function exportTxt(results: GenerationResult[], ctx?: Parameters<typeof adaptGenerationResults>[1]): string {
  const adapted = adaptGenerationResults(results, ctx);
  const out: string[] = [];
  out.push('Generation Results Export (v2)');
  out.push(`Export Date: ${new Date().toISOString()}`);
  out.push(`Total Results: ${adapted.length}`);
  out.push('');
  adapted.forEach((a, idx) => {
    out.push(`Result #${idx+1}`);
    out.push(`  Advance: ${a.advance}`);
    out.push(`  Seed: ${a.seedHex} (${a.seedDec})`);
    out.push(`  PID: ${a.pidHex} (${a.pidDec})`);
    out.push(`  NatureId: ${a.natureId}`);
    out.push(`  NatureName: ${a.natureName}`);
    out.push(`  Shiny: ${a.shinyLabel} (${a.shinyType})`);
    out.push(`  AbilitySlot: ${a.abilitySlot}`);
    out.push(`  Encounter: type=${a.encounterType} slotVal=${a.encounterSlotValue}`);
    out.push(`  SyncApplied: ${a.syncApplied}`);
    out.push(`  GenderValue: ${a.genderValue}`);
    out.push(`  LevelRand: ${a.levelRandHex} (${a.levelRandDec})`);
    if (a.speciesName) out.push(`  Species: ${a.speciesName}`);
    if (a.abilityName) out.push(`  Ability: ${a.abilityName}`);
    if (a.genderResolved) out.push(`  GenderResolved: ${a.genderResolved}`);
    if (a.level != null) out.push(`  Level: ${a.level}`);
    out.push('');
  });
  return out.join('\n');
}
