import type { GenerationResult } from '@/types/generation';

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
  shinyType: number;
  shinyLabel: string;
  abilitySlot: number;
  encounterType: number;
  encounterSlotValue: number;
  syncApplied: boolean;
  genderValue: number;
  levelRandHex: string;
  levelRandDec: string; // BigInt -> string
}

const SHINY_LABEL: Record<number,string> = {
  0: 'None',
  1: 'Square',
  2: 'Star',
};

function toHexBigInt(v: bigint): string {
  // 先頭0埋め幅は確定仕様未定のためそのまま。ただし空防止
  return '0x' + v.toString(16);
}

function toHex32(v: number): string { return '0x' + v.toString(16).padStart(8,'0'); }

export function adaptGenerationResults(results: GenerationResult[]): AdaptedGenerationResult[] {
  return results.map(r => ({
    advance: r.advance,
    seedHex: toHexBigInt(r.seed),
    seedDec: r.seed.toString(),
    pidHex: toHex32(r.pid >>> 0),
    pidDec: r.pid >>> 0,
    natureId: r.nature,
    shinyType: r.shiny_type,
    shinyLabel: SHINY_LABEL[r.shiny_type] ?? String(r.shiny_type),
    abilitySlot: r.ability_slot,
    encounterType: r.encounter_type,
    encounterSlotValue: r.encounter_slot_value,
    syncApplied: r.sync_applied,
    genderValue: r.gender_value,
    levelRandHex: toHexBigInt(r.level_rand_value),
    levelRandDec: r.level_rand_value.toString(),
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
  'Advance','SeedHex','SeedDec','PIDHex','PIDDec','NatureId','ShinyType','ShinyLabel','AbilitySlot','EncounterType','EncounterSlotValue','SyncApplied','GenderValue','LevelRandHex','LevelRandDec'
];

function exportCsv(results: GenerationResult[]): string {
  const adapted = adaptGenerationResults(results);
  const lines = [CSV_HEADERS.join(',')];
  for (const a of adapted) {
    lines.push([
      a.advance,
      a.seedHex,
      a.seedDec,
      a.pidHex,
      a.pidDec,
      a.natureId,
      a.shinyType,
      a.shinyLabel,
      a.abilitySlot,
      a.encounterType,
      a.encounterSlotValue,
      a.syncApplied,
      a.genderValue,
      a.levelRandHex,
      a.levelRandDec,
    ].join(','));
  }
  return lines.join('\n');
}

function exportJson(results: GenerationResult[]): string {
  const adapted = adaptGenerationResults(results);
  const data = {
    exportDate: new Date().toISOString(),
    format: 'generation',
    totalResults: adapted.length,
    results: adapted,
  };
  return JSON.stringify(data, null, 2);
}

function exportTxt(results: GenerationResult[]): string {
  const adapted = adaptGenerationResults(results);
  const out: string[] = [];
  out.push('Generation Results Export');
  out.push(`Export Date: ${new Date().toISOString()}`);
  out.push(`Total Results: ${adapted.length}`);
  out.push('');
  adapted.forEach((a, idx) => {
    out.push(`Result #${idx+1}`);
    out.push(`  Advance: ${a.advance}`);
    out.push(`  Seed: ${a.seedHex} (${a.seedDec})`);
    out.push(`  PID: ${a.pidHex} (${a.pidDec})`);
    out.push(`  NatureId: ${a.natureId}`);
    out.push(`  Shiny: ${a.shinyLabel} (${a.shinyType})`);
    out.push(`  AbilitySlot: ${a.abilitySlot}`);
    out.push(`  Encounter: type=${a.encounterType} slotVal=${a.encounterSlotValue}`);
    out.push(`  SyncApplied: ${a.syncApplied}`);
    out.push(`  GenderValue: ${a.genderValue}`);
    out.push(`  LevelRand: ${a.levelRandHex} (${a.levelRandDec})`);
    out.push('');
  });
  return out.join('\n');
}
