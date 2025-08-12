import { describe, it, expect } from 'vitest';
import { exportGenerationResults, adaptGenerationResults } from '@/lib/export/generation-exporter';
import type { GenerationResult } from '@/types/generation';

function makeDummy(i: number): GenerationResult {
  return {
    advance: i + 1,
    seed: BigInt(1000 + i),
    pid: 0x12340000 + i,
    nature: i % 25,
    sync_applied: (i % 2) === 0,
    ability_slot: i % 2,
    gender_value: (i * 13) % 256,
    encounter_slot_value: i % 10,
    encounter_type: i % 5,
    level_rand_value: BigInt(5000 + i),
    shiny_type: i % 3,
  };
}

describe('generation-exporter', () => {
  const samples: GenerationResult[] = Array.from({ length: 10 }, (_, i) => makeDummy(i));

  it('adaptGenerationResults converts fields', () => {
    const adapted = adaptGenerationResults(samples);
    expect(adapted.length).toBe(10);
    expect(adapted[0].seedHex.startsWith('0x')).toBe(true);
    expect(adapted[0].pidHex).toMatch(/^0x[0-9a-f]{8}$/);
  });

  it('exports CSV with header + rows', () => {
    const csv = exportGenerationResults(samples, { format: 'csv' });
    const lines = csv.split('\n');
    expect(lines[0].split(',')[0]).toBe('Advance');
    expect(lines.length).toBe(11); // header + 10 rows
  });

  it('exports JSON with totalResults=10', () => {
    const json = exportGenerationResults(samples, { format: 'json' });
    const obj = JSON.parse(json);
    expect(obj.totalResults).toBe(10);
    expect(obj.results[0].seedHex || obj.results[0].seedHex === undefined).toBeTruthy();
  });

  it('exports TXT containing Result #1', () => {
    const txt = exportGenerationResults(samples, { format: 'txt' });
    expect(txt).toContain('Result #1');
    expect(txt).toContain('Total Results: 10');
  });

  it('handles empty array', () => {
    const csv = exportGenerationResults([], { format: 'csv' });
    expect(csv.split('\n').length).toBe(1);
    const json = exportGenerationResults([], { format: 'json' });
    const obj = JSON.parse(json); expect(obj.totalResults).toBe(0);
    const txt = exportGenerationResults([], { format: 'txt' });
    expect(txt).toContain('Total Results: 0');
  });
});
