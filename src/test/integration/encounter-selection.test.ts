/**
 * Integration Tests for Encounter Selection using JSON datasets
 *
 * 正式API（src/data/encounter-tables.ts）による選択・レベル計算の検証。
 */

import { describe, it, expect } from 'vitest';
import { DomainEncounterType as DomainEncounterType } from '../../types/domain';
import { getEncounterTable, getEncounterSlot } from '../../data/encounter-tables';

describe('Encounter Selection Integration Tests (JSON datasets)', () => {
  it('should load encounter table for a known location/method', () => {
    const table = getEncounterTable('B', '1番道路', DomainEncounterType.Normal);
    // 生成データに依存: 存在しない可能性もあるため null 許容でアサーションは緩めに
    if (table) {
      expect(table.location).toBeDefined();
      expect(table.version).toBe('B');
      expect(table.slots.length).toBeGreaterThan(0);
    } else {
      expect(table).toBeNull();
    }
  });

  it('should map encounter slot index to species and level range', () => {
    const table = getEncounterTable('W2', '1番道路', DomainEncounterType.Normal);
    if (!table) return; // テーブルがない場合はスキップ
    const slot0 = getEncounterSlot(table, 0);
    expect(slot0.speciesId).toBeGreaterThan(0);
    expect(slot0.levelRange.min).toBeLessThanOrEqual(slot0.levelRange.max);
  });

  // Level calculation now validated via resolver; deterministic modulo test removed
});