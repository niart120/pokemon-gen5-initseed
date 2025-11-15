/**
 * Pokemon Black/White and Black2/White2 encounter tables
 * 
 * Data sources and retrieval dates:
 * - ポケモンの友 (BW/BW2 エンカウントテーブル):
 *   - B: https://pokebook.jp/data/sp5/enc_b (Retrieved: 2025-08-10)
 *   - W: https://pokebook.jp/data/sp5/enc_w (Retrieved: 2025-08-10)
 *   - B2: https://pokebook.jp/data/sp5/enc_b2 (Retrieved: 2025-08-10)
 *   - W2: https://pokebook.jp/data/sp5/enc_w2 (Retrieved: 2025-08-10)
 * - Bulbapedia / Serebii (補助参照)
 */

import type { DomainEncounterType as EncounterType } from '../types/domain';
import type { ROMVersion } from '../types/rom';
import { ensureEncounterRegistryLoaded, getEncounterFromRegistry } from './encounters/loader';

/**
 * Single encounter slot data
 */
export interface EncounterSlot {
  /** Pokemon species national dex number */
  speciesId: number;
  /** Encounter rate percentage */
  rate: number;
  /** Level range */
  levelRange: {
    min: number;
    max: number;
  };
}

/**
 * Encounter table for a specific location and method
 */
export interface EncounterTable {
  /** Location name */
  location: string;
  /** Encounter method */
  method: EncounterType;
  /** Game version */
  version: ROMVersion;
  /** Array of encounter slots (12 slots for normal encounters) */
  slots: EncounterSlot[];
}

// Key生成関数は不要になったため削除（ローダ側に集約）

/**
 * Look up encounter table
 * 
 * @param version Game version
 * @param location Location name
 * @param method Encounter method
 * @returns Encounter table or null if not found
 */
export function getEncounterTable(
  version: ROMVersion,
  location: string,
  method: EncounterType
): EncounterTable | null {
  ensureEncounterRegistryLoaded();
  const hit = getEncounterFromRegistry(version, location, method);
  if (!hit) return null;
  return { location, method, version, slots: hit.slots };
}

/**
 * Get Pokemon species from encounter slot
 * 
 * @param table Encounter table
 * @param slotValue Encounter slot value (0-11 for normal encounters)
 * @returns Encounter slot data
 */
export function getEncounterSlot(
  table: EncounterTable,
  slotValue: number
): EncounterSlot {
  if (slotValue < 0 || slotValue >= table.slots.length) {
    throw new Error(
      `Invalid encounter slot ${slotValue} for table with ${table.slots.length} slots`
    );
  }
  return table.slots[slotValue];
}
// デフォルトテーブル/検証スタブは不要になったため削除