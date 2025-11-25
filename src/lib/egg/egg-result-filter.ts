/**
 * Egg Result Filter
 * 
 * Client-side filtering for Boot-Timing mode results.
 * Timer0/VCount filters are only applicable in boot-timing mode.
 * 
 * Based on: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md §10.8.2
 */

import type {
  EnumeratedEggDataWithBootTiming,
  EggSeedSourceMode,
} from '@/types/egg';

// === Filter Types ===

export interface EggBootTimingFilters {
  timer0Filter?: string;  // 16進数文字列（例: "10A0"）
  vcountFilter?: string;  // 16進数文字列（例: "5C"）
}

// === Filter Functions ===

/**
 * Boot-Timing 専用フィルターを適用
 * LCGモードでは何もフィルタしない
 */
export function applyBootTimingFilters(
  results: EnumeratedEggDataWithBootTiming[],
  filters: EggBootTimingFilters,
  seedSourceMode: EggSeedSourceMode,
): EnumeratedEggDataWithBootTiming[] {
  // LCGモードではフィルタ不要
  if (seedSourceMode !== 'boot-timing') {
    return results;
  }

  const hasTimer0Filter = Boolean(filters.timer0Filter?.trim());
  const hasVcountFilter = Boolean(filters.vcountFilter?.trim());

  // フィルタ指定なしの場合はそのまま返す
  if (!hasTimer0Filter && !hasVcountFilter) {
    return results;
  }

  return results.filter(result => {
    if (hasTimer0Filter && !matchesTimer0(result.timer0, filters.timer0Filter!)) {
      return false;
    }
    if (hasVcountFilter && !matchesVcount(result.vcount, filters.vcountFilter!)) {
      return false;
    }
    return true;
  });
}

/**
 * Timer0 値が16進数フィルタにマッチするか
 */
function matchesTimer0(timer0: number | undefined, filter: string): boolean {
  if (timer0 === undefined) return false;
  const filterValue = parseInt(filter.trim(), 16);
  if (Number.isNaN(filterValue)) return true; // 無効な入力は全てマッチ
  return timer0 === filterValue;
}

/**
 * VCount 値が16進数フィルタにマッチするか
 */
function matchesVcount(vcount: number | undefined, filter: string): boolean {
  if (vcount === undefined) return false;
  const filterValue = parseInt(filter.trim(), 16);
  if (Number.isNaN(filterValue)) return true; // 無効な入力は全てマッチ
  return vcount === filterValue;
}

/**
 * フィルタ入力のバリデーション
 * 空文字または有効な16進数の場合はtrue
 */
export function isValidHexFilter(value: string): boolean {
  const trimmed = value.trim();
  if (trimmed === '') return true;
  return /^[0-9a-fA-F]+$/.test(trimmed);
}

/**
 * Timer0/VCount フィルタの状態を表示用文字列に変換
 */
export function formatBootTimingFilterStatus(filters: EggBootTimingFilters): string {
  const parts: string[] = [];
  if (filters.timer0Filter?.trim()) {
    parts.push(`Timer0: 0x${filters.timer0Filter.trim().toUpperCase()}`);
  }
  if (filters.vcountFilter?.trim()) {
    parts.push(`VCount: 0x${filters.vcountFilter.trim().toUpperCase()}`);
  }
  return parts.length > 0 ? parts.join(', ') : '';
}
