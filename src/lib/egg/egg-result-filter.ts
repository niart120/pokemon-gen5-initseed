/**
 * Egg Result Filter
 * 
 * Client-side filtering for Boot-Timing mode results.
 * Common filtering logic for both Search(Egg) and Generation(Egg).
 * 
 * Based on: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md §10.8.2
 */

import type { ShinyFilterMode } from '@/store/generation-store';
import type {
  EnumeratedEggDataWithBootTiming,
  EggSeedSourceMode,
  StatRange,
  HiddenPowerInfo,
} from '@/types/egg';

// === Common Filter Types ===

/**
 * 共通結果フィルター条件
 * Search(Egg) と Generation(Egg) の両方で使用
 */
export interface CommonEggResultFilters {
  // 色違いフィルター
  shinyFilterMode?: ShinyFilterMode;
  // 性格フィルター (単一選択)
  nature?: number;
  // 性別フィルター
  gender?: 'male' | 'female' | 'genderless';
  // 特性フィルター
  ability?: 0 | 1 | 2;
  // めざパタイプ
  hiddenPowerType?: number;
  // めざパ威力
  hiddenPowerPower?: number;
  // IV範囲
  ivRanges?: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
  // Boot-Timing専用: Timer0フィルター (hex文字列)
  timer0Filter?: string;
  // Boot-Timing専用: VCountフィルター (hex文字列)
  vcountFilter?: string;
}

/**
 * EggBootTimingFilters は CommonEggResultFilters のエイリアス（後方互換性）
 */
export type EggBootTimingFilters = CommonEggResultFilters;

// === Filter Functions ===

/**
 * 共通フィルターを適用
 * 全フィルター項目（IV範囲含む）を完全実装
 */
export function applyCommonEggFilters(
  results: EnumeratedEggDataWithBootTiming[],
  filters: CommonEggResultFilters,
  seedSourceMode: EggSeedSourceMode,
): EnumeratedEggDataWithBootTiming[] {
  // フィルタ指定なしの場合はそのまま返す
  if (!hasAnyFilter(filters)) {
    return results;
  }

  return results.filter(result => {
    const egg = result.egg;

    // Timer0/VCount フィルタは Boot-Timing モード時のみ
    if (seedSourceMode === 'boot-timing') {
      if (filters.timer0Filter?.trim() && !matchesTimer0(result.timer0, filters.timer0Filter)) {
        return false;
      }
      if (filters.vcountFilter?.trim() && !matchesVcount(result.vcount, filters.vcountFilter)) {
        return false;
      }
    }

    // 色違いフィルタ
    if (filters.shinyFilterMode && filters.shinyFilterMode !== 'all') {
      if (!matchesShinyMode(egg.shiny, filters.shinyFilterMode)) {
        return false;
      }
    }

    // 性格フィルタ
    if (filters.nature !== undefined) {
      if (egg.nature !== filters.nature) {
        return false;
      }
    }

    // 性別フィルタ
    if (filters.gender !== undefined) {
      if (egg.gender !== filters.gender) {
        return false;
      }
    }

    // 特性フィルタ
    if (filters.ability !== undefined) {
      if (egg.ability !== filters.ability) {
        return false;
      }
    }

    // めざパタイプフィルタ
    if (filters.hiddenPowerType !== undefined) {
      if (!matchesHiddenPowerType(egg.hiddenPower, filters.hiddenPowerType)) {
        return false;
      }
    }

    // めざパ威力フィルタ
    if (filters.hiddenPowerPower !== undefined) {
      if (!matchesHiddenPowerPower(egg.hiddenPower, filters.hiddenPowerPower)) {
        return false;
      }
    }

    // IV範囲フィルタ
    if (filters.ivRanges) {
      if (!matchesIvRanges(egg.ivs, filters.ivRanges)) {
        return false;
      }
    }

    return true;
  });
}

/**
 * フィルタが1つでも設定されているか
 */
function hasAnyFilter(filters: CommonEggResultFilters): boolean {
  if (filters.shinyFilterMode && filters.shinyFilterMode !== 'all') return true;
  if (filters.nature !== undefined) return true;
  if (filters.gender !== undefined) return true;
  if (filters.ability !== undefined) return true;
  if (filters.hiddenPowerType !== undefined) return true;
  if (filters.hiddenPowerPower !== undefined) return true;
  if (filters.timer0Filter?.trim()) return true;
  if (filters.vcountFilter?.trim()) return true;
  if (filters.ivRanges && !isUnrestrictedIvRanges(filters.ivRanges)) return true;
  return false;
}

/**
 * IV範囲が制限なし (0-32) かどうか
 */
function isUnrestrictedIvRanges(ivRanges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange]): boolean {
  return ivRanges.every(range => range.min === 0 && range.max === 32);
}

/**
 * 後方互換性のため applyBootTimingFilters を維持
 */
export function applyBootTimingFilters(
  results: EnumeratedEggDataWithBootTiming[],
  filters: EggBootTimingFilters,
  seedSourceMode: EggSeedSourceMode,
): EnumeratedEggDataWithBootTiming[] {
  return applyCommonEggFilters(results, filters, seedSourceMode);
}

/**
 * 色違い値が shinyMode にマッチするか
 * shiny: 0 = 通常, 1 = ◇(square), 2 = ☆(star)
 */
function matchesShinyMode(shiny: number, mode: ShinyFilterMode): boolean {
  switch (mode) {
    case 'all':
      return true;
    case 'shiny':
      return shiny > 0; // ◇ または ☆
    case 'star':
      return shiny === 2;
    case 'square':
      return shiny === 1;
    case 'non-shiny':
      return shiny === 0;
  }
}

/**
 * めざパタイプがマッチするか
 */
function matchesHiddenPowerType(hp: HiddenPowerInfo, expectedType: number): boolean {
  if (hp.type === 'unknown') return false;
  return hp.hpType === expectedType;
}

/**
 * めざパ威力がマッチするか
 */
function matchesHiddenPowerPower(hp: HiddenPowerInfo, expectedPower: number): boolean {
  if (hp.type === 'unknown') return false;
  return hp.power === expectedPower;
}

/**
 * IV範囲がマッチするか
 */
function matchesIvRanges(
  ivs: [number, number, number, number, number, number],
  ranges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange],
): boolean {
  for (let i = 0; i < 6; i++) {
    const iv = ivs[i];
    const range = ranges[i];
    // 範囲外チェック
    if (iv < range.min || iv > range.max) {
      return false;
    }
  }
  return true;
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
