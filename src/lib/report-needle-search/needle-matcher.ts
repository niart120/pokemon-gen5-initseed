/**
 * 針マッチャーモジュール
 * 単一シードに対する針列の一致探索ロジック
 */

import { nextSeed, calcNeedleDir, advanceSeed, seedToHex } from './lcg';

/**
 * 針一致結果
 */
export interface NeedleMatchResult {
  /** 一致した消費位置（針列の末尾が一致した位置） */
  consumptionIndex: number;
  /** 初期Seed（LCG状態、16進表記用） */
  initialSeed: string;
  /** 消費位置での現在Seed */
  currentSeed: string;
}

/**
 * 単一シードに対して針列の一致を探索する
 * 
 * @param baseSeed - 探索開始時の64bit LCGシード
 * @param needles - 0-7の針列（Uint8Array）
 * @param rangeStart - 探索開始位置
 * @param rangeEnd - 探索終了位置（含む）
 * @returns 一致した結果の配列
 */
export function findNeedleMatches(
  baseSeed: bigint,
  needles: Uint8Array,
  rangeStart: number,
  rangeEnd: number,
): NeedleMatchResult[] {
  const results: NeedleMatchResult[] = [];
  const m = needles.length;

  if (m === 0 || rangeStart > rangeEnd || rangeStart < 0) {
    return results;
  }

  // rangeStartまでジャンプ
  let seed = advanceSeed(baseSeed, rangeStart);
  const initialSeedHex = seedToHex(baseSeed);

  // rangeStart..=rangeEnd をスキャン
  for (let i = rangeStart; i <= rangeEnd; i++) {
    // 現在位置から針列長分だけチェック
    let testSeed = seed;
    let match = true;

    for (let j = 0; j < m; j++) {
      const dir = calcNeedleDir(testSeed);
      if (dir !== needles[j]) {
        match = false;
        break;
      }
      testSeed = nextSeed(testSeed);
    }

    if (match) {
      // 消費位置は針列の末尾が一致した位置
      const consumptionIndex = i + m - 1;
      // 現在Seedは一致開始位置のシード
      results.push({
        consumptionIndex,
        initialSeed: initialSeedHex,
        currentSeed: seedToHex(seed),
      });
    }

    // 次の位置へ
    seed = nextSeed(seed);
  }

  return results;
}

/**
 * 針列文字列をUint8Arrayに変換
 * 0-7以外の文字は除去
 */
export function parseNeedleString(needleStr: string): Uint8Array {
  const sanitized = needleStr.replace(/[^0-7]/g, '');
  const arr = new Uint8Array(sanitized.length);
  for (let i = 0; i < sanitized.length; i++) {
    arr[i] = parseInt(sanitized[i]!, 10);
  }
  return arr;
}
