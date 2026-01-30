/**
 * Boot-Timing展開器モジュール
 * 起動日時・Timer0・VCount・キー入力からLCG Seedを算出する
 */

import { SeedCalculator } from '@/lib/core/seed-calculator';
import { keyMaskToKeyCode } from '@/lib/utils/key-input';
import type { SearchConditions } from '@/types/search';
import type { ROMVersion, ROMRegion, Hardware } from '@/types/rom';

/**
 * Boot-Timing展開用パラメータ
 */
export interface BootTimingParams {
  /** ROM版 (例: 'B', 'W', 'B2', 'W2') */
  romVersion: ROMVersion;
  /** リージョン (例: 'JPN', 'USA') */
  romRegion: ROMRegion;
  /** ハードウェア ('DS', 'DS_LITE', '3DS') */
  hardware: Hardware;
  /** MACアドレス (6バイト) */
  macAddress: number[];
  /** Timer0範囲 */
  timer0Range: { min: number; max: number };
  /** VCount範囲 */
  vcountRange: { min: number; max: number };
  /** 起動日時 (Date) */
  bootDatetime: Date;
  /** キー入力マスク */
  keyMask: number;
}

/**
 * LCG Seed展開結果
 */
export interface ExpandedSeed {
  /** 64bit LCG Seed */
  lcgSeed: bigint;
  /** Timer0値 */
  timer0: number;
  /** VCount値 */
  vcount: number;
  /** キーコード（変換後） */
  keyCode: number;
}

/**
 * Boot-Timingパラメータから全てのLCG Seedを展開する
 *
 * @param params - Boot-Timing展開用パラメータ
 * @returns 展開されたLCG Seedの配列
 */
export function expandBootTimingSeeds(params: BootTimingParams): ExpandedSeed[] {
  const calculator = new SeedCalculator();
  const results: ExpandedSeed[] = [];

  // キー入力マスクからキーコードに変換（単一値）
  const keyCode = keyMaskToKeyCode(params.keyMask);

  // 検索条件を構築（generateMessage で必要なフィールドのみ）
  const bootDate = params.bootDatetime;
  const conditions: SearchConditions = {
    romVersion: params.romVersion,
    romRegion: params.romRegion,
    hardware: params.hardware,
    macAddress: params.macAddress,
    keyInput: params.keyMask,
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: params.timer0Range,
      vcountRange: params.vcountRange,
    },
    timeRange: {
      hour: { start: 0, end: 23 },
      minute: { start: 0, end: 59 },
      second: { start: 0, end: 59 },
    },
    dateRange: {
      startYear: bootDate.getFullYear(),
      startMonth: bootDate.getMonth() + 1,
      startDay: bootDate.getDate(),
      endYear: bootDate.getFullYear(),
      endMonth: bootDate.getMonth() + 1,
      endDay: bootDate.getDate(),
    },
  };

  // Timer0・VCount範囲をループ
  for (let timer0 = params.timer0Range.min; timer0 <= params.timer0Range.max; timer0++) {
    for (let vcount = params.vcountRange.min; vcount <= params.vcountRange.max; vcount++) {
      try {
        // メッセージを生成
        const message = calculator.generateMessage(
          conditions,
          timer0,
          vcount,
          params.bootDatetime,
          keyCode
        );

        // SHA-1ハッシュからLCG Seedを算出
        const { lcgSeed } = calculator.calculateSeed(message);

        results.push({
          lcgSeed,
          timer0,
          vcount,
          keyCode,
        });
      } catch (e) {
        // ROMパラメータが見つからないなどのエラーはスキップ
        console.warn('Failed to calculate seed:', e);
      }
    }
  }

  return results;
}

/**
 * Boot-TimingパラメータをSearchConditionsに変換するヘルパー
 */
export function bootTimingToSearchConditions(params: BootTimingParams): SearchConditions {
  const bootDate = params.bootDatetime;
  return {
    romVersion: params.romVersion,
    romRegion: params.romRegion,
    hardware: params.hardware,
    macAddress: params.macAddress,
    keyInput: params.keyMask,
    timer0VCountConfig: {
      useAutoConfiguration: false,
      timer0Range: params.timer0Range,
      vcountRange: params.vcountRange,
    },
    timeRange: {
      hour: { start: 0, end: 23 },
      minute: { start: 0, end: 59 },
      second: { start: 0, end: 59 },
    },
    dateRange: {
      startYear: bootDate.getFullYear(),
      startMonth: bootDate.getMonth() + 1,
      startDay: bootDate.getDate(),
      endYear: bootDate.getFullYear(),
      endMonth: bootDate.getMonth() + 1,
      endDay: bootDate.getDate(),
    },
  };
}
