/**
 * レポ針検索Worker
 *
 * UIスレッドをブロックせずに針検索を実行する。
 * LCG SeedモードとBoot-Timingモードの両方をサポート。
 */

import type {
  ReportNeedleWorkerRequest,
  ReportNeedleWorkerResponse,
  ReportNeedleSearchParams,
  ReportNeedleSearchResult,
} from '@/types/report-needle-search';
import {
  parseSeedHex,
  findNeedleMatches,
  parseNeedleString,
  expandBootTimingSeeds,
  type BootTimingParams,
} from '@/lib/report-needle-search';

interface InternalState {
  running: boolean;
  stopRequested: boolean;
  isPaused: boolean;
  pauseResolve: (() => void) | null;
}

const state: InternalState = {
  running: false,
  stopRequested: false,
  isPaused: false,
  pauseResolve: null,
};

const ctx = self as typeof self & { onclose?: () => void };
const post = (message: ReportNeedleWorkerResponse) => ctx.postMessage(message);

// Worker準備完了を通知
post({ type: 'READY' });

ctx.onmessage = (ev: MessageEvent<ReportNeedleWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START':
          await handleStart(msg.params);
          break;
        case 'PAUSE':
          handlePause();
          break;
        case 'RESUME':
          handleResume();
          break;
        case 'STOP':
          state.stopRequested = true;
          if (state.isPaused && state.pauseResolve) {
            state.pauseResolve();
            state.pauseResolve = null;
            state.isPaused = false;
          }
          break;
        default:
          break;
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      post({ type: 'ERROR', message });
    }
  })();
};

function handlePause(): void {
  if (!state.running || state.isPaused) {
    return;
  }
  state.isPaused = true;
  post({ type: 'PAUSED' });
}

function handleResume(): void {
  if (!state.running || !state.isPaused) {
    return;
  }
  state.isPaused = false;
  if (state.pauseResolve) {
    state.pauseResolve();
    state.pauseResolve = null;
  }
  post({ type: 'RESUMED' });
}

async function waitWhilePaused(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));

  if (!state.isPaused) {
    return;
  }

  await new Promise<void>((resolve) => {
    state.pauseResolve = resolve;
  });
}

async function handleStart(params: ReportNeedleSearchParams): Promise<void> {
  if (state.running) {
    post({ type: 'ERROR', message: '検索が既に実行中です' });
    return;
  }

  state.running = true;
  state.stopRequested = false;
  state.isPaused = false;

  try {
    const needles = parseNeedleString(params.needleValue);

    if (needles.length === 0) {
      post({ type: 'ERROR', message: '針列が空です' });
      state.running = false;
      return;
    }

    const allResults: ReportNeedleSearchResult[] = [];

    if (params.mode === 'initial-seed') {
      // LCG Seedモード
      await searchWithInitialSeed(params, needles, allResults);
    } else if (params.mode === 'startup') {
      // Boot-Timingモード
      await searchWithBootTiming(params, needles, allResults);
    }

    if (!state.stopRequested) {
      post({ type: 'COMPLETE', totalFound: allResults.length });
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'ERROR', message });
  } finally {
    state.running = false;
    state.isPaused = false;
    state.pauseResolve = null;
  }
}

/**
 * LCG Seedモードでの検索
 */
async function searchWithInitialSeed(
  params: ReportNeedleSearchParams,
  needles: Uint8Array,
  allResults: ReportNeedleSearchResult[]
): Promise<void> {
  const seedHex = params.seedHex ?? '';
  const baseSeed = parseSeedHex(seedHex);

  if (baseSeed === null) {
    post({ type: 'ERROR', message: '初期Seedの形式が不正です' });
    return;
  }

  // Timer0/VCountの範囲をループ
  for (let timer0 = params.timer0Range.min; timer0 <= params.timer0Range.max; timer0++) {
    for (let vcount = params.vcountRange.min; vcount <= params.vcountRange.max; vcount++) {
      await waitWhilePaused();
      if (state.stopRequested) return;

      const matches = findNeedleMatches(
        baseSeed,
        needles,
        params.advanceRange.start,
        params.advanceRange.end
      );

      for (const match of matches) {
        const result: ReportNeedleSearchResult = {
          consumptionIndex: match.consumptionIndex,
          timer0,
          vcount,
          initialSeed: match.initialSeed,
          currentSeed: match.currentSeed,
        };
        allResults.push(result);
        post({ type: 'RESULTS', results: [result] });
      }
    }
  }
}

/**
 * Boot-Timingモードでの検索
 */
async function searchWithBootTiming(
  params: ReportNeedleSearchParams,
  needles: Uint8Array,
  allResults: ReportNeedleSearchResult[]
): Promise<void> {
  // プロファイル情報の検証
  if (!params.profile) {
    post({ type: 'ERROR', message: 'プロファイル情報が必要です' });
    return;
  }

  if (!params.timestampIso) {
    post({ type: 'ERROR', message: '起動日時が必要です' });
    return;
  }

  const bootDatetime = new Date(params.timestampIso);
  if (Number.isNaN(bootDatetime.getTime())) {
    post({ type: 'ERROR', message: '起動日時の形式が不正です' });
    return;
  }

  // Boot-Timingパラメータを構築
  const bootParams: BootTimingParams = {
    romVersion: params.profile.romVersion,
    romRegion: params.profile.romRegion,
    hardware: params.profile.hardware,
    macAddress: params.profile.macAddress,
    timer0Range: params.timer0Range,
    vcountRange: params.vcountRange,
    bootDatetime,
    keyMask: params.keyMask ?? 0,
  };

  // LCG Seedを展開
  const expandedSeeds = expandBootTimingSeeds(bootParams);

  if (expandedSeeds.length === 0) {
    post({ type: 'ERROR', message: 'シードを展開できませんでした。プロファイル設定を確認してください。' });
    return;
  }

  // 各展開されたシードに対して針検索を実行
  for (const expanded of expandedSeeds) {
    await waitWhilePaused();
    if (state.stopRequested) return;

    const matches = findNeedleMatches(
      expanded.lcgSeed,
      needles,
      params.advanceRange.start,
      params.advanceRange.end
    );

    for (const match of matches) {
      const result: ReportNeedleSearchResult = {
        consumptionIndex: match.consumptionIndex,
        timer0: expanded.timer0,
        vcount: expanded.vcount,
        initialSeed: match.initialSeed,
        currentSeed: match.currentSeed,
      };
      allResults.push(result);
      post({ type: 'RESULTS', results: [result] });
    }
  }
}
