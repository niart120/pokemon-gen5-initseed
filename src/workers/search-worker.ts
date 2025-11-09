/**
 * Web Worker for Pokemon BW/BW2 Initial Seed Search
 * Performs heavy computation off the main thread to prevent UI blocking
 */

import { SeedCalculator } from '../lib/core/seed-calculator';
import { toMacUint8Array } from '@/lib/utils/mac-address';
// import { ProductionPerformanceMonitor } from '../lib/core/performance-monitor';
import type { SearchConditions } from '../types/search';
import type { InitialSeedResult } from '../types/search';
import type { TimerState, WorkerProgressMessage } from '../types/callbacks';
import type { Hardware } from '../types/rom';

// Performance optimization: Use larger batch sizes for better WASM utilization
const BATCH_SIZE_SECONDS = 86400;   // 1日
// const BATCH_SIZE_SECONDS = 86400 * 30;    // 30日


// Worker message types
export interface WorkerRequest {
  type: 'START_SEARCH' | 'PAUSE_SEARCH' | 'RESUME_SEARCH' | 'STOP_SEARCH';
  conditions?: SearchConditions;
  targetSeeds?: number[];
}

export interface WorkerResponse {
  type: 'PROGRESS' | 'RESULT' | 'COMPLETE' | 'ERROR' | 'PAUSED' | 'RESUMED' | 'STOPPED' | 'READY';
  progress?: WorkerProgressMessage;
  result?: InitialSeedResult;
  error?: string;
  errorCode?: string;
  message?: string;
}

// Timer state for accurate elapsed time calculation

// Worker state
const searchState = {
  isRunning: false,
  isPaused: false,
  shouldStop: false
};

// Timer state for elapsed time management
const timerState: TimerState = {
  cumulativeRunTime: 0,
  segmentStartTime: 0,
  isPaused: false
};

let calculator: SeedCalculator;

/**
 * MACアドレスを Uint8Array(6) に変換
 * - number[] や string[]("0x12"/"12") を受け付ける
 * - 値は 0-255 にクランプ
 * - 長さ不一致時は 6 バイトへ切り詰め/ゼロ埋め
 */
// toMacUint8Array は共通ユーティリティから利用

// Initialize calculator
async function initializeCalculator() {
  if (!calculator) {
    calculator = new SeedCalculator();
    // Initialize WebAssembly for integrated search
    try {
      await calculator.initializeWasm();
    } catch (error) {
      console.warn('WebAssembly failed in worker, using TypeScript fallback:', error);
    }
  }
}

/**
 * Timer management functions for accurate elapsed time calculation
 */
function startTimer() {
  timerState.cumulativeRunTime = 0;
  timerState.segmentStartTime = Date.now();
  timerState.isPaused = false;
}

function pauseTimer() {
  if (!timerState.isPaused) {
    timerState.cumulativeRunTime += Date.now() - timerState.segmentStartTime;
    timerState.isPaused = true;
  }
}

function resumeTimer() {
  if (timerState.isPaused) {
    timerState.segmentStartTime = Date.now();
    timerState.isPaused = false;
  }
}

function getElapsedTime(): number {
  return timerState.isPaused 
    ? timerState.cumulativeRunTime
    : timerState.cumulativeRunTime + (Date.now() - timerState.segmentStartTime);
}

/**
 * Process batch using integrated search for maximum performance
 */
async function processBatchIntegrated(
  conditions: SearchConditions,
  startTimestamp: number,
  endTimestamp: number,
  timer0Min: number,
  timer0Max: number,
  vcountMin: number,
  vcountMax: number,
  targetSeedSet: Set<number>,
  targetSeedArray: Uint32Array,
  onResult: (result: InitialSeedResult) => void
): Promise<void> {
  const wasmModule = calculator.getWasmModule();
  
  if (wasmModule && wasmModule.IntegratedSeedSearcher) {
    try {
      const params = calculator.getROMParameters(conditions.romVersion, conditions.romRegion);
      if (!params) {
        throw new Error(`No parameters found for ${conditions.romVersion} ${conditions.romRegion}`);
      }

      // Hardware別のframe値を設定
      const HARDWARE_FRAME_VALUES: Record<Hardware, number> = {
        DS: 8,
        DS_LITE: 6,
        '3DS': 9
      };
      const frameValue = HARDWARE_FRAME_VALUES[conditions.hardware] || 8;

      const searcher = new wasmModule.IntegratedSeedSearcher(
        toMacUint8Array(conditions.macAddress as unknown as Array<number | string>),
        new Uint32Array(params.nazo),
        conditions.hardware,
        conditions.keyInput,
        frameValue
      );

      const rangeSeconds = Math.floor((endTimestamp - startTimestamp) / 1000) + 1;

      if (rangeSeconds <= 0) {
        return;
      }

      // サブチャンク分割処理（15日単位、最大1296000秒）
      const subChunkSeconds = Math.min(1296000, rangeSeconds);
      
      for (let offset = 0; offset < rangeSeconds; offset += subChunkSeconds) {
        // 停止チェック
        if (searchState.shouldStop) break;
        
        // 一時停止チェック
        if (searchState.isPaused) {
          while (searchState.isPaused && !searchState.shouldStop) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        }
        
        if (searchState.shouldStop) break;
        
        const subChunkStart = new Date(startTimestamp + offset * 1000);
        const subChunkEnd = new Date(Math.min(
          startTimestamp + (offset + subChunkSeconds) * 1000,
          endTimestamp + 1000
        ));
  const subChunkRange = Math.floor((subChunkEnd.getTime() - subChunkStart.getTime()) / 1000);
        
  if (subChunkRange <= 0) break;

        // WebAssembly呼び出し前に非同期yield
        await new Promise(resolve => setTimeout(resolve, 0));
        
        // WebAssembly呼び出し前の一時停止チェック
        if (searchState.isPaused) {
          while (searchState.isPaused && !searchState.shouldStop) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          if (searchState.shouldStop) break;
        }

        // サブチャンクの統合検索実行（SIMD版）
        const results = searcher.search_seeds_integrated_simd(
          subChunkStart.getFullYear(),
          subChunkStart.getMonth() + 1,
          subChunkStart.getDate(),
          subChunkStart.getHours(),
          subChunkStart.getMinutes(),
          subChunkStart.getSeconds(),
          subChunkRange,
          timer0Min,
          timer0Max,
          vcountMin,
          vcountMax,
          targetSeedArray
        );
        
        // WebAssembly呼び出し後に非同期yield
        await new Promise(resolve => setTimeout(resolve, 0));

        // サブチャンクの結果を処理
        for (const result of results) {
          const resultDate = new Date(result.year, result.month - 1, result.date, result.hour, result.minute, result.second);
          const message = calculator.generateMessage(conditions, result.timer0, result.vcount, resultDate);
          const { hash } = calculator.calculateSeed(message);

          const searchResult: InitialSeedResult = {
            seed: result.seed,
            datetime: resultDate,
            timer0: result.timer0,
            vcount: result.vcount,
            conditions,
            message,
            sha1Hash: hash,
            isMatch: true,
          };
          onResult(searchResult);
        }
      }

      searcher.free();
      return;
    } catch (error) {
      console.error('Integrated search failed, falling back to individual processing:', error);
    }
  }

  // Fallback to individual processing
  await processBatchIndividual(
    [startTimestamp, endTimestamp],
    conditions,
    timer0Min,
    timer0Max,
    vcountMin,
    vcountMax,
    targetSeedSet,
    calculator,
    onResult
  );
}

/**
 * Process batch using individual calculations (fallback method)
 */
async function processBatchIndividual(
  timestampRange: number[],
  conditions: SearchConditions,
  timer0Min: number,
  timer0Max: number,
  vcountMin: number,
  vcountMax: number,
  targetSeedSet: Set<number>,
  calculator: SeedCalculator,
  onResult: (result: InitialSeedResult) => void
): Promise<void> {
  const [startTimestamp, endTimestamp] = timestampRange;
  const params = calculator.getROMParameters(conditions.romVersion, conditions.romRegion);

  if (!params) {
    throw new Error(`No parameters found for ${conditions.romVersion} ${conditions.romRegion}`);
  }
  
  let processedCount = 0;
  
  for (let timer0 = timer0Min; timer0 <= timer0Max; timer0++) {
    const actualVCount = calculator.getVCountForTimer0(params, timer0);

    for (let timestamp = startTimestamp; timestamp <= endTimestamp; timestamp += 1000) {
      // 停止チェック
      if (searchState.shouldStop) break;
      
      // 一時停止処理（1000操作ごと）
      if (processedCount % 1000 === 0) {
        while (searchState.isPaused && !searchState.shouldStop) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        // Event Loop yield
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      
      if (searchState.shouldStop) break;
      
      const currentDateTime = new Date(timestamp);
      
      try {
        // Generate message and calculate seed
        const message = calculator.generateMessage(conditions, timer0, actualVCount, currentDateTime);
        const { seed, hash } = calculator.calculateSeed(message);

        // Check if seed matches any target
        if (targetSeedSet.has(seed)) {
          const result: InitialSeedResult = {
            seed,
            datetime: currentDateTime,
            timer0,
            vcount: actualVCount,
            conditions,
            message,
            sha1Hash: hash,
            isMatch: true,
          };
          onResult(result);
        }
      } catch (error) {
        console.error('Error calculating seed:', error);
      }
      
      processedCount++;
    }
  }
}

// Main search function
async function performSearch(conditions: SearchConditions, targetSeeds: number[]) {
  try {
    await initializeCalculator();
    
    // Get ROM parameters
    const params = calculator.getROMParameters(conditions.romVersion, conditions.romRegion);
    if (!params) {
      throw new Error(`No parameters found for ${conditions.romVersion} ${conditions.romRegion}`);
    }

    // Calculate search space
    const timer0Min = conditions.timer0VCountConfig.timer0Range.min;
    const timer0Max = conditions.timer0VCountConfig.timer0Range.max;
    const vcountMin = conditions.timer0VCountConfig.vcountRange.min;
    const vcountMax = conditions.timer0VCountConfig.vcountRange.max;

    const timer0Count = timer0Max - timer0Min + 1;
    const vcountCount = vcountMax - vcountMin + 1;

    if (timer0Count <= 0) {
      throw new Error('Timer0 range must be at least 1 value');
    }

    if (vcountCount <= 0) {
      throw new Error('VCount range must be at least 1 value');
    }
    
    const startDate = new Date(
      conditions.dateRange.startYear,
      conditions.dateRange.startMonth - 1,
      conditions.dateRange.startDay,
      conditions.dateRange.startHour,
      conditions.dateRange.startMinute,
      conditions.dateRange.startSecond
    );
    
    const endDate = new Date(
      conditions.dateRange.endYear,
      conditions.dateRange.endMonth - 1,
      conditions.dateRange.endDay,
      conditions.dateRange.endHour,
      conditions.dateRange.endMinute,
      conditions.dateRange.endSecond
    );

    if (startDate > endDate) {
      throw new Error('Start date must be before or equal to end date');
    }

    const dateRange = Math.floor((endDate.getTime() - startDate.getTime()) / 1000) + 1;

    if (dateRange <= 0) {
      throw new Error('Date range must include at least one second');
    }

    const totalSteps = timer0Count * vcountCount * dateRange;

    // Convert target seeds to Set for faster lookup
    const targetSeedSet = new Set(targetSeeds);
    const targetSeedArray = new Uint32Array(targetSeedSet.size > 0 ? [...targetSeedSet] : []);

    let currentStep = 0;
    let matchesFound = 0;
    
    // Start accurate timer for elapsed time calculation
    startTimer();
    let lastProgressUpdate = Date.now();
    const progressUpdateInterval = 500; // Update progress every 500ms

    // Warn if auto-calculated VCount goes outside user-defined range
    for (let timer0 = timer0Min; timer0 <= timer0Max; timer0++) {
      const actualVCount = calculator.getVCountForTimer0(params, timer0);
      if (actualVCount < vcountMin || actualVCount > vcountMax) {
        console.warn(`[WORKER] Calculated VCount ${actualVCount} (0x${actualVCount.toString(16)}) is outside user range ${vcountMin}-${vcountMax}, but continuing search as requested.`);
      }
    }

    // Search using integrated approach
    const timeRangeSize = Math.min(BATCH_SIZE_SECONDS, dateRange);

    for (let timeStart = 0; timeStart < dateRange; timeStart += timeRangeSize) {
      if (searchState.shouldStop) break;

      // Handle pause with more frequent checking
      while (searchState.isPaused && !searchState.shouldStop) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      if (searchState.shouldStop) break;

      const timeEnd = Math.min(timeStart + timeRangeSize, dateRange);
      const secondsProcessed = timeEnd - timeStart;
      if (secondsProcessed <= 0) {
        continue;
      }

      const rangeStartTime = startDate.getTime() + timeStart * 1000;
      const rangeEndTime = startDate.getTime() + (timeEnd - 1) * 1000;

      try {
        // 非同期yield追加
        await new Promise(resolve => setTimeout(resolve, 0));

        await processBatchIntegrated(
          conditions,
          rangeStartTime,
          rangeEndTime,
          timer0Min,
          timer0Max,
          vcountMin,
          vcountMax,
          targetSeedSet,
          targetSeedArray,
          (result) => {
            matchesFound++;
            postMessage({ type: 'RESULT', result } as WorkerResponse);
          }
        );

        currentStep += secondsProcessed * timer0Count * vcountCount;

        // Send progress update only at specified intervals or on completion
        const now = Date.now();
        const shouldUpdateProgress =
          (now - lastProgressUpdate >= progressUpdateInterval) ||
          (currentStep >= totalSteps);

        if (shouldUpdateProgress) {
          lastProgressUpdate = now;

          const elapsedTime = getElapsedTime();

          // More accurate estimated time remaining calculation
          let estimatedTimeRemaining = 0;
          if (currentStep > 0 && currentStep < totalSteps) {
            const avgTimePerStep = elapsedTime / currentStep;
            const remainingSteps = totalSteps - currentStep;
            estimatedTimeRemaining = Math.round(avgTimePerStep * remainingSteps);
          }

          postMessage({
            type: 'PROGRESS',
            progress: {
              currentStep: Math.min(currentStep, totalSteps),
              totalSteps,
              elapsedTime,
              estimatedTimeRemaining,
              matchesFound,
              currentDateTime: new Date(rangeEndTime).toISOString()
            }
          } as WorkerResponse);
        }

      } catch (error) {
        console.error('Search batch error:', error);
      }
    }

    // Send completion message
    const finalElapsedTime = getElapsedTime();
    
    if (searchState.shouldStop) {
      postMessage({
        type: 'STOPPED',
        message: `Search stopped. Found ${matchesFound} matches out of ${currentStep} tested combinations.`,
        progress: {
          currentStep,
          totalSteps,
          elapsedTime: finalElapsedTime,
          estimatedTimeRemaining: 0,
          matchesFound
        }
      } as WorkerResponse);
    } else {
      postMessage({
        type: 'COMPLETE',
        message: `Search completed. Found ${matchesFound} matches out of ${totalSteps} combinations.`,
        progress: {
          currentStep, // 実際の処理ステップ数を保持（totalStepsではなく）
          totalSteps,
          elapsedTime: finalElapsedTime,
          estimatedTimeRemaining: 0,
          matchesFound
        }
      } as WorkerResponse);
    }

  } catch (error) {
    postMessage({
      type: 'ERROR',
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    } as WorkerResponse);
  } finally {
    searchState.isRunning = false;
    searchState.isPaused = false;
    searchState.shouldStop = false;
  }
}

// Message handler
self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { type, conditions, targetSeeds } = event.data;

  switch (type) {
    case 'START_SEARCH':
      if (!conditions || !targetSeeds) {
        postMessage({
          type: 'ERROR',
          error: 'Missing conditions or target seeds'
        } as WorkerResponse);
        return;
      }

      if (searchState.isRunning) {
        postMessage({
          type: 'ERROR',
          error: 'Search is already running'
        } as WorkerResponse);
        return;
      }

      searchState.isRunning = true;
      searchState.isPaused = false;
      searchState.shouldStop = false;

      performSearch(conditions, targetSeeds);
      break;

    case 'PAUSE_SEARCH':
      if (searchState.isRunning && !searchState.isPaused) {
        searchState.isPaused = true;
        pauseTimer(); // タイマーを一時停止
        postMessage({
          type: 'PAUSED',
          message: 'Search paused'
        } as WorkerResponse);
      }
      break;

    case 'RESUME_SEARCH':
      if (searchState.isRunning && searchState.isPaused) {
        searchState.isPaused = false;
        resumeTimer(); // タイマーを再開
        postMessage({
          type: 'RESUMED',
          message: 'Search resumed'
        } as WorkerResponse);
      }
      break;

    case 'STOP_SEARCH':
      if (searchState.isRunning) {
        searchState.shouldStop = true;
        // The search loop will handle the actual stopping
      }
      break;

    default:
      postMessage({
        type: 'ERROR',
        error: `Unknown message type: ${type}`
      } as WorkerResponse);
  }
};

// Worker ready signal
postMessage({
  type: 'READY',
  message: 'Search worker initialized'
} as WorkerResponse);
