# TypeScript 型定義・Worker 設計

## 1. 型定義ファイル構成

```
src/types/
├── egg-boot-timing-search.ts  # 新規: 検索パラメータ・結果型
└── egg.ts                      # 既存: 必要に応じて拡張

src/lib/egg/
├── boot-timing-egg-search.ts  # 新規: 検索ロジック
├── boot-timing-egg-worker-manager.ts  # 新規: Worker管理
└── index.ts                    # 更新: エクスポート追加

src/workers/
└── egg-boot-timing-worker.ts  # 新規: Worker実装
```

## 2. 型定義

### 2.1 検索パラメータ

```typescript
// src/types/egg-boot-timing-search.ts

import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import type { DailyTimeRange } from '@/types/search';
import type {
  EggGenerationConditions,
  ParentsIVs,
  EggIndividualFilter,
  EggGameMode,
} from '@/types/egg';
import type { KeyName } from '@/lib/utils/key-input';

/**
 * 孵化乱数起動時間検索パラメータ
 * SearchConditions + EggGenerationParams の統合
 */
export interface EggBootTimingSearchParams {
  // === 起動時間パラメータ（SearchConditionsから） ===
  
  /** 開始日時 (ISO8601 UTC) */
  startDatetimeIso: string;
  
  /** 検索範囲秒数 */
  rangeSeconds: number;
  
  /** Timer0範囲 */
  timer0Range: {
    min: number;  // 0x0000-0xFFFF
    max: number;
  };
  
  /** VCount範囲 */
  vcountRange: {
    min: number;  // 0x00-0xFF
    max: number;
  };
  
  /** キー入力マスク (ビットマスク) */
  keyInputMask: number;
  
  /** MACアドレス (6バイト) */
  macAddress: readonly [number, number, number, number, number, number];
  
  /** ハードウェア */
  hardware: Hardware;
  
  /** ROMバージョン */
  romVersion: ROMVersion;
  
  /** ROM地域 */
  romRegion: ROMRegion;
  
  /** 時刻範囲フィルター（1日の中で検索する時間帯） */
  timeRange: DailyTimeRange;
  
  /** フレーム (通常8) */
  frame: number;
  
  // === 孵化条件パラメータ（EggGenerationParamsから） ===
  
  /** 生成条件 */
  conditions: EggGenerationConditions;
  
  /** 親個体値 */
  parents: ParentsIVs;
  
  /** 個体フィルター (null = フィルタなし) */
  filter: EggIndividualFilter | null;
  
  /** NPC消費考慮 */
  considerNpcConsumption: boolean;
  
  /** ゲームモード */
  gameMode: EggGameMode;
  
  // === 消費範囲 ===
  
  /** 開始advance */
  userOffset: number;
  
  /** 検索件数上限 (per seed) */
  advanceCount: number;
  
  // === 制限 ===
  
  /** 結果上限数 (全体) */
  maxResults: number;
}

/**
 * デフォルトパラメータ生成
 */
export function createDefaultEggBootTimingSearchParams(): EggBootTimingSearchParams {
  return {
    startDatetimeIso: new Date().toISOString(),
    rangeSeconds: 60,
    timer0Range: { min: 0x0C79, max: 0x0C7B },
    vcountRange: { min: 0x60, max: 0x60 },
    keyInputMask: 0,
    macAddress: [0, 0, 0, 0, 0, 0],
    hardware: 'DS',
    romVersion: 'B',
    romRegion: 'JPN',
    timeRange: {
      hour: { start: 0, end: 23 },
      minute: { start: 0, end: 59 },
      second: { start: 0, end: 59 },
    },
    frame: 8,
    conditions: createDefaultEggConditions(),
    parents: createDefaultParentsIVs(),
    filter: null,
    considerNpcConsumption: false,
    gameMode: EggGameMode.BwContinue,
    userOffset: 0,
    advanceCount: 1000,
    maxResults: 10000,
  };
}

// createDefaultEggConditions, createDefaultParentsIVs は既存のものを流用
import {
  createDefaultEggConditions,
  createDefaultParentsIVs,
} from '@/types/egg';
```

### 2.2 検索結果

```typescript
// src/types/egg-boot-timing-search.ts (続き)

import type { ResolvedEgg, EnumeratedEggData } from '@/types/egg';

/**
 * 起動条件情報
 */
export interface BootCondition {
  /** 起動日時 */
  datetime: Date;
  
  /** Timer0値 */
  timer0: number;
  
  /** VCount値 */
  vcount: number;
  
  /** キーコード (XOR 0x2FFF後) */
  keyCode: number;
  
  /** キー入力名リスト */
  keyInputNames: KeyName[];
  
  /** MACアドレス */
  macAddress: readonly [number, number, number, number, number, number];
}

/**
 * 孵化乱数起動時間検索結果1件
 */
export interface EggBootTimingSearchResult {
  /** 起動条件 */
  boot: BootCondition;
  
  /** LCG Seed (16進文字列) */
  lcgSeedHex: string;
  
  /** 個体情報 */
  egg: EnumeratedEggData;
  
  /** 安定性フラグ */
  isStable: boolean;
}

/**
 * WASM結果からの変換
 */
export function convertWasmResult(wasmResult: WasmEggBootTimingSearchResult): EggBootTimingSearchResult {
  return {
    boot: {
      datetime: new Date(Date.UTC(
        wasmResult.year,
        wasmResult.month - 1,
        wasmResult.date,
        wasmResult.hour,
        wasmResult.minute,
        wasmResult.second
      )),
      timer0: wasmResult.timer0,
      vcount: wasmResult.vcount,
      keyCode: wasmResult.keyCode,
      keyInputNames: keyCodeToNames(wasmResult.keyCode),
      macAddress: [0, 0, 0, 0, 0, 0], // パラメータから復元
    },
    lcgSeedHex: wasmResult.lcgSeedHex,
    egg: {
      advance: Number(wasmResult.advance),
      isStable: wasmResult.isStable,
      egg: {
        lcgSeedHex: wasmResult.lcgSeedHex,
        ivs: wasmResult.ivs as [number, number, number, number, number, number],
        nature: wasmResult.nature,
        gender: genderFromCode(wasmResult.gender),
        ability: wasmResult.ability as 0 | 1 | 2,
        shiny: wasmResult.shiny as 0 | 1 | 2,
        pid: wasmResult.pid,
        hiddenPower: wasmResult.hpKnown
          ? { type: 'known', hpType: wasmResult.hpType, power: wasmResult.hpPower }
          : { type: 'unknown' },
      },
    },
    isStable: wasmResult.isStable,
  };
}

function genderFromCode(code: number): 'male' | 'female' | 'genderless' {
  switch (code) {
    case 0: return 'male';
    case 1: return 'female';
    default: return 'genderless';
  }
}

// keyCodeToNames は既存の key-input.ts から流用
import { keyCodeToNames } from '@/lib/utils/key-input';
```

### 2.3 Worker通信型

```typescript
// src/types/egg-boot-timing-search.ts (続き)

/**
 * Worker リクエスト
 */
export type EggBootTimingWorkerRequest =
  | {
      type: 'START_SEARCH';
      params: EggBootTimingSearchParams;
      requestId?: string;
    }
  | {
      type: 'STOP';
      requestId?: string;
    };

/**
 * Worker レスポンス
 */
export type EggBootTimingWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: EggBootTimingProgress }
  | { type: 'RESULTS'; payload: EggBootTimingResultsPayload }
  | { type: 'COMPLETE'; payload: EggBootTimingCompletion }
  | { type: 'ERROR'; message: string; category: EggBootTimingErrorCategory; fatal: boolean };

/**
 * 進捗情報
 */
export interface EggBootTimingProgress {
  /** 処理済み起動条件の組み合わせ数 */
  processedCombinations: number;
  
  /** 総組み合わせ数 */
  totalCombinations: number;
  
  /** 見つかった結果数 */
  foundCount: number;
  
  /** 進捗率 (0-100) */
  progressPercent: number;
  
  /** 経過時間 (ms) */
  elapsedMs: number;
  
  /** 推定残り時間 (ms) */
  estimatedRemainingMs: number;
}

/**
 * 結果ペイロード（バッチ送信用）
 */
export interface EggBootTimingResultsPayload {
  results: EggBootTimingSearchResult[];
  batchIndex: number;
}

/**
 * 完了情報
 */
export interface EggBootTimingCompletion {
  /** 完了理由 */
  reason: 'completed' | 'stopped' | 'max-results' | 'error';
  
  /** 処理した起動条件の組み合わせ数 */
  processedCombinations: number;
  
  /** 総組み合わせ数 */
  totalCombinations: number;
  
  /** 見つかった結果数 */
  resultsCount: number;
  
  /** 経過時間 (ms) */
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type EggBootTimingErrorCategory = 
  | 'VALIDATION'      // パラメータ検証エラー
  | 'WASM_INIT'       // WASM初期化エラー
  | 'RUNTIME'         // 実行時エラー
  | 'ABORTED';        // 中断

/**
 * 完了理由ラベル
 */
export const COMPLETION_REASON_LABELS: Record<EggBootTimingCompletion['reason'], string> = {
  'completed': '検索完了',
  'stopped': 'ユーザー停止',
  'max-results': '結果上限到達',
  'error': 'エラー終了',
};

/**
 * 型ガード
 */
export function isEggBootTimingWorkerResponse(data: unknown): data is EggBootTimingWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as { type?: unknown };
  if (typeof obj.type !== 'string') return false;
  return ['READY', 'PROGRESS', 'RESULTS', 'COMPLETE', 'ERROR'].includes(obj.type);
}
```

### 2.4 バリデーション

```typescript
// src/types/egg-boot-timing-search.ts (続き)

/**
 * パラメータバリデーション
 */
export function validateEggBootTimingSearchParams(params: EggBootTimingSearchParams): string[] {
  const errors: string[] = [];
  
  // 日時検証
  const startDate = new Date(params.startDatetimeIso);
  if (isNaN(startDate.getTime())) {
    errors.push('開始日時が無効です');
  }
  
  // 範囲検証
  if (params.rangeSeconds < 1 || params.rangeSeconds > 86400 * 365) {
    errors.push('検索範囲は1秒から1年以内である必要があります');
  }
  
  // Timer0検証
  if (params.timer0Range.min > params.timer0Range.max) {
    errors.push('Timer0の最小値は最大値以下である必要があります');
  }
  if (params.timer0Range.min < 0 || params.timer0Range.max > 0xFFFF) {
    errors.push('Timer0は0x0000-0xFFFFの範囲である必要があります');
  }
  
  // VCount検証
  if (params.vcountRange.min > params.vcountRange.max) {
    errors.push('VCountの最小値は最大値以下である必要があります');
  }
  if (params.vcountRange.min < 0 || params.vcountRange.max > 0xFF) {
    errors.push('VCountは0x00-0xFFの範囲である必要があります');
  }
  
  // MACアドレス検証
  if (params.macAddress.length !== 6 || params.macAddress.some(b => b < 0 || b > 255)) {
    errors.push('MACアドレスは6バイトの配列である必要があります');
  }
  
  // 時刻範囲検証
  const { hour, minute, second } = params.timeRange;
  if (hour.start > hour.end || hour.start < 0 || hour.end > 23) {
    errors.push('時の範囲が無効です');
  }
  if (minute.start > minute.end || minute.start < 0 || minute.end > 59) {
    errors.push('分の範囲が無効です');
  }
  if (second.start > second.end || second.start < 0 || second.end > 59) {
    errors.push('秒の範囲が無効です');
  }
  
  // 消費範囲検証
  if (params.userOffset < 0 || params.userOffset > Number.MAX_SAFE_INTEGER) {
    errors.push('開始advanceは0以上の整数である必要があります');
  }
  if (params.advanceCount < 1 || params.advanceCount > 1000000) {
    errors.push('検索件数は1-1000000の範囲である必要があります');
  }
  
  // 結果上限検証
  if (params.maxResults < 1 || params.maxResults > 100000) {
    errors.push('結果上限は1-100000の範囲である必要があります');
  }
  
  return errors;
}

/**
 * 計算量見積もり
 */
export function estimateSearchCombinations(params: EggBootTimingSearchParams): number {
  const timer0Count = params.timer0Range.max - params.timer0Range.min + 1;
  const vcountCount = params.vcountRange.max - params.vcountRange.min + 1;
  
  // キーコード数を概算（ビットマスクから）
  const keyCodeCount = Math.pow(2, countBits(params.keyInputMask));
  
  // 時刻範囲内の秒数
  const { hour, minute, second } = params.timeRange;
  const hourCount = hour.end - hour.start + 1;
  const minuteCount = minute.end - minute.start + 1;
  const secondCount = second.end - second.start + 1;
  const allowedSecondsPerDay = hourCount * minuteCount * secondCount;
  
  // 日数
  const days = Math.ceil(params.rangeSeconds / 86400);
  const effectiveSeconds = Math.min(params.rangeSeconds, allowedSecondsPerDay * days);
  
  return effectiveSeconds * timer0Count * vcountCount * keyCodeCount;
}

function countBits(n: number): number {
  let count = 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  return count;
}
```

## 3. Worker実装

```typescript
// src/workers/egg-boot-timing-worker.ts

import init, {
  EggBootTimingSearcher,
  GenerationConditionsJs,
  ParentsIVsJs,
  IndividualFilterJs,
  EverstonePlanJs,
  TrainerIds,
  GenderRatio,
  GameMode,
} from '@/wasm/pokemon_bw_seed_wasm';

import type {
  EggBootTimingWorkerRequest,
  EggBootTimingWorkerResponse,
  EggBootTimingSearchParams,
} from '@/types/egg-boot-timing-search';

import { resolveNazoValue } from '@/lib/core/nazo-resolver';

let wasmInitialized = false;
let stopRequested = false;

// Worker初期化
async function initialize(): Promise<void> {
  if (!wasmInitialized) {
    await init();
    wasmInitialized = true;
  }
  postMessage({ type: 'READY', version: '1' } satisfies EggBootTimingWorkerResponse);
}

// メッセージハンドラ
self.onmessage = async (event: MessageEvent<EggBootTimingWorkerRequest>) => {
  const request = event.data;
  
  try {
    switch (request.type) {
      case 'START_SEARCH':
        await handleSearch(request.params);
        break;
      case 'STOP':
        stopRequested = true;
        break;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    postMessage({
      type: 'ERROR',
      message,
      category: 'RUNTIME',
      fatal: true,
    } satisfies EggBootTimingWorkerResponse);
  }
};

async function handleSearch(params: EggBootTimingSearchParams): Promise<void> {
  if (!wasmInitialized) {
    await initialize();
  }
  
  stopRequested = false;
  const startTime = performance.now();
  
  // nazo値を解決
  const nazo = resolveNazoValue(params.romVersion, params.romRegion);
  
  // WASM用条件オブジェクト構築
  const conditionsJs = buildConditionsJs(params);
  const parentsJs = buildParentsJs(params);
  const filterJs = params.filter ? buildFilterJs(params.filter) : undefined;
  
  // GameMode変換
  const gameMode = params.gameMode as number as GameMode;
  
  // Searcher構築
  const searcher = new EggBootTimingSearcher(
    new Uint8Array(params.macAddress),
    new Uint32Array(nazo),
    params.hardware,
    params.keyInputMask,
    params.frame,
    params.timeRange.hour.start,
    params.timeRange.hour.end,
    params.timeRange.minute.start,
    params.timeRange.minute.end,
    params.timeRange.second.start,
    params.timeRange.second.end,
    conditionsJs,
    parentsJs,
    filterJs ?? undefined,
    params.considerNpcConsumption,
    gameMode,
    BigInt(params.userOffset),
    params.advanceCount,
  );
  
  // 検索実行
  const startDate = new Date(params.startDatetimeIso);
  
  const wasmResults = searcher.search_eggs_integrated_simd(
    startDate.getUTCFullYear(),
    startDate.getUTCMonth() + 1,
    startDate.getUTCDate(),
    startDate.getUTCHours(),
    startDate.getUTCMinutes(),
    startDate.getUTCSeconds(),
    params.rangeSeconds,
    params.timer0Range.min,
    params.timer0Range.max,
    params.vcountRange.min,
    params.vcountRange.max,
  );
  
  // 結果変換と送信
  const results = [];
  for (let i = 0; i < wasmResults.length && i < params.maxResults; i++) {
    if (stopRequested) break;
    results.push(convertWasmResultToTs(wasmResults[i], params.macAddress));
  }
  
  const elapsedMs = performance.now() - startTime;
  
  // 結果送信
  postMessage({
    type: 'RESULTS',
    payload: { results, batchIndex: 0 },
  } satisfies EggBootTimingWorkerResponse);
  
  // 完了通知
  postMessage({
    type: 'COMPLETE',
    payload: {
      reason: stopRequested ? 'stopped' : 'completed',
      processedCombinations: 0, // TODO: 計算
      totalCombinations: 0,
      resultsCount: results.length,
      elapsedMs,
    },
  } satisfies EggBootTimingWorkerResponse);
}

function buildConditionsJs(params: EggBootTimingSearchParams): GenerationConditionsJs {
  const conditions = new GenerationConditionsJs();
  
  conditions.has_nidoran_flag = params.conditions.hasNidoranFlag;
  conditions.uses_ditto = params.conditions.usesDitto;
  conditions.allow_hidden_ability = params.conditions.femaleParentAbility === 2;
  conditions.female_parent_has_hidden = params.conditions.femaleParentAbility === 2;
  conditions.reroll_count = params.conditions.masudaMethod ? 3 : 0;
  
  // Everstone設定
  if (params.conditions.everstone.type === 'fixed') {
    conditions.set_everstone(EverstonePlanJs.fixed(params.conditions.everstone.nature));
  } else {
    conditions.set_everstone(EverstonePlanJs.none());
  }
  
  // TrainerIds設定
  conditions.set_trainer_ids(new TrainerIds(params.conditions.tid, params.conditions.sid));
  
  // GenderRatio設定
  conditions.set_gender_ratio(
    new GenderRatio(
      params.conditions.genderRatio.threshold,
      params.conditions.genderRatio.genderless
    )
  );
  
  return conditions;
}

function buildParentsJs(params: EggBootTimingSearchParams): ParentsIVsJs {
  const parents = new ParentsIVsJs();
  parents.male = params.parents.male;
  parents.female = params.parents.female;
  return parents;
}

function buildFilterJs(filter: EggIndividualFilter): IndividualFilterJs {
  const filterJs = new IndividualFilterJs();
  
  filter.ivRanges.forEach((range, i) => {
    filterJs.set_iv_range(i, range.min, range.max);
  });
  
  if (filter.nature !== undefined) {
    filterJs.set_nature(filter.nature);
  }
  if (filter.gender !== undefined) {
    filterJs.set_gender(filter.gender === 'male' ? 0 : filter.gender === 'female' ? 1 : 2);
  }
  if (filter.ability !== undefined) {
    filterJs.set_ability(filter.ability);
  }
  if (filter.shiny !== undefined) {
    filterJs.set_shiny(filter.shiny);
  }
  if (filter.hiddenPowerType !== undefined) {
    filterJs.set_hidden_power_type(filter.hiddenPowerType);
  }
  if (filter.hiddenPowerPower !== undefined) {
    filterJs.set_hidden_power_power(filter.hiddenPowerPower);
  }
  
  return filterJs;
}

function convertWasmResultToTs(
  wasmResult: unknown,
  macAddress: readonly [number, number, number, number, number, number]
): EggBootTimingSearchResult {
  const r = wasmResult as {
    year: number; month: number; date: number;
    hour: number; minute: number; second: number;
    timer0: number; vcount: number; keyCode: number;
    lcgSeedHex: string;
    advance: bigint; isStable: boolean;
    ivs: number[]; nature: number; gender: number;
    ability: number; shiny: number; pid: number;
    hpType: number; hpPower: number; hpKnown: boolean;
  };
  
  return {
    boot: {
      datetime: new Date(Date.UTC(r.year, r.month - 1, r.date, r.hour, r.minute, r.second)),
      timer0: r.timer0,
      vcount: r.vcount,
      keyCode: r.keyCode,
      keyInputNames: keyCodeToNames(r.keyCode),
      macAddress,
    },
    lcgSeedHex: r.lcgSeedHex,
    egg: {
      advance: Number(r.advance),
      isStable: r.isStable,
      egg: {
        lcgSeedHex: r.lcgSeedHex,
        ivs: r.ivs as [number, number, number, number, number, number],
        nature: r.nature,
        gender: r.gender === 0 ? 'male' : r.gender === 1 ? 'female' : 'genderless',
        ability: r.ability as 0 | 1 | 2,
        shiny: r.shiny as 0 | 1 | 2,
        pid: r.pid,
        hiddenPower: r.hpKnown
          ? { type: 'known', hpType: r.hpType, power: r.hpPower }
          : { type: 'unknown' },
      },
    },
    isStable: r.isStable,
  };
}

import { keyCodeToNames } from '@/lib/utils/key-input';
import type { EggIndividualFilter, EggBootTimingSearchResult } from '@/types/egg-boot-timing-search';

// 初期化実行
initialize().catch(console.error);
```

## 4. WorkerManager実装

```typescript
// src/lib/egg/boot-timing-egg-worker-manager.ts

import type {
  EggBootTimingSearchParams,
  EggBootTimingSearchResult,
  EggBootTimingWorkerRequest,
  EggBootTimingWorkerResponse,
  EggBootTimingCompletion,
  EggBootTimingProgress,
} from '@/types/egg-boot-timing-search';

export interface EggBootTimingWorkerCallbacks {
  onReady?: () => void;
  onProgress?: (progress: EggBootTimingProgress) => void;
  onResults?: (results: EggBootTimingSearchResult[]) => void;
  onComplete?: (completion: EggBootTimingCompletion) => void;
  onError?: (error: { message: string; category: string; fatal: boolean }) => void;
}

export class EggBootTimingWorkerManager {
  private worker: Worker | null = null;
  private callbacks: EggBootTimingWorkerCallbacks = {};
  private isRunning = false;
  
  constructor() {}
  
  /**
   * Worker初期化
   */
  async initialize(callbacks: EggBootTimingWorkerCallbacks): Promise<void> {
    this.callbacks = callbacks;
    
    // Worker作成
    this.worker = new Worker(
      new URL('@/workers/egg-boot-timing-worker.ts', import.meta.url),
      { type: 'module' }
    );
    
    // メッセージハンドラ設定
    this.worker.onmessage = (event: MessageEvent<EggBootTimingWorkerResponse>) => {
      this.handleMessage(event.data);
    };
    
    this.worker.onerror = (error) => {
      this.callbacks.onError?.({
        message: error.message || 'Worker error',
        category: 'RUNTIME',
        fatal: true,
      });
    };
    
    // READY待機
    await new Promise<void>((resolve) => {
      const originalOnReady = this.callbacks.onReady;
      this.callbacks.onReady = () => {
        originalOnReady?.();
        resolve();
      };
    });
  }
  
  /**
   * 検索開始
   */
  async startSearch(params: EggBootTimingSearchParams): Promise<void> {
    if (!this.worker) {
      throw new Error('Worker not initialized');
    }
    if (this.isRunning) {
      throw new Error('Search already running');
    }
    
    this.isRunning = true;
    
    const request: EggBootTimingWorkerRequest = {
      type: 'START_SEARCH',
      params,
      requestId: crypto.randomUUID(),
    };
    
    this.worker.postMessage(request);
  }
  
  /**
   * 検索停止
   */
  stopSearch(): void {
    if (!this.worker || !this.isRunning) return;
    
    const request: EggBootTimingWorkerRequest = {
      type: 'STOP',
    };
    
    this.worker.postMessage(request);
  }
  
  /**
   * Worker終了
   */
  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.isRunning = false;
  }
  
  /**
   * 実行中フラグ
   */
  get running(): boolean {
    return this.isRunning;
  }
  
  private handleMessage(response: EggBootTimingWorkerResponse): void {
    switch (response.type) {
      case 'READY':
        this.callbacks.onReady?.();
        break;
        
      case 'PROGRESS':
        this.callbacks.onProgress?.(response.payload);
        break;
        
      case 'RESULTS':
        this.callbacks.onResults?.(response.payload.results);
        break;
        
      case 'COMPLETE':
        this.isRunning = false;
        this.callbacks.onComplete?.(response.payload);
        break;
        
      case 'ERROR':
        if (response.fatal) {
          this.isRunning = false;
        }
        this.callbacks.onError?.({
          message: response.message,
          category: response.category,
          fatal: response.fatal,
        });
        break;
    }
  }
}
```

## 5. 既存コードとの統合ポイント

### 5.1 nazo-resolver

```typescript
// src/lib/core/nazo-resolver.ts の resolveNazoValue を使用
import { resolveNazoValue } from '@/lib/core/nazo-resolver';

// 使用例
const nazo = resolveNazoValue('B', 'JPN');
// => [0x02215F10, 0x02761150, 0x00000000, 0x00000000, 0x02761150]
```

### 5.2 key-input

```typescript
// src/lib/utils/key-input.ts の keyCodeToNames を使用
import { keyCodeToNames } from '@/lib/utils/key-input';

// 使用例
const names = keyCodeToNames(0x2FF7);
// => ['L', 'Start']
```

### 5.3 既存のEgg型

```typescript
// src/types/egg.ts から流用
import type {
  EggGenerationConditions,
  ParentsIVs,
  EggIndividualFilter,
  EggGameMode,
  EnumeratedEggData,
  ResolvedEgg,
} from '@/types/egg';
```

## 6. エクスポート設定

```typescript
// src/types/index.ts に追加
export * from './egg-boot-timing-search';

// src/lib/egg/index.ts に追加
export * from './boot-timing-egg-worker-manager';
```
