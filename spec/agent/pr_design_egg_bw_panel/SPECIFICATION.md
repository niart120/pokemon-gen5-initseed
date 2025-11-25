# EggBWPanel 設計仕様書

## 1. 概要

### 1.1 目的
EggSeedEnumerator (wasm-pkg/src/egg_seed_enumerator.rs) のインタフェース仕様に基づき、タマゴ個体生成一覧表示機能を実装する。
ユーザーは初期Seed（または起動時間）と親個体情報を指定し、指定した消費範囲の個体一覧をフィルタリングして取得できる。

### 1.2 要件
- 初期Seed または 起動時間から個体生成
- 親個体値・親個体情報（メタモン利用、夢特性有無）の指定
- かわらずのいし有無、性別比、種族特性（通常/ニドラン系/バルビート系）の設定
- NPC消費考慮の有無
- 指定消費範囲での個体一覧（テーブル）表示
- フィルター適用（個体値範囲、特性、性格、性別、めざパタイプ・威力）
- Panel内でのモード切り替え（起動時間検索[WIP] / 個体一覧表示）

### 1.3 アーキテクチャ方針
既存のGenerationPanel実装パターンに準拠:
- Worker ベースの非同期処理
- WorkerManager によるライフサイクル管理とコールバック配信
- Zustand による状態管理
- 責任分離: Worker=計算、Manager=制御、UI=表示

## 2. データ型定義

### 2.1 型ファイル: `src/types/egg.ts`

#### 2.1.1 基本型

```typescript
/**
 * 親個体の役割
 */
export type ParentRole = 'male' | 'female';

/**
 * かわらずのいし設定
 */
export type EverstonePlan = 
  | { type: 'none' }
  | { type: 'fixed'; nature: number }; // 0-24

/**
 * 親個体のIVセット
 * 各値は 0-31 または 32 (Unknown)
 */
export type IvSet = [number, number, number, number, number, number]; // HP, Atk, Def, SpA, SpD, Spe

/**
 * 親個体情報
 */
export interface ParentsIVs {
  male: IvSet;
  female: IvSet;
}

/**
 * 遺伝スロット情報
 */
export interface InheritanceSlot {
  stat: number; // 0-5 (HP, Atk, Def, SpA, SpD, Spe)
  parent: ParentRole;
}

/**
 * 個体値範囲フィルター
 */
export interface StatRange {
  min: number; // 0-32
  max: number; // 0-32
}

/**
 * めざめるパワー情報
 */
export type HiddenPowerInfo =
  | { type: 'known'; hpType: number; power: number } // hpType: 0-15
  | { type: 'unknown' };

/**
 * 生成条件
 */
export interface EggGenerationConditions {
  hasNidoranFlag: boolean;        // ニドラン系/バルビート系
  everstone: EverstonePlan;       // かわらずのいし
  usesDitto: boolean;             // メタモン利用
  allowHiddenAbility: boolean;    // 夢特性許可
  femaleParentHasHidden: boolean; // 親♀が夢特性を持つか
  rerollCount: number;            // PIDリロール回数 (0-5, 国際孵化用)
  tid: number;                    // 0-65535
  sid: number;                    // 0-65535
  genderRatio: {
    threshold: number;            // 0-255
    genderless: boolean;
  };
}

/**
 * 個体フィルター
 */
export interface EggIndividualFilter {
  ivRanges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
  nature?: number;                // 0-24
  gender?: 'male' | 'female' | 'genderless';
  ability?: 0 | 1 | 2;            // 0=特性1, 1=特性2, 2=夢特性
  shiny?: 0 | 1 | 2;              // 0=通常, 1=正方形色違い, 2=星型色違い
  hiddenPowerType?: number;       // 0-15
  hiddenPowerPower?: number;      // 30-70
}

/**
 * 生成された個体データ
 */
export interface ResolvedEgg {
  ivs: IvSet;
  nature: number;         // 0-24
  gender: 'male' | 'female' | 'genderless';
  ability: 0 | 1 | 2;
  shiny: 0 | 1 | 2;
  pid: number;            // u32
  hiddenPower: HiddenPowerInfo;
}

/**
 * 列挙された個体データ（advance情報付き）
 */
export interface EnumeratedEggData {
  advance: number;        // bigint → number に変換
  egg: ResolvedEgg;
  isStable: boolean;      // NPC消費考慮時の安定性
}
```

#### 2.1.2 パラメータ型

```typescript
/**
 * タマゴ生成パラメータ
 */
export interface EggGenerationParams {
  baseSeed: bigint;                      // 初期Seed
  userOffset: bigint;                    // 開始advance (0から開始が基本)
  count: number;                         // 列挙上限 (1-100000)
  conditions: EggGenerationConditions;   // 生成条件
  parents: ParentsIVs;                   // 親個体値
  filter: EggIndividualFilter | null;    // フィルター (null=全件)
  considerNpcConsumption: boolean;       // NPC消費考慮
  gameMode: number;                      // GameMode (0=BwNew, 1=BwContinue, 2=Bw2New, 3=Bw2Continue)
}

/**
 * UI用16進数パラメータ
 */
export interface EggGenerationParamsHex {
  baseSeedHex: string;
  userOffsetHex: string;
  count: number;
  conditions: EggGenerationConditions;
  parents: ParentsIVs;
  filter: EggIndividualFilter | null;
  considerNpcConsumption: boolean;
  gameMode: number;
}

/**
 * パラメータ変換関数
 */
export function hexParamsToEggParams(h: EggGenerationParamsHex): EggGenerationParams {
  return {
    baseSeed: BigInt('0x' + h.baseSeedHex.toLowerCase().replace(/^0x/, '')),
    userOffset: BigInt('0x' + h.userOffsetHex.toLowerCase().replace(/^0x/, '')),
    count: h.count,
    conditions: h.conditions,
    parents: h.parents,
    filter: h.filter,
    considerNpcConsumption: h.considerNpcConsumption,
    gameMode: h.gameMode,
  };
}

/**
 * パラメータバリデーション
 */
export function validateEggParams(params: EggGenerationParams): string[] {
  const errors: string[] = [];
  
  if (params.count < 1 || params.count > 100000) {
    errors.push('count must be 1-100000');
  }
  
  if (params.conditions.rerollCount < 0 || params.conditions.rerollCount > 5) {
    errors.push('rerollCount must be 0-5');
  }
  
  if (params.conditions.tid < 0 || params.conditions.tid > 65535) {
    errors.push('tid must be 0-65535');
  }
  
  if (params.conditions.sid < 0 || params.conditions.sid > 65535) {
    errors.push('sid must be 0-65535');
  }
  
  // IV値検証
  const validateIvSet = (ivs: IvSet, name: string) => {
    ivs.forEach((iv, i) => {
      if (iv < 0 || iv > 32) {
        errors.push(`${name}[${i}] must be 0-32`);
      }
    });
  };
  
  validateIvSet(params.parents.male, 'parents.male');
  validateIvSet(params.parents.female, 'parents.female');
  
  return errors;
}
```

#### 2.1.3 Worker通信型

```typescript
/**
 * Worker リクエスト
 */
export type EggWorkerRequest =
  | { type: 'START_GENERATION'; params: EggGenerationParams; requestId?: string }
  | { type: 'STOP'; requestId?: string };

/**
 * Worker レスポンス
 */
export type EggWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'RESULTS'; payload: EggResultsPayload }
  | { type: 'COMPLETE'; payload: EggCompletion }
  | { type: 'ERROR'; message: string; category: EggErrorCategory; fatal: boolean };

/**
 * 結果ペイロード
 */
export interface EggResultsPayload {
  results: EnumeratedEggData[];
}

/**
 * 完了情報
 */
export interface EggCompletion {
  reason: 'max-count' | 'stopped' | 'error';
  processedCount: number;    // 実際に処理した個体数
  filteredCount: number;     // フィルター適用後の個体数
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type EggErrorCategory = 'VALIDATION' | 'WASM_INIT' | 'RUNTIME';

/**
 * 型ガード
 */
export function isEggWorkerResponse(data: unknown): data is EggWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return typeof obj.type === 'string' && 
    ['READY', 'RESULTS', 'COMPLETE', 'ERROR'].includes(obj.type);
}
```

## 3. Worker実装

### 3.1 ファイル: `src/workers/egg-worker.ts`

```typescript
// Egg Worker - タマゴ個体列挙専用Worker
import {
  type EggWorkerRequest,
  type EggWorkerResponse,
  type EggGenerationParams,
  type EnumeratedEggData,
  type EggResultsPayload,
  type EggCompletion,
  validateEggParams,
} from '@/types/egg';
import {
  initWasm,
  getWasm,
  isWasmReady,
} from '@/lib/core/wasm-interface';

interface InternalState {
  params: EggGenerationParams | null;
  running: boolean;
  stopRequested: boolean;
}

const state: InternalState = {
  params: null,
  running: false,
  stopRequested: false,
};

const ctx = self as typeof self & { onclose?: () => void };
const post = (message: EggWorkerResponse) => ctx.postMessage(message);

post({ type: 'READY', version: '1' });

ctx.onmessage = (ev: MessageEvent<EggWorkerRequest>) => {
  const msg = ev.data;
  (async () => {
    try {
      switch (msg.type) {
        case 'START_GENERATION':
          await handleStart(msg.params);
          break;
        case 'STOP':
          state.stopRequested = true;
          break;
        default:
          break;
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      post({ type: 'ERROR', message, category: 'RUNTIME', fatal: false });
    }
  })();
};

async function handleStart(params: EggGenerationParams) {
  if (state.running) return;
  
  const errors = validateEggParams(params);
  if (errors.length) {
    post({ type: 'ERROR', message: errors.join(', '), category: 'VALIDATION', fatal: false });
    return;
  }
  
  state.params = params;
  state.stopRequested = false;
  state.running = true;
  
  try {
    await ensureWasm();
    const runOutcome = executeEnumeration(params);
    postResults(runOutcome.results);
    post({ type: 'COMPLETE', payload: runOutcome.completion });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ type: 'ERROR', message, category: 'WASM_INIT', fatal: true });
  } finally {
    cleanupState();
  }
}

async function ensureWasm() {
  if (!isWasmReady()) await initWasm();
}

function executeEnumeration(params: EggGenerationParams) {
  const wasm = getWasm();
  if (!wasm.EggSeedEnumerator) {
    throw new Error('EggSeedEnumerator not exposed in WASM');
  }
  
  const results: EnumeratedEggData[] = [];
  let processedCount = 0;
  let filteredCount = 0;
  let reason: EggCompletion['reason'] = 'max-count';
  
  const startTime = performance.now();
  
  // ParentsIVs 構築
  const parentsIVs = new wasm.ParentsIVs();
  parentsIVs.male = params.parents.male;
  parentsIVs.female = params.parents.female;
  
  // GenerationConditions 構築
  const conditions = new wasm.GenerationConditions();
  conditions.has_nidoran_flag = params.conditions.hasNidoranFlag;
  conditions.everstone = buildEverstone(wasm, params.conditions.everstone);
  conditions.uses_ditto = params.conditions.usesDitto;
  conditions.allow_hidden_ability = params.conditions.allowHiddenAbility;
  conditions.female_parent_has_hidden = params.conditions.femaleParentHasHidden;
  conditions.reroll_count = params.conditions.rerollCount;
  conditions.trainer_ids = new wasm.TrainerIds(params.conditions.tid, params.conditions.sid);
  conditions.gender_ratio = new wasm.GenderRatio(
    params.conditions.genderRatio.threshold,
    params.conditions.genderRatio.genderless
  );
  
  // IndividualFilter 構築
  const filter = params.filter ? buildFilter(wasm, params.filter) : null;
  
  // EggSeedEnumerator 作成
  const enumerator = new wasm.EggSeedEnumerator(
    params.baseSeed,
    params.userOffset,
    params.count,
    conditions,
    parentsIVs,
    filter,
    params.considerNpcConsumption,
    params.gameMode
  );
  
  // 列挙ループ
  try {
    while (true) {
      if (state.stopRequested) {
        reason = 'stopped';
        break;
      }
      
      const rawData = enumerator.next_egg();
      if (!rawData) {
        reason = 'max-count';
        break;
      }
      
      processedCount++;
      
      // WASM から EnumeratedEggData を取得
      const eggData = parseEnumeratedEggData(rawData);
      if (eggData) {
        results.push(eggData);
        filteredCount++;
      }
    }
  } catch (e) {
    reason = 'error';
    throw e;
  } finally {
    enumerator.free();
    conditions.free();
    parentsIVs.free();
    if (filter) filter.free();
  }
  
  const completion: EggCompletion = {
    reason,
    processedCount,
    filteredCount,
    elapsedMs: performance.now() - startTime,
  };
  
  return { results, completion };
}

function buildEverstone(wasm: any, plan: any) {
  if (plan.type === 'none') {
    return wasm.EverstonePlan.None();
  } else {
    return wasm.EverstonePlan.Fixed(plan.nature);
  }
}

function buildFilter(wasm: any, filter: any) {
  const wasmFilter = new wasm.IndividualFilter();
  
  // IV範囲設定
  for (let i = 0; i < 6; i++) {
    const range = filter.ivRanges[i];
    wasmFilter.set_iv_range(i, range.min, range.max);
  }
  
  // Optional条件設定
  if (filter.nature !== undefined) {
    wasmFilter.nature = filter.nature;
  }
  if (filter.gender !== undefined) {
    wasmFilter.gender = genderToWasm(filter.gender);
  }
  if (filter.ability !== undefined) {
    wasmFilter.ability = filter.ability;
  }
  if (filter.shiny !== undefined) {
    wasmFilter.shiny = filter.shiny;
  }
  if (filter.hiddenPowerType !== undefined) {
    wasmFilter.hidden_power_type = filter.hiddenPowerType;
  }
  if (filter.hiddenPowerPower !== undefined) {
    wasmFilter.hidden_power_power = filter.hiddenPowerPower;
  }
  
  return wasmFilter;
}

function genderToWasm(gender: string): number {
  switch (gender) {
    case 'male': return 0;
    case 'female': return 1;
    case 'genderless': return 2;
    default: return 0;
  }
}

function parseEnumeratedEggData(raw: any): EnumeratedEggData | null {
  if (!raw) return null;
  
  try {
    const advance = Number(raw.advance);
    const egg = raw.egg;
    const isStable = raw.is_stable;
    
    const resolvedEgg = {
      ivs: [
        egg.ivs[0], egg.ivs[1], egg.ivs[2],
        egg.ivs[3], egg.ivs[4], egg.ivs[5]
      ] as IvSet,
      nature: egg.nature,
      gender: wasmGenderToDomain(egg.gender),
      ability: egg.ability,
      shiny: egg.shiny,
      pid: egg.pid,
      hiddenPower: parseHiddenPower(egg.hidden_power),
    };
    
    return {
      advance,
      egg: resolvedEgg,
      isStable,
    };
  } catch (e) {
    return null;
  }
}

function wasmGenderToDomain(wasmGender: number): 'male' | 'female' | 'genderless' {
  switch (wasmGender) {
    case 0: return 'male';
    case 1: return 'female';
    case 2: return 'genderless';
    default: return 'male';
  }
}

function parseHiddenPower(raw: any): HiddenPowerInfo {
  if (raw.type === 'unknown') {
    return { type: 'unknown' };
  } else {
    return {
      type: 'known',
      hpType: raw.hp_type,
      power: raw.power,
    };
  }
}

function postResults(results: EnumeratedEggData[]) {
  const payload: EggResultsPayload = { results };
  post({ type: 'RESULTS', payload });
}

function cleanupState() {
  state.running = false;
  state.params = null;
  state.stopRequested = false;
}

ctx.onclose = () => {
  cleanupState();
};

export {};
```

## 4. WorkerManager実装

### 4.1 ファイル: `src/lib/egg/egg-worker-manager.ts`

```typescript
// EggWorkerManager - Worker ライフサイクル管理とコールバック配信

import {
  type EggGenerationParams,
  type EggWorkerRequest,
  type EggWorkerResponse,
  type EggResultsPayload,
  type EggCompletion,
  type EggErrorCategory,
  validateEggParams,
  isEggWorkerResponse,
} from '@/types/egg';

type ResultsCb = (payload: EggResultsPayload) => void;
type CompleteCb = (c: EggCompletion) => void;
type ErrorCb = (msg: string, cat: EggErrorCategory, fatal: boolean) => void;

interface CallbackRegistry {
  results: ResultsCb[];
  complete: CompleteCb[];
  error: ErrorCb[];
}

export class EggWorkerManager {
  private worker: Worker | null = null;
  private callbacks: CallbackRegistry = { results: [], complete: [], error: [] };
  private running = false;
  private terminated = false;
  private currentRequestId: string | null = null;
  private status: 'idle' | 'running' | 'stopping' = 'idle';

  constructor(
    private readonly createWorker: () => Worker = () =>
      new Worker(new URL('@/workers/egg-worker.ts', import.meta.url), { type: 'module' }),
  ) {}

  // --- Public API ---
  start(params: EggGenerationParams): Promise<void> {
    if (this.running) {
      throw new Error('egg generation already running');
    }
    const validation = validateEggParams(params);
    if (validation.length) {
      return Promise.reject(new Error(validation.join(', ')));
    }
    const needsFreshWorker = this.terminated || this.worker === null;
    this.ensureWorker(needsFreshWorker);
    this.running = true;
    this.status = 'running';
    const rid = this.generateRequestId();
    this.currentRequestId = rid;

    const req: EggWorkerRequest = {
      type: 'START_GENERATION',
      params,
      requestId: rid,
    };
    this.worker!.postMessage(req);
    return Promise.resolve();
  }

  stop(): void {
    if (!this.running) return;
    this.status = 'stopping';
    const requestId = this.currentRequestId || undefined;
    this.worker?.postMessage({ type: 'STOP', requestId } satisfies EggWorkerRequest);
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
    }
    this.worker = null;
    this.running = false;
    this.terminated = true;
    this.status = 'idle';
    this.currentRequestId = null;
  }

  onResults(cb: ResultsCb) { this.callbacks.results.push(cb); return this; }
  onComplete(cb: CompleteCb) { this.callbacks.complete.push(cb); return this; }
  onError(cb: ErrorCb) { this.callbacks.error.push(cb); return this; }

  getStatus(): 'idle' | 'running' | 'stopping' {
    return this.status;
  }

  isRunning() { return this.running; }

  // --- Internal ---
  private ensureWorker(forceNew = false) {
    if (this.worker && (forceNew || this.terminated)) {
      this.worker.terminate();
      this.worker = null;
    }
    if (this.worker) return;
    this.worker = this.createWorker();
    this.terminated = false;
    this.worker.onmessage = (ev: MessageEvent) => this.handleMessage(ev.data);
    this.worker.onerror = () => {
      this.emitError('Worker error event', 'RUNTIME', true);
      this.terminate();
    };
  }

  private handleMessage(raw: unknown) {
    if (!isEggWorkerResponse(raw)) return;
    const msg: EggWorkerResponse = raw;
    switch (msg.type) {
      case 'READY':
        break; // noop
      case 'RESULTS':
        this.callbacks.results.forEach(cb => cb(msg.payload));
        break;
      case 'COMPLETE': {
        this.running = false;
        this.status = 'idle';
        this.currentRequestId = null;
        const completedWorker = this.worker;
        this.worker = null;
        this.terminated = true;
        this.callbacks.complete.forEach(cb => cb(msg.payload));
        completedWorker?.terminate();
        break;
      }
      case 'ERROR':
        this.emitError(msg.message, msg.category, msg.fatal);
        if (msg.fatal) {
          this.running = false;
          this.terminate();
        }
        break;
    }
  }

  private emitError(message: string, category: EggErrorCategory, fatal: boolean) {
    this.callbacks.error.forEach(cb => cb(message, category, fatal));
  }

  private generateRequestId(): string {
    return 'egg-' + Math.random().toString(36).slice(2, 10);
  }
}
```

## 5. UIコンポーネント設計

### 5.1 コンポーネント構造

```
EggBWPanel (レイアウト親コンポーネント)
├── EggParamsCard (パラメータ入力)
│   ├── 基本設定セクション
│   │   ├── 初期Seed入力
│   │   ├── 開始advance入力
│   │   ├── 列挙上限入力
│   │   └── GameMode選択
│   ├── 親個体情報セクション
│   │   ├── ♂親IV入力 (6値)
│   │   ├── ♀親IV入力 (6値)
│   │   └── メタモン利用チェックボックス
│   ├── 生成条件セクション
│   │   ├── かわらずのいし設定
│   │   ├── 性別比設定
│   │   ├── 種族特性（通常/ニドラン系/バルビート系）
│   │   ├── 夢特性許可チェックボックス
│   │   ├── 親♀夢特性チェックボックス
│   │   ├── PIDリロール回数（国際孵化）
│   │   └── TID/SID入力
│   └── その他設定セクション
│       └── NPC消費考慮チェックボックス
├── EggFilterCard (フィルター設定)
│   ├── 個体値範囲フィルター (6値×2スライダー)
│   ├── 性格フィルター (選択)
│   ├── 性別フィルター (選択)
│   ├── 特性フィルター (選択)
│   ├── 色違いフィルター (選択)
│   └── めざパフィルター (タイプ・威力)
├── EggRunCard (実行制御)
│   ├── 開始/停止ボタン
│   ├── ステータス表示
│   └── 進捗表示
└── EggResultsCard (結果表示)
    ├── 結果制御バー (エクスポート等)
    └── 結果テーブル
        ├── advance列
        ├── IV列 (6値)
        ├── 性格列
        ├── 性別列
        ├── 特性列
        ├── 色違い列
        ├── PID列
        ├── めざパ列
        └── 安定性列 (NPC消費時)
```

### 5.2 状態管理 (Zustand)

#### 5.2.1 ストア: `src/store/egg-store.ts`

```typescript
import { create } from 'zustand';
import {
  type EggGenerationParams,
  type EggGenerationParamsHex,
  type EnumeratedEggData,
  type EggCompletion,
  hexParamsToEggParams,
  validateEggParams,
} from '@/types/egg';
import { EggWorkerManager } from '@/lib/egg/egg-worker-manager';

export type EggStatus = 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'error';

interface EggStore {
  // パラメータ
  draftParams: EggGenerationParamsHex;
  params: EggGenerationParams | null;
  validationErrors: string[];
  
  // 実行状態
  status: EggStatus;
  workerManager: EggWorkerManager | null;
  
  // 結果
  results: EnumeratedEggData[];
  lastCompletion: EggCompletion | null;
  errorMessage: string | null;
  
  // アクション
  updateDraftParams: (updates: Partial<EggGenerationParamsHex>) => void;
  validateDraft: () => void;
  startGeneration: () => Promise<void>;
  stopGeneration: () => void;
  clearResults: () => void;
  reset: () => void;
}

const DEFAULT_DRAFT: EggGenerationParamsHex = {
  baseSeedHex: '0',
  userOffsetHex: '0',
  count: 100,
  conditions: {
    hasNidoranFlag: false,
    everstone: { type: 'none' },
    usesDitto: false,
    allowHiddenAbility: false,
    femaleParentHasHidden: false,
    rerollCount: 0,
    tid: 0,
    sid: 0,
    genderRatio: {
      threshold: 127,
      genderless: false,
    },
  },
  parents: {
    male: [31, 31, 31, 31, 31, 31],
    female: [31, 31, 31, 31, 31, 31],
  },
  filter: null,
  considerNpcConsumption: false,
  gameMode: 1, // BwContinue
};

export const useEggStore = create<EggStore>((set, get) => ({
  draftParams: DEFAULT_DRAFT,
  params: null,
  validationErrors: [],
  status: 'idle',
  workerManager: null,
  results: [],
  lastCompletion: null,
  errorMessage: null,
  
  updateDraftParams: (updates) => {
    set((state) => ({
      draftParams: { ...state.draftParams, ...updates },
    }));
  },
  
  validateDraft: () => {
    const draft = get().draftParams;
    try {
      const params = hexParamsToEggParams(draft);
      const errors = validateEggParams(params);
      set({ validationErrors: errors, params: errors.length === 0 ? params : null });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ validationErrors: [message], params: null });
    }
  },
  
  startGeneration: async () => {
    const { params, workerManager: existingManager } = get();
    if (!params) {
      set({ errorMessage: 'Invalid parameters' });
      return;
    }
    
    // Worker初期化
    const manager = existingManager || new EggWorkerManager();
    
    manager
      .onResults((payload) => {
        set((state) => ({
          results: [...state.results, ...payload.results],
        }));
      })
      .onComplete((completion) => {
        set({
          status: 'completed',
          lastCompletion: completion,
        });
      })
      .onError((message, category, fatal) => {
        set({
          status: fatal ? 'error' : get().status,
          errorMessage: message,
        });
      });
    
    set({
      workerManager: manager,
      status: 'starting',
      results: [],
      lastCompletion: null,
      errorMessage: null,
    });
    
    try {
      await manager.start(params);
      set({ status: 'running' });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ status: 'error', errorMessage: message });
    }
  },
  
  stopGeneration: () => {
    const { workerManager } = get();
    if (workerManager) {
      set({ status: 'stopping' });
      workerManager.stop();
    }
  },
  
  clearResults: () => {
    set({ results: [], lastCompletion: null, errorMessage: null });
  },
  
  reset: () => {
    const { workerManager } = get();
    if (workerManager) {
      workerManager.terminate();
    }
    set({
      draftParams: DEFAULT_DRAFT,
      params: null,
      validationErrors: [],
      status: 'idle',
      workerManager: null,
      results: [],
      lastCompletion: null,
      errorMessage: null,
    });
  },
}));
```

### 5.3 パネル実装サンプル

#### 5.3.1 `src/components/egg/EggBWPanel.tsx`

```typescript
import React from 'react';
import { EggParamsCard } from './EggParamsCard';
import { EggFilterCard } from './EggFilterCard';
import { EggRunCard } from './EggRunCard';
import { EggResultsCard } from './EggResultsCard';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { LEFT_COLUMN_WIDTH_CLAMP } from '@/components/layout/constants';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';

export function EggBWPanel() {
  const { isStack, uiScale } = useResponsiveLayout();
  const sizes = getResponsiveSizes(uiScale);

  if (isStack) {
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        <div className="flex-none">
          <EggRunCard />
        </div>
        <div className="flex-none">
          <EggParamsCard />
        </div>
        <div className="flex-none">
          <EggFilterCard />
        </div>
        <div className="flex-1 min-h-0">
          <EggResultsCard />
        </div>
      </div>
    );
  }

  // デスクトップ: 2カラム
  return (
    <div className={`flex flex-col ${sizes.gap} max-w-full h-full min-h-0 min-w-fit overflow-hidden`}>
      <div className={`flex ${sizes.gap} max-w-full flex-1 min-h-0 min-w-fit overflow-hidden`}>
        {/* Left Column */}
        <div
          className={`flex-1 flex flex-col ${sizes.gap} min-w-0 overflow-y-auto`}
          style={{
            minHeight: 0,
            width: LEFT_COLUMN_WIDTH_CLAMP,
            flex: `0 0 ${LEFT_COLUMN_WIDTH_CLAMP}`
          }}
        >
          <div className="flex-none">
            <EggRunCard />
          </div>
          <div className="flex-1 min-h-0">
            <EggParamsCard />
          </div>
          <div className="flex-none">
            <EggFilterCard />
          </div>
        </div>
        {/* Right Column */}
        <div className={`flex-1 flex flex-col ${sizes.gap} min-w-0 ${sizes.columnWidth} overflow-y-auto`} style={{ minHeight: 0 }}>
          <div className="flex-1 min-h-0">
            <EggResultsCard />
          </div>
        </div>
      </div>
    </div>
  );
}
```

## 6. テスト戦略

### 6.1 単体テスト

#### 6.1.1 型・バリデーション (`src/types/egg.test.ts`)
- パラメータ変換（hex → bigint）
- バリデーション関数
- IV範囲チェック

#### 6.1.2 WorkerManager (`src/lib/egg/egg-worker-manager.test.ts`)
- start/stop/terminate の動作
- コールバック配信
- エラーハンドリング

### 6.2 統合テスト

#### 6.2.1 Worker統合 (`src/test/egg/egg-worker.test.ts`)
- WASM初期化
- EggSeedEnumerator 呼び出し
- 結果パース

### 6.3 E2Eテスト

#### 6.3.1 UI操作 (Playwright)
- パラメータ入力
- フィルター設定
- 実行・停止
- 結果表示

## 7. 実装順序

### Phase 1: 型定義とWorker基盤
1. `src/types/egg.ts` 作成
2. `src/workers/egg-worker.ts` 作成
3. `src/lib/egg/egg-worker-manager.ts` 作成
4. 単体テスト作成・実行

### Phase 2: 状態管理
1. `src/store/egg-store.ts` 作成
2. ストアテスト作成・実行

### Phase 3: UIコンポーネント
1. `src/components/egg/EggBWPanel.tsx` 作成
2. `src/components/egg/EggParamsCard.tsx` 作成
3. `src/components/egg/EggFilterCard.tsx` 作成
4. `src/components/egg/EggRunCard.tsx` 作成
5. `src/components/egg/EggResultsCard.tsx` 作成

### Phase 4: 統合とテスト
1. WASM統合テスト
2. E2Eテスト
3. ドキュメント更新

## 8. 参考資料

- `/spec/implementation/06-egg-iv-handling.md` - タマゴIV仕様
- `wasm-pkg/src/egg_seed_enumerator.rs` - Rust実装
- `wasm-pkg/src/egg_iv.rs` - タマゴIV計算
- `src/workers/generation-worker.ts` - 既存Worker実装パターン
- `src/lib/generation/generation-worker-manager.ts` - 既存Manager実装パターン
- `src/components/layout/GenerationPanel.tsx` - 既存Panel実装パターン

## 9. 注意事項

### 9.1 WASM境界
- すべてのWASM呼び出しは `src/lib/core/wasm-interface.ts` 経由
- メモリ管理: WASM オブジェクトの `.free()` 呼び出し必須

### 9.2 BigInt シリアライゼーション
- Worker通信では BigInt を Number に変換
- UI では16進数文字列で管理

### 9.3 Unknown IV (32) の入力仕様

#### 9.3.1 親個体IV入力
- **基本入力範囲**: 0-31（数値入力フィールド）
- **Unknown (32) の設定方法**: チェックボックスによる切り替え
  - チェックボックス OFF: 数値入力フィールドで 0-31 を入力可能
  - チェックボックス ON: 入力フィールドは無効化され、値は自動的に 32 (Unknown) に設定

```typescript
/**
 * 親IV入力の状態
 */
export interface ParentIvInputState {
  value: number;        // 0-31 の値（チェック時は無効）
  isUnknown: boolean;   // true の場合、実際の値は 32 (Unknown)
}

/**
 * IvSetへの変換
 */
function toIvSet(inputs: ParentIvInputState[]): IvSet {
  return inputs.map(input => input.isUnknown ? 32 : input.value) as IvSet;
}
```

#### 9.3.2 フィルター IV範囲入力
- **基本入力範囲**: 0-31（範囲スライダーまたは数値入力）
- **任意範囲指定モード**: チェックボックスで有効化
  - チェックボックス OFF: 範囲入力 0-31 のみ（Unknown を含む個体はフィルター不可）
  - チェックボックス ON: 範囲上限が自動的に 32 に強制設定され、Unknown を許可

```typescript
/**
 * フィルターIV範囲入力の状態
 */
export interface FilterIvRangeInputState {
  min: number;           // 0-31
  max: number;           // 0-31（includeUnknown=true時は32に強制）
  includeUnknown: boolean; // Unknownを含める場合はtrue
}

/**
 * StatRangeへの変換
 */
function toStatRange(input: FilterIvRangeInputState): StatRange {
  return {
    min: input.min,
    max: input.includeUnknown ? 32 : input.max,
  };
}
```

#### 9.3.3 UI表示
- Unknown IV は結果テーブルで `?` または `不明` として表示
- めざめるパワーは Unknown IV を含む場合 `?/?` と表示

### 9.4 パフォーマンス
- 大量個体生成時はバッチ処理
- 結果更新は適度な間隔でUI反映

### 9.5 国際化
- すべてのUI文字列は i18n 対応
- `src/lib/i18n/strings/egg-*.ts` にラベル定義

## 10. 拡張設計: 起動時間関連機能

### 10.1 概要
起動時間に関連する機能として、以下の2つの異なるモードが必要となる:

| モード | 目的 | 入力 | 出力 |
|--------|------|------|------|
| **起動時間列挙モード** | 指定した起動時間候補（Timer0/VCount範囲）から個体を列挙 | 起動時間パラメータ + フィルター | 各候補の個体一覧 |
| **起動時間検索モード** | 条件を満たす個体が得られる起動時間を検索 | 目標条件 + 日時範囲 + 消費範囲 | 条件を満たす起動時間リスト |

### 10.2 起動時間列挙モード（Boot Timing Enumeration）

#### 10.2.1 概要
起動時間から初期Seedを導出し、複数のTimer0/VCount候補に対してタマゴ個体生成を実行する機能。
既存の GenerationPanel の boot-timing モードと同様のアーキテクチャを採用する。

#### 10.2.2 アーキテクチャ図
```
┌─────────────────────────────────────────────────────────────┐
│                    EggBWPanel                                │
│  ┌────────────────┐  ┌───────────────────────────────────┐  │
│  │ Mode Switch    │  │  起動時間パラメータ / Seed入力    │  │
│  │ [LCG] [Boot]   │  │  (BootTimingDraft or SeedHex)     │  │
│  └────────────────┘  └───────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │              egg-store.ts                       │
        │  seedSourceMode: 'lcg' | 'boot-timing'         │
        │  bootTimingDraft?: BootTimingDraft             │
        └────────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         │ if seedSourceMode === 'boot-timing'            │
         ▼                                                │
┌────────────────────────────────┐                        │
│ deriveBootTimingEggSeedJobs()  │                        │
│ - Timer0/VCount範囲からSeed導出 │                        │
│ - 複数のDerivedSeedJobを生成    │                        │
└────────────────┬───────────────┘                        │
                 │                                        │
                 ▼                                        ▼
        ┌────────────────────────────────────────────────────┐
        │              EggWorkerManager                       │
        │  - 単一Seedモード: 1つのジョブを実行               │
        │  - 起動時間モード: 複数ジョブを順次実行            │
        │    (DerivedSeedRunState で進捗管理)               │
        └────────────────────────┬───────────────────────────┘
                                 │
                                 ▼ (同一Worker)
                    ┌─────────────────────────┐
                    │    egg-worker.ts         │
                    │  EggSeedEnumerator使用   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  EggSeedEnumerator      │
                    │     (WASM/Rust)         │
                    └─────────────────────────┘
```

#### 10.2.3 処理フロー

1. **UI入力**: ユーザーが `seedSourceMode = 'boot-timing'` を選択
2. **BootTimingDraft収集**: タイムスタンプ、Timer0/VCount範囲、MACアドレス等
3. **Seed導出**: `deriveBootTimingEggSeedJobs()` で候補Seedリストを生成
4. **順次実行**: EggWorkerManager が DerivedSeedRunState を管理し、各Seedに対して個体生成
5. **結果集約**: 全Seed候補の結果を統合して表示

#### 10.2.4 型定義拡張

```typescript
/**
 * 起動時間検索用パラメータ (UI入力)
 */
export interface EggBootTimingDraft {
  timestampIso?: string;
  keyMask: number;
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
  romRegion: ROMRegion;
  hardware: Hardware;
  macAddress: readonly [number, number, number, number, number, number];
}

/**
 * 導出されたSeedジョブ
 */
export interface DerivedEggSeedJob {
  params: EggGenerationParams;
  metadata: {
    seedSourceMode: 'boot-timing';
    derivedSeedIndex: number;
    timer0: number;
    vcount: number;
    keyMask: number;
    bootTimestampIso: string;
    macAddress: readonly [number, number, number, number, number, number];
    seedSourceSeedHex: string;
  };
}

/**
 * 起動時間検索の進捗状態
 */
export interface DerivedEggSeedRunState {
  readonly jobs: DerivedEggSeedJob[];
  readonly cursor: number;
  readonly total: number;
  readonly aggregate: {
    processedCount: number;
    filteredCount: number;
    elapsedMs: number;
  };
  readonly abortRequested: boolean;
}
```

### 10.3 起動時間列挙の実装方針
- 既存の `src/lib/generation/boot-timing-derivation.ts` のパターンを踏襲
- `src/lib/egg/boot-timing-egg-derivation.ts` として同様の機能を実装
- EggWorkerManager に `startBootTimingGeneration()` メソッドを追加
- 結果テーブルに Timer0/VCount 情報を表示可能にする

### 10.4 起動時間検索モード（Boot Timing Search）- SearchPanel類似機能

#### 10.4.1 概要
SearchPanel と類似の機能で、一定期間・一定消費数範囲内で条件を満たす起動時刻が存在するかを検索する。
これは「起動時間列挙」とは逆方向の検索であり、目標個体条件から起動時間を逆算する。

#### 10.4.2 入力と出力
- **入力**:
  - 目標個体条件（IV、性格、性別、特性、色違い等）
  - 日時範囲（開始日時 ～ 終了日時）
  - 消費範囲（最小消費数 ～ 最大消費数）
  - Timer0/VCount 範囲
  - 親個体条件
- **出力**: 
  - 条件を満たす起動時間・Timer0・VCount・消費数のリスト

#### 10.4.3 アーキテクチャ図
```
┌─────────────────────────────────────────────────────────────┐
│                  EggSearchPanel (将来実装)                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 検索条件入力                                           │  │
│  │ - 目標個体条件 (IV, 性格, 性別, 特性, 色違い等)        │  │
│  │ - 日時範囲 (開始日時 ～ 終了日時)                      │  │
│  │ - 消費範囲 (最小消費 ～ 最大消費)                      │  │
│  │ - Timer0/VCount 範囲                                   │  │
│  │ - 親個体条件                                           │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────┐
        │           egg-search-store.ts                   │
        │  - searchConditions: EggSearchConditions        │
        │  - dateTimeRange: DateTimeRange                │
        │  - consumptionRange: { min, max }              │
        └────────────────────────┬───────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │         EggSearchWorkerManager                  │
        │  - 日時範囲を分割して並列検索                   │
        │  - 条件を満たすSeedを収集                       │
        └────────────────────────┬───────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │          egg-search-worker.ts                   │
        │  - 各日時候補に対してSeedを計算                 │
        │  - 指定消費範囲で個体生成                       │
        │  - 条件マッチングを実行                         │
        └────────────────────────┬───────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  EggSeedEnumerator (WASM) + フィルタリング      │
        │  - 消費範囲内で条件を満たす個体を検索           │
        └────────────────────────────────────────────────┘
```

#### 10.4.4 検索フロー

1. **条件入力**: 目標個体条件、日時範囲、消費範囲を入力
2. **検索空間構築**: 日時×Timer0×VCount の組み合わせを列挙
3. **並列検索**: 各候補に対してSeed計算→個体生成→条件マッチング
4. **結果収集**: 条件を満たす起動時間・消費数のリストを生成
5. **結果表示**: 発見した起動時間候補を表示

#### 10.4.5 型定義

```typescript
/**
 * タマゴ検索条件
 */
export interface EggSearchConditions {
  targetFilter: EggIndividualFilter;  // 目標個体のフィルター条件
  parentConditions: EggGenerationConditions;
  parents: ParentsIVs;
}

/**
 * 検索範囲設定
 */
export interface EggSearchRange {
  dateTimeRange: {
    start: string;  // ISO形式
    end: string;
  };
  consumptionRange: {
    min: number;
    max: number;
  };
  timer0Range: { min: number; max: number };
  vcountRange: { min: number; max: number };
}

/**
 * 検索結果
 */
export interface EggSearchResult {
  bootTimestamp: string;
  timer0: number;
  vcount: number;
  seed: bigint;
  consumption: number;  // 何消費目で条件を満たすか
  matchedEgg: ResolvedEgg;
}
```

#### 10.4.6 実装方針（将来実装）
- 既存の SearchPanel のアーキテクチャを参考に設計
- 別途 `EggSearchPanel` として独立実装（EggBWPanel とは別Panel）
- 専用の `egg-search-worker.ts` と `EggSearchWorkerManager` を用意
- 並列処理による検索高速化

## 11. 拡張設計: BW2版 EggPanel

### 11.1 概要
BW2 のタマゴ生成ロジックは BW とは**根本的に異なる**ため、WASM レイヤーから完全に独立した実装が必要となる。
BW2 用の `EggBW2SeedEnumerator` (仮称) は未実装であり、将来的に独立して開発される予定。

### 11.2 BW と BW2 のロジック差異

| 項目 | BW | BW2 |
|------|-----|------|
| **LCG Seed 決定** | 既存ロジック | **完全に異なる** (未実装) |
| **個体値決定** | `EggSeedEnumerator` 内で一体的に処理 | **独立したインタフェース** (未実装) |
| **PID 決定** | `EggSeedEnumerator` 内で一体的に処理 | **独立したインタフェース** (未実装) |
| **WASM 実装** | `EggSeedEnumerator` | **`EggBW2IVGenerator` + `EggBW2PIDGenerator`** (仮称、未実装) |

### 11.3 アーキテクチャ図（将来構想）

```
┌─────────────────────────────────────────────────────────────┐
│                      App.tsx                                 │
│  ┌────────────────┐        ┌────────────────┐               │
│  │   EggBWPanel   │        │  EggBW2Panel   │               │
│  │  (BW専用UI)    │        │  (BW2専用UI)   │               │
│  └───────┬────────┘        └───────┬────────┘               │
│          │                         │                         │
│          ▼                         ▼                         │
│  ┌───────────────┐         ┌───────────────┐                │
│  │ EggBWStore    │         │ EggBW2Store   │                │
│  │ (BW専用状態)  │         │ (BW2専用状態) │                │
│  └───────┬───────┘         └───────┬───────┘                │
│          │                         │                         │
│          ▼                         ▼                         │
│  ┌────────────────┐        ┌─────────────────┐              │
│  │ EggBWWorker    │        │  EggBW2Worker   │              │
│  │ Manager       │        │  Manager        │              │
│  └───────┬────────┘        └───────┬─────────┘              │
│          │                         │                         │
│          ▼                         ▼                         │
│  ┌────────────────┐        ┌─────────────────┐              │
│  │ egg-bw-worker  │        │ egg-bw2-worker  │              │
│  └───────┬────────┘        └───────┬─────────┘              │
└──────────┼─────────────────────────┼────────────────────────┘
           │                         │
           ▼                         ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│ EggSeedEnumerator   │    │ EggBW2IVGenerator +             │
│ (BW用、既存)        │    │ EggBW2PIDGenerator              │
│                     │    │ (BW2用、未実装)                  │
└─────────────────────┘    └─────────────────────────────────┘
```

### 11.4 共通化と差分の方針

#### 11.4.1 共通化可能なコンポーネント（UI層のみ）
| レイヤー | コンポーネント | 共通化可否 |
|---------|---------------|-----------|
| UI | EggResultsCard | ⚠️ 一部共通化可能（結果表示形式が同じ場合） |
| UI | EggRunCard | ⚠️ 一部共通化可能（開始/停止UIは共通） |
| UI | 基本レイアウト | ⚠️ スタイルは共通化可能 |
| 型定義 | ResolvedEgg (結果型) | ⚠️ 出力形式が揃えば共通化可能 |

#### 11.4.2 分離が必要なコンポーネント
| レイヤー | コンポーネント | 分離理由 |
|---------|---------------|---------|
| **WASM** | Enumerator/Generator | **ロジックが根本的に異なる** |
| **Worker** | egg-worker.ts | **異なるWASMを呼び出す** |
| **Manager** | WorkerManager | **異なるWorkerを管理** |
| **Store** | Zustand ストア | **パラメータ構造が異なる可能性** |
| **UI** | EggParamsCard | **入力パラメータが異なる** |
| **UI** | EggFilterCard | **フィルター条件が異なる可能性** |

### 11.5 パラメータの流用可能性

一部のUIパラメータは流用可能だが、バックエンドへの渡し方は完全に異なる:

```typescript
// BW と BW2 で流用可能なパラメータ（UI入力層）
interface CommonEggUIParams {
  tid: number;
  sid: number;
  // 親個体条件の一部
}

// BW 専用パラメータ
interface EggBWParams extends CommonEggUIParams {
  // BW固有のパラメータ
}

// BW2 専用パラメータ（将来定義）
interface EggBW2Params extends CommonEggUIParams {
  memoryLink: boolean;
  // BW2固有のパラメータ（未定）
}
```

### 11.6 BW2 WASM インタフェース（将来構想）

BW2 では個体値生成と性格値生成が独立したインタフェースを持つ予定:

```typescript
// BW2 個体値生成器（将来実装予定）
interface EggBW2IVGenerator {
  // BW2固有の個体値生成ロジック
  generateIVs(seed: bigint, params: EggBW2IVParams): IvSet;
}

// BW2 性格値生成器（将来実装予定）
interface EggBW2PIDGenerator {
  // BW2固有のPID生成ロジック
  generatePID(seed: bigint, params: EggBW2PIDParams): PIDResult;
}

// これらは EggSeedEnumerator (BW用) とは完全に異なる実装となる
```

### 11.7 実装順序（将来）

1. **Phase A**: BW2 WASM ロジックの設計・仕様策定
2. **Phase B**: `EggBW2IVGenerator`, `EggBW2PIDGenerator` の Rust 実装
3. **Phase C**: `egg-bw2-worker.ts`, `EggBW2WorkerManager` の TypeScript 実装
4. **Phase D**: `EggBW2Store`, `EggBW2Panel` の UI 実装
5. **Phase E**: テスト・ドキュメント更新

### 11.8 注意事項

- BW2 の WASM 実装は**未実装**であり、本仕様書は将来的なアーキテクチャ構想を示すもの
- BW と BW2 で `EggSeedEnumerator` を共有する設計は**採用しない**
- BW2 実装時には、WASM インタフェースの詳細仕様を別途策定する必要がある
