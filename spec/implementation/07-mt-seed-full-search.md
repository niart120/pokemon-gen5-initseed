# MT Seed 32bit全探索 実装設計

## 概要

MT消費数と検索条件（個体値範囲/めざパタイプ/めざパ威力）が与えられたとき、メルセンヌツイスタ（MT19937）のSeed空間 $2^{32}$ を全探索し、所定の個体値パターンを生成するMT Seedを探索する機能の実装仕様を定義する。

## 背景

BW/BW2の孵化乱数における個体値決定は、初期Seedから導出されるMT Seedを起点としたMT19937乱数列によって行われる。目的の個体値パターンを得るには、対応するMT Seedを特定する必要がある。

- MT Seed空間: $2^{32}$ = 約43億パターン
- GPU/CPU両対応による高速化が必要

## 機能要件

### 入力パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `mtAdvances` | `number` | MT消費数（0以上の整数） |
| `ivFilter` | `IvSearchFilter` | 個体値フィルター（IV範囲/めざパ条件） |

### 出力

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `mtSeed` | `number` | 条件を満たすMT Seed (u32) |
| `ivSet` | `IvSet` | 生成される個体値 |
| `hiddenPower` | `HiddenPowerInfo` | めざめるパワー情報 |

### 検索フロー

```
検索条件入力
    ↓
IVコードリスト生成（TS側）
    ↓
[1024件超過チェック]──超過──→ エラーメッセージ表示
    ↓ OK
ジョブ計画（GPU/CPU別）
    ↓
WorkerManager
    ↓
┌───────────────────────┐
│  GPU Worker           │  ← GPU利用可能時
│  (WebGPU Compute)     │
└───────────────────────┘
         or
┌───────────────────────┐
│  CPU並列Worker        │  ← GPUフォールバック時
│  (WASM + SharedArrayBuffer)│
└───────────────────────┘
    ↓
結果集約・表示
```

---

## IVコード変換仕様

### IVコード圧縮表現

6ステータスの個体値（各0-31）を30bitの整数値にエンコードする。

```typescript
type IvCode = number; // u30 (0 ~ 1,073,741,823)

/**
 * IVセットをIVコードにエンコード
 * 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]
 */
function encodeIvCode(ivs: IvSet): IvCode {
  return (ivs[0] << 25) | (ivs[1] << 20) | (ivs[2] << 15) 
       | (ivs[3] << 10) | (ivs[4] << 5)  | ivs[5];
}

/**
 * IVコードをIVセットにデコード
 */
function decodeIvCode(code: IvCode): IvSet {
  return [
    (code >> 25) & 0x1F,
    (code >> 20) & 0x1F,
    (code >> 15) & 0x1F,
    (code >> 10) & 0x1F,
    (code >> 5)  & 0x1F,
    code & 0x1F,
  ];
}
```

### 検索条件からIVコードリストへの変換

```typescript
interface IvSearchFilter {
  ivRanges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
  hiddenPowerType?: number;    // 0-15
  hiddenPowerPower?: number;   // 30-70
}

interface IvCodeGenerationResult {
  success: true;
  ivCodes: IvCode[];
} | {
  success: false;
  error: 'TOO_MANY_COMBINATIONS';
  estimatedCount: number;
}

const MAX_IV_CODES = 1024;

/**
 * 検索条件からIVコードリストを生成
 * 
 * @returns 成功時はIVコード配列、失敗時はエラー情報
 */
function generateIvCodes(filter: IvSearchFilter): IvCodeGenerationResult {
  const candidates: IvCode[] = [];
  
  // 6重ループで全組み合わせを列挙
  for (let hp = filter.ivRanges[0].min; hp <= filter.ivRanges[0].max; hp++) {
    for (let atk = filter.ivRanges[1].min; atk <= filter.ivRanges[1].max; atk++) {
      for (let def = filter.ivRanges[2].min; def <= filter.ivRanges[2].max; def++) {
        for (let spa = filter.ivRanges[3].min; spa <= filter.ivRanges[3].max; spa++) {
          for (let spd = filter.ivRanges[4].min; spd <= filter.ivRanges[4].max; spd++) {
            for (let spe = filter.ivRanges[5].min; spe <= filter.ivRanges[5].max; spe++) {
              const ivs: IvSet = [hp, atk, def, spa, spd, spe];
              
              // めざパフィルター適用
              if (!matchesHiddenPowerFilter(ivs, filter)) {
                continue;
              }
              
              candidates.push(encodeIvCode(ivs));
              
              // 上限チェック（早期終了）
              if (candidates.length > MAX_IV_CODES) {
                return {
                  success: false,
                  error: 'TOO_MANY_COMBINATIONS',
                  estimatedCount: estimateTotalCombinations(filter),
                };
              }
            }
          }
        }
      }
    }
  }
  
  return { success: true, ivCodes: candidates };
}

/**
 * めざパフィルターとのマッチング判定
 */
function matchesHiddenPowerFilter(ivs: IvSet, filter: IvSearchFilter): boolean {
  if (filter.hiddenPowerType === undefined && filter.hiddenPowerPower === undefined) {
    return true;
  }
  
  const hp = calculateHiddenPower(ivs);
  
  if (filter.hiddenPowerType !== undefined && hp.type !== filter.hiddenPowerType) {
    return false;
  }
  if (filter.hiddenPowerPower !== undefined && hp.power !== filter.hiddenPowerPower) {
    return false;
  }
  
  return true;
}
```

### エラーハンドリング

1024件を超過する場合、検索を開始せずにエラーメッセージを表示する。

```typescript
const errorMessages = {
  TOO_MANY_COMBINATIONS: (count: number) => 
    `検索条件が広すぎます。個体値の組み合わせが${count.toLocaleString()}件あります（上限: ${MAX_IV_CODES}件）。条件を絞り込んでください。`,
};
```

---

## ジョブ定義

### MtSeedSearchJob型

```typescript
interface MtSeedSearchJob {
  /** 検索範囲（閉区間）[start, end] */
  searchRange: {
    start: number;  // u32
    end: number;    // u32
  };
  
  /** 検索対象IVコードリスト */
  ivCodes: IvCode[];
  
  /** MT消費数 */
  mtAdvances: number;
  
  /** ジョブID（進捗追跡用） */
  jobId: number;
}

/**
 * 単一のマッチ結果
 */
interface MtSeedMatch {
  /** 発見されたMT Seed */
  mtSeed: number;
  
  /** 対応するIVコード */
  ivCode: IvCode;
}

/**
 * ジョブ単位の検索結果
 * 1ジョブに対して複数のマッチが発生しうるため配列で保持
 */
interface MtSeedSearchResult {
  /** ジョブID */
  jobId: number;
  
  /** マッチしたSeed/IVコードのペア配列 */
  matches: MtSeedMatch[];
}
```

### ジョブ計画インターフェース

```typescript
interface JobPlannerConfig {
  /** 全探索範囲 */
  fullRange: { start: number; end: number };  // 通常 [0, 0xFFFFFFFF]
  
  /** IVコードリスト */
  ivCodes: IvCode[];
  
  /** MT消費数 */
  mtAdvances: number;
}

interface JobPlan {
  jobs: MtSeedSearchJob[];
  totalSearchSpace: number;
  estimatedTimeMs: number;
}
```

---

## ジョブ分割ストラテジー

### GPU向け分割

GPU計算では、デバイス制約の範囲内で可能な限り広い検索範囲を単一ジョブに割り当てる。

```typescript
interface GpuJobPlannerConfig extends JobPlannerConfig {
  /** WebGPUデバイス制約 */
  deviceLimits: {
    maxComputeWorkgroupsPerDimension: number;
    maxStorageBufferBindingSize: number;
  };
  
  /** ワークグループサイズ */
  workgroupSize: number;
}

function planGpuJobs(config: GpuJobPlannerConfig): JobPlan {
  const { fullRange, ivCodes, mtAdvances, deviceLimits, workgroupSize } = config;
  
  // 1ディスパッチあたりの最大処理数
  const maxSeedsPerDispatch = 
    deviceLimits.maxComputeWorkgroupsPerDimension * workgroupSize;
  
  // IVコードバッファサイズ制約
  const ivCodeBufferSize = ivCodes.length * 4; // 4 bytes per IvCode
  const maxIvCodesPerDispatch = Math.floor(
    deviceLimits.maxStorageBufferBindingSize / 4
  );
  
  // ジョブ分割
  const jobs: MtSeedSearchJob[] = [];
  let cursor = fullRange.start;
  let jobId = 0;
  
  while (cursor <= fullRange.end) {
    const rangeSize = Math.min(
      maxSeedsPerDispatch,
      fullRange.end - cursor + 1
    );
    
    jobs.push({
      searchRange: { start: cursor, end: cursor + rangeSize - 1 },
      ivCodes,
      mtAdvances,
      jobId: jobId++,
    });
    
    cursor += rangeSize;
  }
  
  return {
    jobs,
    totalSearchSpace: fullRange.end - fullRange.start + 1,
    estimatedTimeMs: estimateGpuTime(jobs.length, ivCodes.length),
  };
}
```

### CPU向け分割

CPU並列計算では、論理コア数に基づいてSeed空間を均等分割する。

```typescript
interface CpuJobPlannerConfig extends JobPlannerConfig {
  /** 使用するWorker数（通常 navigator.hardwareConcurrency） */
  workerCount: number;
}

function planCpuJobs(config: CpuJobPlannerConfig): JobPlan {
  const { fullRange, ivCodes, mtAdvances, workerCount } = config;
  
  const totalRange = fullRange.end - fullRange.start + 1;
  const rangePerWorker = Math.ceil(totalRange / workerCount);
  
  const jobs: MtSeedSearchJob[] = [];
  let cursor = fullRange.start;
  
  for (let i = 0; i < workerCount && cursor <= fullRange.end; i++) {
    const rangeEnd = Math.min(cursor + rangePerWorker - 1, fullRange.end);
    
    jobs.push({
      searchRange: { start: cursor, end: rangeEnd },
      ivCodes,
      mtAdvances,
      jobId: i,
    });
    
    cursor = rangeEnd + 1;
  }
  
  return {
    jobs,
    totalSearchSpace: totalRange,
    estimatedTimeMs: estimateCpuTime(totalRange, ivCodes.length, workerCount),
  };
}
```

---

## Worker構成

### 通信フロー

```
┌─────────────────────────────────────────────────────────────────┐
│  Main Thread                                                    │
│  ┌─────────────────┐                                           │
│  │ MtSeedSearchManager                                         │
│  │  - ジョブ計画                                                │
│  │  - Worker管理                                                │
│  │  - 結果集約                                                  │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐     ┌─────────────────┐                  │
│  │ GPU Path        │     │ CPU Path        │                  │
│  │ (WebGPU利用可)  │     │ (フォールバック) │                  │
│  └────────┬────────┘     └────────┬────────┘                  │
└───────────│──────────────────────│──────────────────────────────┘
            │                      │
            ▼                      ▼
┌───────────────────┐   ┌───────────────────────────────────────┐
│ GPU Worker        │   │ CPU Worker Pool                       │
│ (single)          │   │ ┌─────────┐┌─────────┐┌─────────┐    │
│                   │   │ │Worker 0 ││Worker 1 ││Worker N │    │
│ WebGPU Compute    │   │ │ (WASM)  ││ (WASM)  ││ (WASM)  │    │
└───────────────────┘   │ └─────────┘└─────────┘└─────────┘    │
                        └───────────────────────────────────────┘
```

### メッセージ型定義

既存のWorkerパターン（`READY/PROGRESS/RESULTS/COMPLETE/ERROR`）を踏襲する。

```typescript
// === リクエスト ===
type MtSeedSearchWorkerRequest =
  | { type: 'START'; job: MtSeedSearchJob }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'STOP' };

// === レスポンス ===
type MtSeedSearchWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: MtSeedSearchProgress }
  | { type: 'RESULTS'; payload: MtSeedSearchResultBatch }
  | { type: 'COMPLETE'; payload: MtSeedSearchCompletion }
  | { type: 'ERROR'; message: string; category: MtSeedSearchErrorCategory };

interface MtSeedSearchProgress {
  jobId: number;
  processedCount: number;
  totalCount: number;
  elapsedMs: number;
  matchesFound: number;
}

interface MtSeedSearchResultBatch {
  results: MtSeedSearchResult[];
}

interface MtSeedSearchCompletion {
  reason: 'finished' | 'stopped' | 'error';
  totalProcessed: number;
  totalMatches: number;
  elapsedMs: number;
}

type MtSeedSearchErrorCategory = 
  | 'VALIDATION'
  | 'WASM_INIT'
  | 'GPU_INIT'
  | 'RUNTIME';
```

### 型ガード

```typescript
function isMtSeedSearchWorkerResponse(data: unknown): data is MtSeedSearchWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return typeof obj.type === 'string' &&
    ['READY', 'PROGRESS', 'RESULTS', 'COMPLETE', 'ERROR'].includes(obj.type);
}
```

---

## WASM実装

### 既存資産の活用

| モジュール | 用途 | 変更要否 |
|-----------|------|---------|
| `Mt19937` (`mt19937.rs`) | MT19937乱数生成（単体） | 変更不要 |
| `hidden_power_from_iv` (`egg_iv.rs`) | めざパ計算 | 変更不要 |
| `StatRange` (`egg_iv.rs`) | IV範囲フィルター | 変更不要 |

### 新規実装: SIMD版 Mt19937 (`mt19937_simd.rs`)

4系統の乱数生成器を並列に持つSIMD最適化版Mt19937を新規実装する。WASM SIMDの128bit幅ベクトル演算（`v128`）を活用し、4つの異なるSeedから同時に乱数列を生成することでスループットを向上させる。

**設計方針**:
- 既存の `mt19937.rs` は単体版として維持（他機能との互換性確保）
- `mt19937_simd.rs` を新規追加し、MT Seed全探索専用に使用
- state配列を `[v128; 624]` で保持し、4系統分のstateをインターリーブ

### 検索実装

```rust
// wasm-pkg/src/mt_seed_search.rs

use crate::mt19937::Mt19937;
use std::collections::HashSet;

/// IVコード型（30bit圧縮表現）
pub type IvCode = u32;

/// MT Seedから指定消費数後のIVセットを導出
pub fn derive_iv_set(mt_seed: u32, advances: u32) -> [u8; 6] {
    let mut mt = Mt19937::new(mt_seed);
    
    // MT消費
    for _ in 0..advances {
        mt.next_u32();
    }
    
    // IV取得（各5bit × 6ステータス）
    let mut ivs = [0u8; 6];
    for iv in ivs.iter_mut() {
        *iv = (mt.next_u32() >> 27) as u8; // 上位5bit
    }
    
    ivs
}

/// IVセットをIVコードにエンコード
pub fn encode_iv_code(ivs: &[u8; 6]) -> IvCode {
    ((ivs[0] as u32) << 25)
        | ((ivs[1] as u32) << 20)
        | ((ivs[2] as u32) << 15)
        | ((ivs[3] as u32) << 10)
        | ((ivs[4] as u32) << 5)
        | (ivs[5] as u32)
}

/// 検索セグメント実行
/// 
/// # Arguments
/// * `start` - 検索開始Seed (inclusive)
/// * `end` - 検索終了Seed (inclusive)
/// * `advances` - MT消費数
/// * `target_codes` - 検索対象IVコードのHashSet
/// 
/// # Returns
/// マッチしたMT SeedとIVコードのペア配列
pub fn search_mt_seed_segment(
    start: u32,
    end: u32,
    advances: u32,
    target_codes: &HashSet<IvCode>,
) -> Vec<(u32, IvCode)> {
    let mut results = Vec::new();
    
    for seed in start..=end {
        let ivs = derive_iv_set(seed, advances);
        let code = encode_iv_code(&ivs);
        
        if target_codes.contains(&code) {
            results.push((seed, code));
        }
    }
    
    results
}
```

### WASM公開関数

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn mt_seed_search_segment(
    start: u32,
    end: u32,
    advances: u32,
    target_codes_ptr: *const u32,
    target_codes_len: usize,
) -> Vec<u32> {
    // target_codesをHashSetに変換
    let target_codes: HashSet<IvCode> = unsafe {
        std::slice::from_raw_parts(target_codes_ptr, target_codes_len)
    }.iter().cloned().collect();
    
    // 検索実行
    let results = search_mt_seed_segment(start, end, advances, &target_codes);
    
    // 結果をフラット配列で返す [seed0, code0, seed1, code1, ...]
    results.into_iter()
        .flat_map(|(seed, code)| vec![seed, code])
        .collect()
}
```

---

## GPU実装

### WGSLシェーダー設計

MT19937のstate配列（624×32bit）はGPUのprivateメモリに収まりにくいため、以下の最適化を検討する。

```wgsl
// mt-seed-search.wgsl

// MT19937定数
const N: u32 = 624u;
const M: u32 = 397u;
const MATRIX_A: u32 = 0x9908B0DFu;
const UPPER_MASK: u32 = 0x80000000u;
const LOWER_MASK: u32 = 0x7FFFFFFFu;

// バインディング
@group(0) @binding(0) var<storage, read> target_codes: array<u32>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;
@group(0) @binding(2) var<storage, read_write> result_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SearchParams;

struct SearchParams {
    start_seed: u32,
    advances: u32,
    target_count: u32,
    max_results: u32,
}

// MT19937 state（privateメモリ）
var<private> mt_state: array<u32, 624>;
var<private> mt_index: u32;

fn mt_init(seed: u32) {
    mt_state[0] = seed;
    for (var i = 1u; i < N; i++) {
        let prev = mt_state[i - 1];
        mt_state[i] = 1812433253u * (prev ^ (prev >> 30u)) + i;
    }
    mt_index = N;
}

fn mt_twist() {
    for (var i = 0u; i < N; i++) {
        let x = (mt_state[i] & UPPER_MASK) | (mt_state[(i + 1u) % N] & LOWER_MASK);
        var x_a = x >> 1u;
        if ((x & 1u) != 0u) {
            x_a ^= MATRIX_A;
        }
        mt_state[i] = mt_state[(i + M) % N] ^ x_a;
    }
    mt_index = 0u;
}

fn mt_next() -> u32 {
    if (mt_index >= N) {
        mt_twist();
    }
    
    var y = mt_state[mt_index];
    mt_index++;
    
    // Tempering
    y ^= y >> 11u;
    y ^= (y << 7u) & 0x9D2C5680u;
    y ^= (y << 15u) & 0xEFC60000u;
    y ^= y >> 18u;
    
    return y;
}

fn encode_iv_code(ivs: array<u32, 6>) -> u32 {
    return (ivs[0] << 25u) | (ivs[1] << 20u) | (ivs[2] << 15u)
         | (ivs[3] << 10u) | (ivs[4] << 5u)  | ivs[5];
}

/// 線形探索によるマッチング判定
/// IVコード数は最大1024件のため、GPUの並列性を活かせば線形探索で十分高速
fn linear_search(code: u32) -> bool {
    for (var i = 0u; i < params.target_count; i++) {
        if (target_codes[i] == code) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seed = params.start_seed + global_id.x;
    
    // オーバーフローチェック
    if (seed < params.start_seed) {
        return;
    }
    
    mt_init(seed);
    
    // MT消費
    for (var i = 0u; i < params.advances; i++) {
        mt_next();
    }
    
    // IV取得
    var ivs: array<u32, 6>;
    for (var i = 0u; i < 6u; i++) {
        ivs[i] = mt_next() >> 27u;
    }
    
    let code = encode_iv_code(ivs);
    
    // 線形探索でマッチング判定
    // 最大1024件のため分岐コストより探索コストの方が小さい
    if (linear_search(code)) {
        let idx = atomicAdd(&result_count, 2u);
        if (idx + 1u < params.max_results * 2u) {
            results[idx] = seed;
            results[idx + 1u] = code;
        }
    }
}
```

### GPU制約と対策

| 制約 | 値（一般的なGPU） | 対策 |
|------|------------------|------|
| privateメモリ | 16KB程度 | MT19937 state (2.5KB) は許容範囲 |
| workgroup size | 256〜1024 | 64を推奨（メモリ効率） |
| dispatch制限 | 65535 | ジョブ分割で対応 |

---

## 型定義ファイル配置

### 新規ファイル

```
src/types/mt-seed-search.ts
```

### 内容

```typescript
// src/types/mt-seed-search.ts

import type { IvSet } from './egg';

/**
 * IVコード（30bit圧縮表現）
 * 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]
 */
export type IvCode = number;

/**
 * 個体値検索フィルター
 */
export interface IvSearchFilter {
  ivRanges: [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
  hiddenPowerType?: number;    // 0-15
  hiddenPowerPower?: number;   // 30-70
}

/**
 * ステータス範囲
 */
export interface StatRange {
  min: number; // 0-31
  max: number; // 0-31
}

/**
 * IVコード生成結果
 */
export type IvCodeGenerationResult =
  | { success: true; ivCodes: IvCode[] }
  | { success: false; error: 'TOO_MANY_COMBINATIONS'; estimatedCount: number };

/**
 * MT Seed検索ジョブ
 */
export interface MtSeedSearchJob {
  searchRange: {
    start: number;  // u32
    end: number;    // u32
  };
  ivCodes: IvCode[];
  mtAdvances: number;
  jobId: number;
}

/**
 * 単一のマッチ結果
 */
export interface MtSeedMatch {
  mtSeed: number;
  ivCode: IvCode;
  ivSet: IvSet;
}

/**
 * ジョブ単位の検索結果
 */
export interface MtSeedSearchResult {
  jobId: number;
  matches: MtSeedMatch[];
}

/**
 * 検索進捗
 */
export interface MtSeedSearchProgress {
  jobId: number;
  processedCount: number;
  totalCount: number;
  elapsedMs: number;
  matchesFound: number;
}

/**
 * 検索完了情報
 */
export interface MtSeedSearchCompletion {
  reason: 'finished' | 'stopped' | 'error';
  totalProcessed: number;
  totalMatches: number;
  elapsedMs: number;
}

/**
 * エラーカテゴリ
 */
export type MtSeedSearchErrorCategory =
  | 'VALIDATION'
  | 'WASM_INIT'
  | 'GPU_INIT'
  | 'RUNTIME';

/**
 * Workerリクエスト
 */
export type MtSeedSearchWorkerRequest =
  | { type: 'START'; job: MtSeedSearchJob }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'STOP' };

/**
 * Workerレスポンス
 */
export type MtSeedSearchWorkerResponse =
  | { type: 'READY'; version: string }
  | { type: 'PROGRESS'; payload: MtSeedSearchProgress }
  | { type: 'RESULTS'; payload: MtSeedSearchResult }  // 1ジョブ完了時に送信
  | { type: 'COMPLETE'; payload: MtSeedSearchCompletion }
  | { type: 'ERROR'; message: string; category: MtSeedSearchErrorCategory };

/**
 * ジョブ計画設定（共通）
 */
export interface JobPlannerConfig {
  fullRange: { start: number; end: number };
  ivCodes: IvCode[];
  mtAdvances: number;
}

/**
 * GPU向けジョブ計画設定
 */
export interface GpuJobPlannerConfig extends JobPlannerConfig {
  deviceLimits: {
    maxComputeWorkgroupsPerDimension: number;
    maxStorageBufferBindingSize: number;
  };
  workgroupSize: number;
}

/**
 * CPU向けジョブ計画設定
 */
export interface CpuJobPlannerConfig extends JobPlannerConfig {
  workerCount: number;
}

/**
 * ジョブ計画
 */
export interface JobPlan {
  jobs: MtSeedSearchJob[];
  totalSearchSpace: number;
  estimatedTimeMs: number;
}

// === ユーティリティ関数 ===

export const MAX_IV_CODES = 1024;

export function encodeIvCode(ivs: IvSet): IvCode {
  return (ivs[0] << 25) | (ivs[1] << 20) | (ivs[2] << 15)
       | (ivs[3] << 10) | (ivs[4] << 5)  | ivs[5];
}

export function decodeIvCode(code: IvCode): IvSet {
  return [
    (code >> 25) & 0x1F,
    (code >> 20) & 0x1F,
    (code >> 15) & 0x1F,
    (code >> 10) & 0x1F,
    (code >> 5)  & 0x1F,
    code & 0x1F,
  ];
}

export function isMtSeedSearchWorkerResponse(
  data: unknown
): data is MtSeedSearchWorkerResponse {
  if (!data || typeof data !== 'object') return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.type === 'string' &&
    ['READY', 'PROGRESS', 'RESULTS', 'COMPLETE', 'ERROR'].includes(obj.type)
  );
}
```

---

## ファイル構成

### 新規作成ファイル

```
src/
  types/
    mt-seed-search.ts           # 型定義
  lib/
    mt-seed-search/
      iv-code-generator.ts      # IVコード変換ロジック
      job-planner-gpu.ts        # GPUジョブ分割
      job-planner-cpu.ts        # CPUジョブ分割
      mt-seed-search-manager.ts # 検索マネージャー
  workers/
    mt-seed-search-worker-gpu.ts   # GPU Worker
    mt-seed-search-worker-cpu.ts   # CPU Worker
  lib/webgpu/mt-seed-search/
    kernel/
      mt-seed-search.wgsl       # WGSLシェーダー
    mt-seed-search-engine.ts    # GPU実行エンジン
wasm-pkg/src/
  mt19937_simd.rs               # SIMD版 MT19937 (4系統並列)
  mt_seed_search.rs             # WASM検索実装
```

---

## テスト計画

### ユニットテスト

| テスト対象 | ファイル | 検証内容 |
|-----------|---------|---------|
| IVコード変換 | `iv-code-generator.test.ts` | encode/decode往復、めざパフィルター |
| ジョブ分割 | `job-planner.test.ts` | GPU/CPU分割ロジック、境界条件 |
| WASM検索 | `mt_seed_search.rs` (Rust test) | 既知Seedの検証 |

### 統合テスト

- `test-integration.html` に MT Seed検索セクションを追加
- 既知のMT Seed/IVペアで検索・検証

### パフォーマンステスト

- GPU vs CPU スループット比較
- IVコード数による性能変化測定

---

## 実装ステップ

1. 型定義ファイル作成 (`src/types/mt-seed-search.ts`)
2. IVコード変換ロジック実装 (`iv-code-generator.ts`)
3. WASM検索関数実装 (`mt_seed_search.rs`)
4. CPU Workerプール実装
5. GPUシェーダー・エンジン実装
6. 検索マネージャー統合
7. UIコンポーネント作成
8. テスト・ドキュメント整備

---

**作成日**: 2025年12月1日  
**バージョン**: 1.0  
**依存**: `05-webgpu-seed-search.md`, `06-egg-iv-handling.md`
