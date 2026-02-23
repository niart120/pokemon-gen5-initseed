/**
 * MT Seed Search テストヘルパー
 */
import type {
  MtSeedSearchWorkerResponse,
  MtSeedSearchJob,
  IvCode,
} from '@/types/mt-seed-search';
import { encodeIvCode } from '@/types/mt-seed-search';
import type { IvSet } from '@/types/egg';

// === 固定IVコード定数 ===

/**
 * 6V (31-31-31-31-31-31)
 */
export const IV_SET_6V: IvSet = [31, 31, 31, 31, 31, 31];
export const IV_CODE_6V: IvCode = encodeIvCode(IV_SET_6V);

/**
 * 5V 0A (31-0-31-31-31-31)
 * 特殊アタッカー向け
 */
export const IV_SET_5V_0A: IvSet = [31, 0, 31, 31, 31, 31];
export const IV_CODE_5V_0A: IvCode = encodeIvCode(IV_SET_5V_0A);

/**
 * 4V (31-31-x-31-x-31) - Def/SpD個体値不問
 * 複数のIVコードを生成
 */
export function generate4VIvCodes(): IvCode[] {
  const codes: IvCode[] = [];
  // HP, Atk, SpA, Spe = 31 固定、Def/SpD = 0-31
  for (let def = 0; def <= 31; def++) {
    for (let spd = 0; spd <= 31; spd++) {
      const ivSet: IvSet = [31, 31, def, 31, spd, 31];
      codes.push(encodeIvCode(ivSet));
    }
  }
  return codes;
}

// 事前計算した4Vコードセットをキャッシュ
let cached4VCodes: IvCode[] | null = null;
export function getIvCodes4V(): IvCode[] {
  if (!cached4VCodes) {
    cached4VCodes = generate4VIvCodes();
  }
  return cached4VCodes;
}

// === Worker ヘルパー ===

/**
 * CPU Workerを作成
 */
export function createCpuWorker(): Worker {
  return new Worker(
    new URL('@/workers/mt-seed-search-worker-cpu.ts', import.meta.url),
    { type: 'module' }
  );
}

/**
 * GPU Workerを作成
 */
export function createGpuWorker(): Worker {
  return new Worker(
    new URL('@/workers/mt-seed-search-worker-gpu.ts', import.meta.url),
    { type: 'module' }
  );
}

/**
 * Workerからのメッセージを待機
 *
 * @param worker - 対象Worker
 * @param predicate - メッセージ判定関数
 * @param timeoutMs - タイムアウト（ミリ秒）
 * @returns 条件を満たしたメッセージ
 */
export function waitForMessage(
  worker: Worker,
  predicate: (msg: MtSeedSearchWorkerResponse) => boolean,
  timeoutMs = 30000
): Promise<MtSeedSearchWorkerResponse> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout waiting for message (${timeoutMs}ms)`));
    }, timeoutMs);

    const handler = (ev: MessageEvent<MtSeedSearchWorkerResponse>) => {
      if (predicate(ev.data)) {
        clearTimeout(timer);
        worker.removeEventListener('message', handler);
        resolve(ev.data);
      }
    };

    worker.addEventListener('message', handler);
  });
}

/**
 * 複数のメッセージタイプを収集
 *
 * @param worker - 対象Worker
 * @param types - 収集するメッセージタイプ
 * @param endPredicate - 終了条件
 * @param timeoutMs - タイムアウト（ミリ秒）
 * @returns 収集したメッセージ配列
 */
export function collectMessages(
  worker: Worker,
  types: MtSeedSearchWorkerResponse['type'][],
  endPredicate: (msg: MtSeedSearchWorkerResponse) => boolean,
  timeoutMs = 60000
): Promise<MtSeedSearchWorkerResponse[]> {
  return new Promise((resolve, reject) => {
    const collected: MtSeedSearchWorkerResponse[] = [];
    const timer = setTimeout(() => {
      reject(new Error(`Timeout collecting messages (${timeoutMs}ms)`));
    }, timeoutMs);

    const handler = (ev: MessageEvent<MtSeedSearchWorkerResponse>) => {
      const msg = ev.data;
      if (types.includes(msg.type)) {
        collected.push(msg);
      }
      if (endPredicate(msg)) {
        clearTimeout(timer);
        worker.removeEventListener('message', handler);
        resolve(collected);
      }
    };

    worker.addEventListener('message', handler);
  });
}

// === ジョブ作成ヘルパー ===

/**
 * テスト用ジョブを作成
 */
export function createTestJob(params: {
  start?: number;
  end?: number;
  ivCodes?: IvCode[];
  mtAdvances?: number;
  jobId?: number;
}): MtSeedSearchJob {
  return {
    searchRange: {
      start: params.start ?? 0,
      end: params.end ?? 999_999,
    },
    ivCodes: params.ivCodes ?? [IV_CODE_6V],
    mtAdvances: params.mtAdvances ?? 0,
    jobId: params.jobId ?? 0,
  };
}

// === パフォーマンス計測ヘルパー ===

export interface PerformanceResult {
  mode: 'cpu' | 'gpu';
  searchRange: number;
  elapsedMs: number;
  seedsPerSecond: number;
  matchesFound: number;
}

/**
 * スループットを計算
 */
export function calculateThroughput(
  searchRange: number,
  elapsedMs: number
): number {
  return searchRange / (elapsedMs / 1000);
}

/**
 * 人間可読な数値フォーマット
 */
export function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

// === WebGPU サポート確認 ===

/**
 * WebGPU が利用可能かどうか
 */
export function isWebGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined';
}
