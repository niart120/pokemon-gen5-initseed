/**
 * vitest グローバル設定
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import fetch, { Request, Response } from 'node-fetch';
import { TextDecoder as NodeTextDecoder, TextEncoder as NodeTextEncoder } from 'util';

// Node.js環境でのfetch polyfill（Node 18+ では既に存在するが、未定義時のみ設定）
if (typeof (global as any).fetch === 'undefined') {
  (global as any).fetch = fetch as unknown as typeof global.fetch;
  (global as any).Request = Request as unknown as typeof global.Request;
  (global as any).Response = Response as unknown as typeof global.Response;
}

// TextEncoder/Decoder の polyfill（未定義時のみ設定）
(global as any).TextDecoder = (global as any).TextDecoder || NodeTextDecoder;
(global as any).TextEncoder = (global as any).TextEncoder || NodeTextEncoder;

// ------------------------------------------------------------
// localStorage モック（Zustand persist 警告抑制用）
// ------------------------------------------------------------
interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  clear(): void;
  key(index: number): string | null;
  readonly length: number;
}

function createInMemoryStorage(): StorageLike {
  const store = new Map<string, string>();
  return {
    getItem(key: string) {
      return store.has(key) ? store.get(key)! : null;
    },
    setItem(key: string, value: string) {
      store.set(key, String(value));
    },
    removeItem(key: string) {
      store.delete(key);
    },
    clear() {
      store.clear();
    },
    key(index: number) {
      const keys = Array.from(store.keys());
      return index >= 0 && index < keys.length ? keys[index] : null;
    },
    get length() {
      return store.size;
    },
  };
}

// Node 環境では window が無いので最小限のモックを提供
if (typeof (global as any).window === 'undefined') {
  (global as any).window = {} as any;
}

if (!(global as any).window.localStorage) {
  const ls = createInMemoryStorage();
  (global as any).window.localStorage = ls;
  (global as any).localStorage = ls; // 直接参照にも対応
}

// WebAssembly環境の有無を通知（必要なら）
if (typeof (global as any).WebAssembly === 'undefined') {
  // eslint-disable-next-line no-console
  console.warn('WebAssembly not available in test environment');
}

// ファイルシステムからWASMを読み込むヘルパー（必要なテストのみ使用）
;(global as any).loadWasmFromFile = async (wasmPath: string) => {
  try {
    const wasmFile = readFileSync(resolve(wasmPath));
    const wasmModule = await WebAssembly.instantiate(wasmFile as unknown as BufferSource);
    return (wasmModule as WebAssembly.WebAssemblyInstantiatedSource).instance;
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error('WASM loading error:', error);
    throw error;
  }
};
