/**
 * vitest グローバル設定
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import fetch, { Request, Response } from 'node-fetch';
import { TextDecoder as NodeTextDecoder, TextEncoder as NodeTextEncoder } from 'util';

// Node.js環境でのfetch polyfill（Node 18+ では既に存在するが、未定義時のみ設定）
declare global {
  // Node の test 環境向けに必要な最小のグローバル拡張
  interface Window { localStorage?: StorageLike }
}

const g = globalThis as typeof globalThis & {
  fetch?: typeof global.fetch;
  Request?: typeof global.Request;
  Response?: typeof global.Response;
  TextDecoder?: typeof NodeTextDecoder;
  TextEncoder?: typeof NodeTextEncoder;
  window?: { localStorage?: StorageLike };
  WebAssembly?: typeof WebAssembly;
  localStorage?: StorageLike;
};

if (typeof g.fetch === 'undefined') {
  g.fetch = fetch as unknown as typeof global.fetch;
  g.Request = Request as unknown as typeof global.Request;
  g.Response = Response as unknown as typeof global.Response;
}

// TextEncoder/Decoder の polyfill（未定義時のみ設定）
g.TextDecoder = g.TextDecoder || (NodeTextDecoder as unknown as typeof g.TextDecoder);
g.TextEncoder = g.TextEncoder || (NodeTextEncoder as unknown as typeof g.TextEncoder);

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
if (typeof g.window === 'undefined') {
  // テスト環境の Window 互換オブジェクト（最小）
  (g as unknown as { window: { localStorage?: StorageLike } }).window = {};
}

if (!g.window.localStorage) {
  const ls = createInMemoryStorage();
  g.window.localStorage = ls;
  g.localStorage = ls; // 直接参照にも対応
}

// WebAssembly環境の有無を通知（必要なら）
if (typeof g.WebAssembly === 'undefined') {
  console.warn('WebAssembly not available in test environment');
}

// ファイルシステムからWASMを読み込むヘルパー（必要なテストのみ使用）
g.loadWasmFromFile = async (wasmPath: string) => {
  try {
    const wasmFile = readFileSync(resolve(wasmPath));
    const wasmModule = await WebAssembly.instantiate(wasmFile as unknown as BufferSource);
  return (wasmModule as WebAssembly.WebAssemblyInstantiatedSource).instance;
  } catch (error) {
    console.error('WASM loading error:', error);
    throw error;
  }
};
