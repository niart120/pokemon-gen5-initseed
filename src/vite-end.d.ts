/// <reference types="vite/client" />

// declare const GITHUB_RUNTIME_PERMANENT_NAME: string
// declare const BASE_KV_SERVICE_URL: string

// Wake Lock API types for screen sleep prevention
interface WakeLockSentinel {
  readonly released: boolean;
  readonly type: 'screen';
  release(): Promise<void>;
  addEventListener(type: 'release', listener: () => void): void;
  removeEventListener(type: 'release', listener: () => void): void;
}

interface WakeLock {
  request(type: 'screen'): Promise<WakeLockSentinel>;
}

// Extend Navigator only for typing; avoid unused lint by exporting a type
export interface NavigatorWithWakeLock extends Navigator {
  wakeLock?: WakeLock;
}

// Runtime feature flags exposure in development
declare global {
  interface Window {
    featureFlags?: typeof import('./lib/core/feature-flags').featureFlags;
  }
}

export {};