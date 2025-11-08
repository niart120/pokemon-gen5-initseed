import { useAppStore } from '@/store/app-store';

/**
 * WebGPUサポート判定
 */
export function isWebGpuSupported(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }
  return typeof (navigator as Navigator & { gpu?: unknown }).gpu !== 'undefined';
}

/**
 * WebGPU検索モードを利用すべきかを判定
 */
export function shouldUseWebGpuSearch(): boolean {
  if (!isWebGpuSupported()) {
    return false;
  }

  try {
    const state = useAppStore.getState();
    return state.searchExecutionMode === 'gpu';
  } catch (error) {
    console.warn('[search-mode] Failed to access app store for WebGPU flag:', error);
    return false;
  }
}
