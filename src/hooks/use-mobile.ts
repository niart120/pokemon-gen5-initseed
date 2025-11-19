import { useSyncExternalStore, useMemo, useCallback, useRef } from 'react';
import rawBreakpoints from '@/config/breakpoints.json';

const BREAKPOINTS = rawBreakpoints as Record<'sm' | 'md' | 'lg' | 'xl' | '2xl', number>;
const BASE_BREAKPOINT = 'base' as const;
type TailwindBreakpoint = keyof typeof BREAKPOINTS;
type ResponsiveBreakpoint = TailwindBreakpoint | typeof BASE_BREAKPOINT;

const MOBILE_BREAKPOINT = BREAKPOINTS.md;
const STACK_PORTRAIT_BREAKPOINT = BREAKPOINTS.lg;
const DESCENDING_BREAKPOINTS = (Object.entries(BREAKPOINTS) as Array<[
  TailwindBreakpoint,
  number
]>).sort((a, b) => b[1] - a[1]);

const resolveBreakpoint = (width: number): ResponsiveBreakpoint => {
  for (const [breakpoint, minWidth] of DESCENDING_BREAKPOINTS) {
    if (width >= minWidth) {
      return breakpoint;
    }
  }
  return BASE_BREAKPOINT;
};

const calculateUiScale = (width: number): number => {
  if (width <= 1366) {
    return 0.85;
  }
  if (width <= 1920) {
    return 1.0;
  }
  if (width <= 2048) {
    return 1.1;
  }
  if (width <= 2560) {
    return 1.33;
  }
  if (width <= 3840) {
    return 1.5;
  }
  return Math.min(2.0, width / 1920);
};

/**
 * モバイル検出フック
 * MediaQuery API を使用してモバイルデバイスを検出
 */
export function useIsMobile() {
  return useSyncExternalStore(
    // subscribe: MediaQueryの変更を監視
    (callback) => {
      const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`);
      mql.addEventListener('change', callback);
      return () => mql.removeEventListener('change', callback);
    },
    // getSnapshot: 現在のモバイル状態を取得
    () => window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`).matches,
    // getServerSnapshot: SSR用の初期値
    () => false
  );
}

/**
 * レスポンシブレイアウト検出フック
 * ウィンドウサイズに応じてスタックレイアウトとUIスケールを決定
 */
export function useResponsiveLayout() {
  // RAF（RequestAnimationFrame）ベースのスロットリング
  const rafIdRef = useRef<number | undefined>(undefined);
  
  // subscribe関数を安定化
  const subscribe = useCallback((callback: () => void) => {
    const handleResize = () => {
      // RAF で次のフレームまで遅延させてパフォーマンス最適化
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
      
      rafIdRef.current = requestAnimationFrame(() => {
        callback();
      });
    };
    
    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, []);
  
  // getSnapshot関数を安定化 - パフォーマンスを考慮した実装
  const getSnapshot = useCallback(() => {
    // DOM読み取りは必要最小限に
    const currentWidth = typeof window !== 'undefined' ? window.innerWidth : 1920;
    const currentHeight = typeof window !== 'undefined' ? window.innerHeight : 1080;
    return `${currentWidth}x${currentHeight}`;
  }, []);
  
  // SSR用スナップショット
  const getServerSnapshot = useCallback(() => "1920x1080", []);

  // ウィンドウサイズの監視
  const sizeString = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // レスポンシブ状態の計算（メモ化で重複計算防止）
  return useMemo(() => {
    const [widthStr, heightStr] = sizeString.split('x');
    const width = parseInt(widthStr, 10);
    const height = parseInt(heightStr, 10);
    const breakpoint = resolveBreakpoint(width);
    const isPortrait = height > width;
    
    // スタックレイアウト判定
    const isStack = width < MOBILE_BREAKPOINT || (isPortrait && width < STACK_PORTRAIT_BREAKPOINT);
    const uiScale = calculateUiScale(width);
    
    return {
      isStack,
      uiScale,
      breakpoint,
      dimensions: {
        width,
        height
      }
    };
  }, [sizeString]);
}

// 後方互換性のためのエイリアス
export const useIsStackLayout = useResponsiveLayout;
