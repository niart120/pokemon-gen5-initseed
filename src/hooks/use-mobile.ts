import { useSyncExternalStore, useMemo, useCallback, useRef } from 'react';
import rawBreakpoints from '@/config/breakpoints.json';

const BREAKPOINTS = rawBreakpoints as Record<'sm' | 'md' | 'lg' | 'xl' | '2xl', number>;
const BASE_BREAKPOINT = 'base' as const;
const BREAKPOINT_SEQUENCE = [
  BASE_BREAKPOINT,
  'sm',
  'md',
  'lg',
  'xl',
  '2xl'
] as const;

type ResponsiveBreakpoint = (typeof BREAKPOINT_SEQUENCE)[number];

const MOBILE_BREAKPOINT = BREAKPOINTS.md;

const STACK_STATE_BY_BREAKPOINT: Record<ResponsiveBreakpoint, boolean> = {
  base: true,
  sm: true,
  md: false,
  lg: false,
  xl: false,
  '2xl': false
};

const UI_SCALE_BY_BREAKPOINT: Record<ResponsiveBreakpoint, number> = {
  base: 0.85,
  sm: 0.9,
  md: 1.0,
  lg: 1.0,
  xl: 1.1,
  '2xl': 1.33
};

const getMinWidth = (breakpoint: ResponsiveBreakpoint): number => {
  if (breakpoint === BASE_BREAKPOINT) {
    return 0;
  }
  return BREAKPOINTS[breakpoint];
};

const resolveBreakpoint = (width: number): ResponsiveBreakpoint => {
  for (let index = BREAKPOINT_SEQUENCE.length - 1; index >= 0; index -= 1) {
    const breakpoint = BREAKPOINT_SEQUENCE[index];
    if (width >= getMinWidth(breakpoint)) {
      return breakpoint;
    }
  }
  return BASE_BREAKPOINT;
};

const getBreakpointMatches = (width: number): Record<ResponsiveBreakpoint, boolean> => {
  return BREAKPOINT_SEQUENCE.reduce((acc, breakpoint) => {
    acc[breakpoint] = width >= getMinWidth(breakpoint);
    return acc;
  }, {} as Record<ResponsiveBreakpoint, boolean>);
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
    const matches = getBreakpointMatches(width);

    return {
      breakpoint,
      isStack: STACK_STATE_BY_BREAKPOINT[breakpoint],
      uiScale: UI_SCALE_BY_BREAKPOINT[breakpoint],
      matches,
      dimensions: {
        width,
        height
      }
    };
  }, [sizeString]);
}

// 後方互換性のためのエイリアス
export const useIsStackLayout = useResponsiveLayout;
