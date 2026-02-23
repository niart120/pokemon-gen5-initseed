import * as React from 'react';
import { useVirtualizer, type VirtualItem } from '@tanstack/react-virtual';

interface UseTableVirtualizationOptions {
  rowCount: number;
  defaultRowHeight: number;
  overscan?: number;
}

interface UseTableVirtualizationResult {
  containerRef: React.RefObject<HTMLDivElement | null>;
  virtualRows: VirtualItem[];
  paddingTop: number;
  paddingBottom: number;
  measureRow: (element: Element | null) => void;
  isVirtualized: boolean;
}

export function useTableVirtualization({
  rowCount,
  defaultRowHeight,
  overscan = 6,
}: UseTableVirtualizationOptions): UseTableVirtualizationResult {
  const containerRef = React.useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => containerRef.current,
    estimateSize: React.useCallback(() => Math.max(defaultRowHeight, 1), [defaultRowHeight]),
    overscan,
  });

  const virtualRows = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();
  const paddingTop = virtualRows.length > 0 ? virtualRows[0].start : 0;
  const paddingBottom = virtualRows.length > 0 ? Math.max(0, totalSize - virtualRows[virtualRows.length - 1].end) : 0;

  const measureRow = React.useCallback(
    (element: Element | null) => {
      if (element) {
        virtualizer.measureElement(element);
      }
    },
    [virtualizer],
  );

  return {
    containerRef,
    virtualRows,
    paddingTop,
    paddingBottom,
    measureRow,
    isVirtualized: rowCount > 0 && virtualRows.length < rowCount,
  };
}
