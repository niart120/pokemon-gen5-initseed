import React from 'react';
import { EggParamsCard } from '@/components/egg/EggParamsCard';
import { EggFilterCard } from '@/components/egg/EggFilterCard';
import { EggRunCard } from '@/components/egg/EggRunCard';
import { EggResultsCard } from '@/components/egg/EggResultsCard';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { LEFT_COLUMN_WIDTH_CLAMP } from './constants';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';

/**
 * EggGenerationPanel
 * タマゴ個体生成一覧表示機能のメインパネル
 * 仕様: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md
 */
export function EggGenerationPanel() {
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
