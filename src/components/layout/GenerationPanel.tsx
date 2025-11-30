import React from 'react';
import { GenerationParamsCard } from '@/components/generation/GenerationParamsCard';
import { GenerationRunCard } from '@/components/generation/GenerationRunCard';
import { GenerationFilterCard } from '@/components/generation/GenerationFilterCard';
import { GenerationResultsTableCard } from '@/components/generation/GenerationResultsTableCard';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { LEFT_COLUMN_WIDTH_CLAMP } from './constants';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';
import { ProfileCard } from '@/components/profile/ProfileCard';

/**
 * GenerationPanel
 * デスクトップ: 3カラム (左: 実行/パラメータ, 中央: フィルター, 右: 結果テーブル)
 * モバイル: 縦積み
 */
export function GenerationPanel() {
  const { isStack, uiScale } = useResponsiveLayout();

  // スケールに応じたレスポンシブサイズ
  const sizes = getResponsiveSizes(uiScale);

  if (isStack) {
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        <div className="flex-none">
          <ProfileCard />
        </div>
        <div className="flex-none">
          <GenerationRunCard />
        </div>
        <div className="flex-none">
          <GenerationParamsCard />
        </div>
        <div className="flex-none">
          <GenerationFilterCard />
        </div>
        <div className="flex-1 min-h-0">
          <GenerationResultsTableCard />
        </div>
      </div>
    );
  }

  // デスクトップ: 3カラム (左: 実行+パラメータ / 中央: フィルター / 右: 結果テーブル)
  return (
    <div className={`flex flex-col ${sizes.gap} max-w-full h-full min-h-0 min-w-fit overflow-hidden`}>
      <div className="flex-none">
        <ProfileCard />
      </div>
      <div className={`flex ${sizes.gap} max-w-full flex-1 min-h-0 min-w-fit overflow-hidden`}>
        {/* Left Column: GenerationRunCard + GenerationParamsCard */}
        <div
          className={`flex-1 flex flex-col ${sizes.gap} min-w-0 overflow-y-auto`}
          style={{
            minHeight: 0,
            width: LEFT_COLUMN_WIDTH_CLAMP,
            flex: `0 0 ${LEFT_COLUMN_WIDTH_CLAMP}`
          }}
        >
          <div className="flex-none">
            <GenerationRunCard />
          </div>
          <div className="flex-1 min-h-0">
            <GenerationParamsCard />
          </div>
        </div>
        {/* Center Column: GenerationFilterCard */}
        <div
          className={`flex flex-col ${sizes.gap} min-w-0 max-w-xs overflow-y-auto`}
          style={{ minHeight: 0 }}
        >
          <div className="flex-1 min-h-0">
            <GenerationFilterCard />
          </div>
        </div>
        {/* Right Column: GenerationResultsTableCard */}
        <div
          className={`flex-[2] flex flex-col ${sizes.gap} min-w-0 overflow-y-auto overflow-x-auto`}
          style={{ minHeight: 0 }}
        >
          <div className="flex-1 min-h-0 overflow-x-auto">
            <GenerationResultsTableCard />
          </div>
        </div>
      </div>
    </div>
  );
}
