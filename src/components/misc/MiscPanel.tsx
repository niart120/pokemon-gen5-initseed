/**
 * MiscPanel.tsx
 * その他機能パネル
 * ProfileCardをトップに配置、その下に2列グリッド/モバイル時スタック配置
 */

import React from 'react';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { ProfileCard } from '@/components/profile/ProfileCard';
import { MtSeedSearchCard } from './MtSeedSearchCard';
import { IdAdjustmentCard } from './IdAdjustmentCard';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';

export const MiscPanel: React.FC = () => {
  const { isStack, uiScale } = useResponsiveLayout();
  const sizes = getResponsiveSizes(uiScale);

  if (isStack) {
    // モバイル: 縦積み
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        <div className="flex-none">
          <ProfileCard />
        </div>
        <div className="flex-none">
          <MtSeedSearchCard />
        </div>
        <div className="flex-1 min-h-0">
          <IdAdjustmentCard />
        </div>
      </div>
    );
  }

  // デスクトップ: ProfileCard上部、その下に2列グリッド
  return (
    <div className={`flex flex-col ${sizes.gap} max-w-full h-full min-h-0 min-w-0 overflow-hidden`}>
      {/* トップ: ProfileCard */}
      <div className="flex-none">
        <ProfileCard />
      </div>

      {/* メインコンテンツ: 3列グリッド */}
      <div className={`grid grid-cols-3 ${sizes.gap} flex-1 min-h-0 overflow-hidden`}>
        {/* Column 1: MT Seed 検索 */}
        <div className="flex flex-col min-h-0 overflow-y-auto">
          <MtSeedSearchCard />
        </div>

        {/* Column 2: ID調整 */}
        <div className="flex flex-col min-h-0 overflow-y-auto">
          <IdAdjustmentCard />
        </div>

        {/* Column 3: その他機能 */}
        <div className="flex flex-col min-h-0 overflow-y-auto">
          {/* Add other miscellaneous components here */}
        </div>
      </div>
    </div>
  );
};
