/**
 * MiscPanel.tsx
 * その他機能パネル
 * 3列均等幅/モバイル時スタック配置
 */

import React from 'react';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { MtSeedSearchCard } from './MtSeedSearchCard';

export const MiscPanel: React.FC = () => {
  const { isStack } = useResponsiveLayout();

  // 3列均等幅 / モバイル時はスタック
  const gridClasses = isStack
    ? 'flex flex-col gap-3'
    : 'grid grid-cols-3 gap-3 h-full';

  return (
    <div className={gridClasses}>
      {/* Column 1: MT Seed 検索 */}
      <div className={isStack ? '' : 'flex flex-col'}>
        <MtSeedSearchCard />
      </div>

      {/* Column 2: 将来の機能用 */}
      <div className={isStack ? '' : 'flex flex-col'}>
        {/* Placeholder for future cards */}
      </div>

      {/* Column 3: 将来の機能用 */}
      <div className={isStack ? '' : 'flex flex-col'}>
        {/* Placeholder for future cards */}
      </div>
    </div>
  );
};
