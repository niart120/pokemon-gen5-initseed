/**
 * EggSearchPanel
 * Search(Egg)タブのメインパネル
 * 仕様: spec/agent/pr_egg_boot_timing_search/UI_DESIGN.md
 */

import React, { useEffect } from 'react';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';
import { LEFT_COLUMN_WIDTH_CLAMP } from '@/components/layout/constants';
import { ProfileCard } from '@/components/profile/ProfileCard';
import { useAppStore } from '@/store/app-store';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { EggSearchRunCard } from './EggSearchRunCard';
import { EggSearchParamsCard } from './EggSearchParamsCard';
import { EggSearchFilterCard } from './EggSearchFilterCard';
import { EggSearchResultsCard } from './EggSearchResultsCard';

export function EggSearchPanel() {
  const { isStack, uiScale } = useResponsiveLayout();
  const sizes = getResponsiveSizes(uiScale);

  // Profile同期: アクティブプロファイル変更時にstore反映
  const profiles = useAppStore((s) => s.profiles);
  const activeProfileId = useAppStore((s) => s.activeProfileId);
  const applyProfile = useEggBootTimingSearchStore((s) => s.applyProfile);

  useEffect(() => {
    const profile = profiles.find((p) => p.id === activeProfileId) ?? profiles[0];
    if (profile) {
      applyProfile(profile);
    }
  }, [profiles, activeProfileId, applyProfile]);

  if (isStack) {
    // モバイル: スタックレイアウト
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        <div className="flex-none">
          <ProfileCard />
        </div>
        <div className="flex-none">
          <EggSearchRunCard />
        </div>
        <div className="flex-none">
          <EggSearchParamsCard />
        </div>
        <div className="flex-none">
          <EggSearchFilterCard />
        </div>
        <div className="flex-1 min-h-0">
          <EggSearchResultsCard />
        </div>
      </div>
    );
  }

  // デスクトップ: 3カラムレイアウト
  return (
    <div className={`flex flex-col ${sizes.gap} max-w-full h-full min-h-0 min-w-fit overflow-hidden`}>
      {/* トップ: ProfileCard */}
      <div className="flex-none">
        <ProfileCard />
      </div>
      
      {/* 3カラム */}
      <div className={`flex ${sizes.gap} max-w-full flex-1 min-h-0 min-w-fit overflow-hidden`}>
        {/* 左カラム: RunCard + SearchParamsCard */}
        <div
          className={`flex-1 flex flex-col ${sizes.gap} min-w-0 overflow-y-auto`}
          style={{
            minHeight: 0,
            width: LEFT_COLUMN_WIDTH_CLAMP,
            flex: `0 0 ${LEFT_COLUMN_WIDTH_CLAMP}`,
          }}
        >
          <div className="flex-none">
            <EggSearchRunCard />
          </div>
          <div className="flex-1 min-h-0">
            <EggSearchParamsCard />
          </div>
        </div>
        
        {/* 中央カラム: FilterCard */}
        <div
          className={`flex flex-col ${sizes.gap} min-w-0 max-w-xs overflow-y-auto`}
          style={{ minHeight: 0 }}
        >
          <div className="flex-1 min-h-0">
            <EggSearchFilterCard />
          </div>
        </div>
        
        {/* 右カラム: ResultsCard */}
        <div
          className={`flex-[2] flex flex-col ${sizes.gap} min-w-0 overflow-y-auto overflow-x-auto`}
          style={{ minHeight: 0 }}
        >
          <div className="flex-1 min-h-0 overflow-x-auto">
            <EggSearchResultsCard />
          </div>
        </div>
      </div>
    </div>
  );
}
