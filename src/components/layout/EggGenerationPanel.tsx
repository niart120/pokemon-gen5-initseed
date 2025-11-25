import React, { useEffect } from 'react';
import { EggParamsCard } from '@/components/egg/EggParamsCard';
import { EggFilterCard } from '@/components/egg/EggFilterCard';
import { EggRunCard } from '@/components/egg/EggRunCard';
import { EggResultsCard } from '@/components/egg/EggResultsCard';
import { ProfileCard } from '@/components/profile/ProfileCard';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { LEFT_COLUMN_WIDTH_CLAMP } from './constants';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';
import { useAppStore } from '@/store/app-store';
import { useEggStore } from '@/store/egg-store';

/**
 * EggGenerationPanel
 * タマゴ個体生成一覧表示機能のメインパネル
 * 仕様: spec/agent/pr_design_egg_bw_panel/SPECIFICATION.md
 */
export function EggGenerationPanel() {
  const { isStack, uiScale } = useResponsiveLayout();
  const sizes = getResponsiveSizes(uiScale);

  // Profile同期: アクティブプロファイル変更時にegg-storeへ反映
  const profiles = useAppStore((s) => s.profiles);
  const activeProfileId = useAppStore((s) => s.activeProfileId);
  const applyProfile = useEggStore((s) => s.applyProfile);

  useEffect(() => {
    const profile = profiles.find((p) => p.id === activeProfileId) ?? profiles[0];
    if (profile) {
      applyProfile(profile);
    }
  }, [profiles, activeProfileId, applyProfile]);

  if (isStack) {
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        <div className="flex-none">
          <ProfileCard />
        </div>
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

  // デスクトップ: 3カラム (EggRunCard + EggParamsCard / EggFilterCard / EggResultsCard)
  return (
    <div className={`flex flex-col ${sizes.gap} max-w-full h-full min-h-0 min-w-fit overflow-hidden`}>
      <div className="flex-none">
        <ProfileCard />
      </div>
      <div className={`flex ${sizes.gap} max-w-full flex-1 min-h-0 min-w-fit overflow-hidden`}>
        {/* Left Column: EggRunCard + EggParamsCard */}
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
        </div>
        {/* Center Column: EggFilterCard */}
        <div className={`flex-1 flex flex-col ${sizes.gap} min-w-0 ${sizes.columnWidth} overflow-y-auto`} style={{ minHeight: 0 }}>
          <div className="flex-1 min-h-0">
            <EggFilterCard />
          </div>
        </div>
        {/* Right Column: EggResultsCard */}
        <div className={`flex-1 flex flex-col ${sizes.gap} min-w-0 ${sizes.columnWidth} overflow-y-auto`} style={{ minHeight: 0 }}>
          <div className="flex-1 min-h-0">
            <EggResultsCard />
          </div>
        </div>
      </div>
    </div>
  );
}
