import React from 'react';
import { GenerationParamsCard } from '@/components/generation/GenerationParamsCard';
// 統合カード (Control + Progress)
import { GenerationRunCard } from '@/components/generation/GenerationRunCard';
import { GenerationResultsControlCard } from '@/components/generation/GenerationResultsControlCard';
import { GenerationResultsTableCard } from '@/components/generation/GenerationResultsTableCard';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { LEFT_COLUMN_WIDTH_CLAMP } from './constants';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';
import { ProfileCard } from '@/components/profile/ProfileCard';

/**
 * GenerationPanel (Phase1 / Layout Refactor)
 * 案A: デスクトップ/タブレットは 2カラム (左=設定/制御/進捗, 右=結果)。
 * モバイル幅では従来どおり縦積みにフォールバック (後続PhaseでAccordion化予定)。
 * - SearchPanel と同基準のレイアウトスケールを用い、PCは2カラム構成を維持。
 */
export const GenerationPanel: React.FC = () => {
  const { isStack, uiScale } = useResponsiveLayout();

  // スケールに応じたレスポンシブサイズ
  const sizes = getResponsiveSizes(uiScale);

  // モバイル/スタックレイアウト
  if (isStack) {
    // SearchPanelモバイル挙動へ合わせる: ページ全体スクロールに委ね、
    // 個別overflow-autoコンテナを廃止し縦スタック構造に統一。
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        {/* 自然高さのカードは flex-none */}
        <div className="flex-none">
          <ProfileCard />
        </div>
        <div className="flex-none">
          <GenerationParamsCard />
        </div>
        <div className="flex-none">
          <GenerationRunCard />
        </div>
        <div className="flex-none">
          <GenerationResultsControlCard />
        </div>
        {/* 結果テーブルは領域に収める（mainにスクロールを閉じ込める） */}
        <div className="flex-1 min-h-[200px]">
          <GenerationResultsTableCard />
        </div>
      </div>
    );
  }

  // デスクトップ: 2カラム (左: 制御+パラメータ 固定幅clamp / 右: 結果エリア)
  return (
    <div className={`flex flex-col ${sizes.gap} h-full min-h-0 min-w-fit overflow-hidden`}>
      <div className="flex-none">
        <ProfileCard />
      </div>
      <div className={`flex ${sizes.gap} flex-1 min-h-0 min-w-fit overflow-hidden`}>
        {/* Left Column */}
        <div
          className={`flex flex-col ${sizes.gap} min-h-0 overflow-y-auto`}
          style={{
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
        {/* Right Column */}
        <div className={`flex flex-col flex-1 min-w-0 min-h-0 overflow-y-auto ${sizes.gap}`}>
          <div className="flex-none">
            <GenerationResultsControlCard />
          </div>
          <div className="flex-1 min-h-0">
            <GenerationResultsTableCard />
          </div>
        </div>
      </div>
    </div>
  );
};
