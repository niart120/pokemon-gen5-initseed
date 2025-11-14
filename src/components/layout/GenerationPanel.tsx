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
 * - 右カラム上部に sticky header を置く前提だが、Phase1では土台のみ。
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
    <div className="flex flex-col gap-3 h-full min-h-0 w-full overflow-hidden">
      <div className="flex-none">
        <ProfileCard />
      </div>
      <div className="flex gap-3 flex-1 min-h-0 w-full overflow-hidden">
        {/* Left Column */}
        <div
          className="flex flex-col gap-3 min-h-0"
          style={{
            width: LEFT_COLUMN_WIDTH_CLAMP,
            flex: `0 0 ${LEFT_COLUMN_WIDTH_CLAMP}`
          }}
        >
          <GenerationRunCard />
          <GenerationParamsCard />
        </div>
        {/* Right Column */}
        <div className="flex flex-col gap-3 min-h-0 overflow-hidden flex-1">
          <div
            className={[
              'flex flex-col gap-2',
              'sticky top-0 z-10 backdrop-blur bg-background/90 border-b border-border/50 p-1 rounded-md',
            ].join(' ')}
            role="region"
            aria-label="Generation results controls"
            data-testid="gen-results-sticky"
          >
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
