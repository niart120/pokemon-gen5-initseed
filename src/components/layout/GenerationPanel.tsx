import React from 'react';
import { GenerationParamsCard } from '@/components/generation/GenerationParamsCard';
// 統合カード (Control + Progress)
import { GenerationRunCard } from '@/components/generation/GenerationRunCard';
import { GenerationResultsControlCard } from '@/components/generation/GenerationResultsControlCard';
import { GenerationResultsTableCard } from '@/components/generation/GenerationResultsTableCard';

/**
 * GenerationPanel (Phase1 / Layout Refactor)
 * 案A: デスクトップ/タブレットは 2カラム (左=設定/制御/進捗, 右=結果)。
 * モバイル幅では従来どおり縦積みにフォールバック (後続PhaseでAccordion化予定)。
 * - 右カラム上部に sticky header を置く前提だが、Phase1では土台のみ。
 */
export const GenerationPanel: React.FC = () => {
  return (
    <div
      className={[
        // 2カラムグリッド: 1200+:340px/1fr, ~md は1カラム
        'h-full min-h-0 w-full',
        'grid gap-3',
      // 1024px~: 左カラム 480px に拡張 (視認性向上)。1280px~で 600px に微調整可。
      'lg:grid-cols-[480px_minmax(0,1fr)] xl:grid-cols-[600px_minmax(0,1fr)]',
        'auto-rows-min',
      ].join(' ')}
    >
      {/* Left Column */}
      <div className="flex flex-col gap-3 min-h-0">
        <GenerationRunCard />
        <GenerationParamsCard />
      </div>
      {/* Right Column */}
        <div className="flex flex-col gap-3 min-h-0 overflow-hidden">
        {/* sticky header 土台 (Phase2でsticky化/mini progress統合) */}
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
        <div className="flex-1 min-h-0 overflow-auto" data-testid="gen-results-scroll">
          <GenerationResultsTableCard />
        </div>
      </div>
    </div>
  );
};
