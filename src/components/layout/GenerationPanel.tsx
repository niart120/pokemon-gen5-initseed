import React from 'react';
import { GenerationParamsCard } from '@/components/generation/GenerationParamsCard';
import { GenerationControlCard } from '@/components/generation/GenerationControlCard';
import { GenerationProgressCard } from '@/components/generation/GenerationProgressCard';
import { GenerationResultsControlCard } from '@/components/generation/GenerationResultsControlCard';
import { GenerationResultsTableCard } from '@/components/generation/GenerationResultsTableCard';

// Phase1: 単純な縦積み。後続フェーズで Responsive 3カラム化を検討。
export const GenerationPanel: React.FC = () => {
  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      <GenerationParamsCard />
      <GenerationControlCard />
      <GenerationProgressCard />
      <GenerationResultsControlCard />
      <div className="flex-1 min-h-0">
        <GenerationResultsTableCard />
      </div>
    </div>
  );
};
