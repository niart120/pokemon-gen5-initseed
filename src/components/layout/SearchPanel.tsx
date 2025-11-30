import React, { useState, useMemo } from 'react';
import { useAppStore } from '../../store/app-store';
import {
  SearchParamsCard,
  TargetSeedsCard,
} from '../search/configuration';
import { SearchControlCard, SearchProgressCard } from '../search/control';
import { ResultsCard, ResultDetailsDialog } from '../search/results';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { getResponsiveSizes } from '@/lib/utils/responsive-sizes';
import { LEFT_COLUMN_WIDTH_CLAMP } from './constants';
import type { InitialSeedResult, SearchResult } from '../../types/search';
import { ProfileCard } from '@/components/profile/ProfileCard';

export function SearchPanel() {
  const { searchResults } = useAppStore();
  
  const { isStack, uiScale } = useResponsiveLayout();
  
  // スケールに応じたレスポンシブサイズ
  const sizes = getResponsiveSizes(uiScale);

  // Results state management
  const [selectedResult, setSelectedResult] = useState<InitialSeedResult | null>(null);
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);

  // Convert InitialSeedResult to SearchResult for export
  const convertToSearchResults: SearchResult[] = useMemo(() => {
    return searchResults.map(result => ({
      seed: result.seed,
      datetime: result.datetime,
      timer0: result.timer0,
      vcount: result.vcount,
      romVersion: result.conditions.romVersion,
      romRegion: result.conditions.romRegion,
      hardware: result.conditions.hardware,
      macAddress: result.conditions.macAddress,
      keyCode: result.keyCode,
      message: result.message,
      hash: result.sha1Hash,
    }));
  }, [searchResults]);

  // Sort results by datetime (ascending - oldest first)
  const sortedResults = useMemo(() => {
    return [...searchResults].sort((a, b) => a.datetime.getTime() - b.datetime.getTime());
  }, [searchResults]);

  const handleShowDetails = (result: InitialSeedResult) => {
    setSelectedResult(result);
    setIsDetailsOpen(true);
  };

  if (isStack) {
    return (
      <div className={`${sizes.gap} flex flex-col h-full overflow-y-auto overflow-x-hidden`}>
        <div className="flex-none">
          <ProfileCard />
        </div>
        <div className="flex-none">
          <SearchParamsCard />
        </div>
        <div className="flex-none">
          <TargetSeedsCard />
        </div>
        <div className="flex-none">
          <SearchControlCard />
        </div>
        <div className="flex-none">
          <SearchProgressCard />
        </div>
        <div className="flex-1 min-h-0">
          <ResultsCard
            sortedResults={sortedResults}
            convertedResults={convertToSearchResults}
            onShowDetails={handleShowDetails}
          />
        </div>
        <ResultDetailsDialog
          result={selectedResult}
          isOpen={isDetailsOpen}
          onOpenChange={setIsDetailsOpen}
        />
      </div>
    );
  }

  // PC: 3カラム配置（設定 | 検索制御・進捗 | 結果）
  return (
    <>
      <div className={`flex flex-col ${sizes.gap} max-w-full h-full min-h-0 min-w-0 overflow-hidden`}>
        <div className="flex-none">
          <ProfileCard />
        </div>
        <div className={`flex ${sizes.gap} max-w-full flex-1 min-h-0 min-w-0 overflow-hidden`}>
          {/* 左カラム: 検索制御・進捗エリア */}
          <div className={`flex-1 flex flex-col ${sizes.gap} min-w-0 ${sizes.columnWidth} overflow-y-auto`} style={{ minHeight: 0 }}>
            <div className="flex-none">
              <SearchControlCard />
            </div>
            <div className="flex-1 min-h-0">
              <SearchProgressCard />
            </div>
          </div>
          {/* 中央カラム: 設定エリア */}
          <div
            className={`flex-1 flex flex-col ${sizes.gap} min-w-0 overflow-y-auto`}
            style={{ minHeight: 0, width: LEFT_COLUMN_WIDTH_CLAMP, flex: `0 0 ${LEFT_COLUMN_WIDTH_CLAMP}` }}
          >
            <div className="flex-none">
              <SearchParamsCard />
            </div>
            <div className="flex-1 min-h-0">
              <TargetSeedsCard />
            </div>
          </div>

          {/* 右カラム: 結果エリア */}
          <div className={`flex-1 flex flex-col ${sizes.gap} min-w-0 ${sizes.columnWidth} overflow-y-auto`} style={{ minHeight: 0 }}>
            <div className="flex-1 min-h-0">
              <ResultsCard
                sortedResults={sortedResults}
                convertedResults={convertToSearchResults}
                onShowDetails={handleShowDetails}
              />
            </div>
          </div>
        </div>
      </div>

      <ResultDetailsDialog
        result={selectedResult}
        isOpen={isDetailsOpen}
        onOpenChange={setIsDetailsOpen}
      />
    </>
  );
}
