import { Funnel } from '@phosphor-icons/react';
import { Button } from '../../ui/button';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '../../ui/input';
import { Label } from '../../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import { ExportButton } from './ExportButton';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import type { SearchResult } from '../../../types/search';

export type SortField = 'datetime' | 'seed' | 'timer0' | 'vcount';

interface ResultsControlCardProps {
  filteredResultsCount: number;
  convertedResults: SearchResult[];
  filterSeed: string;
  setFilterSeed: (value: string) => void;
  sortField: SortField;
  setSortField: (field: SortField) => void;
}

export function ResultsControlCard({
  filteredResultsCount,
  convertedResults,
  filterSeed,
  setFilterSeed,
  sortField,
  setSortField,
}: ResultsControlCardProps) {
  const { isStack } = useResponsiveLayout();
  const { searchResults, clearSearchResults } = useAppStore();

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title="Results Control"
      headerActions={
        <div className="flex gap-2">
          <ExportButton 
            results={convertedResults}
            disabled={filteredResultsCount === 0}
          />
          <Button 
            variant="destructive" 
            size="sm" 
            onClick={clearSearchResults}
            disabled={searchResults.length === 0}
          >
            Clear Results
          </Button>
        </div>
      }
      className={isStack ? 'max-h-96' : undefined}
      fullHeight={!isStack}
      scrollMode="parent"
      contentClassName="overflow-hidden"
    >
        {/* Filters */}
        <div className="flex gap-4 items-end flex-shrink-0">
          <div className="flex-1">
            <Label htmlFor="filter-seed">Filter by Seed</Label>
            <Input
              id="filter-seed"
              placeholder="Enter seed value (hex)"
              value={filterSeed}
              onChange={(e) => setFilterSeed(e.target.value)}
              className="font-mono"
            />
          </div>
          <div>
            <Label htmlFor="sort-field">Sort by</Label>
            <Select value={sortField} onValueChange={(value) => setSortField(value as SortField)}>
              <SelectTrigger id="sort-field" className="w-28 sm:w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="datetime">Date/Time</SelectItem>
                <SelectItem value="seed">MT Seed</SelectItem>
                <SelectItem value="timer0">Timer0</SelectItem>
                <SelectItem value="vcount">VCount</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
    </PanelCard>
  );
}
