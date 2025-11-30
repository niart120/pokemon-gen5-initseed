import { Funnel, Trash } from '@phosphor-icons/react';
import { Button } from '../../ui/button';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '../../ui/input';
import { Label } from '../../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatSearchResultsSortOption,
  searchResultsControlClearButton,
  searchResultsControlFilterLabel,
  searchResultsControlFilterPlaceholder,
  searchResultsControlSortLabel,
  searchResultsControlSortPlaceholder,
  searchResultsControlTitle,
  type SearchResultsSortKey,
} from '@/lib/i18n/strings/search-results-control';

export type SortField = SearchResultsSortKey;

interface ResultsControlCardProps {
  filterSeed: string;
  setFilterSeed: (value: string) => void;
  sortField: SortField;
  setSortField: (field: SortField) => void;
}

export function ResultsControlCard({
  filterSeed,
  setFilterSeed,
  sortField,
  setSortField,
}: ResultsControlCardProps) {
  const { isStack } = useResponsiveLayout();
  const { searchResults, clearSearchResults } = useAppStore();
  const locale = useLocale();
  const title = resolveLocaleValue(searchResultsControlTitle, locale);
  const clearLabel = resolveLocaleValue(searchResultsControlClearButton, locale);
  const filterLabel = resolveLocaleValue(searchResultsControlFilterLabel, locale);
  const filterPlaceholder = resolveLocaleValue(searchResultsControlFilterPlaceholder, locale);
  const sortLabel = resolveLocaleValue(searchResultsControlSortLabel, locale);
  const sortPlaceholder = resolveLocaleValue(searchResultsControlSortPlaceholder, locale);

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={title}
      headerActions={
        <div className="flex gap-2">
          <Button 
            variant="destructive" 
            size="sm" 
            onClick={clearSearchResults}
            disabled={searchResults.length === 0}
            className="gap-1"
          >
            <Trash size={14} />
            {clearLabel}
          </Button>
        </div>
      }
      className={isStack ? 'max-h-96' : undefined}
      fullHeight={!isStack}
    >
        {/* Filters */}
        <div className="flex gap-4 items-end flex-shrink-0">
          <div className="flex-1">
            <Label htmlFor="filter-seed">{filterLabel}</Label>
            <Input
              id="filter-seed"
              placeholder={filterPlaceholder}
              value={filterSeed}
              onChange={(e) => setFilterSeed(e.target.value)}
              className="font-mono"
            />
          </div>
          <div>
            <Label htmlFor="sort-field">{sortLabel}</Label>
            <Select value={sortField} onValueChange={(value) => setSortField(value as SortField)}>
              <SelectTrigger id="sort-field" className="w-28 sm:w-32">
                <SelectValue placeholder={sortPlaceholder} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="datetime">{formatSearchResultsSortOption('datetime', locale)}</SelectItem>
                <SelectItem value="seed">{formatSearchResultsSortOption('seed', locale)}</SelectItem>
                <SelectItem value="timer0">{formatSearchResultsSortOption('timer0', locale)}</SelectItem>
                <SelectItem value="vcount">{formatSearchResultsSortOption('vcount', locale)}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
    </PanelCard>
  );
}
