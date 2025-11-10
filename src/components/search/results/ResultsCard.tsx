import { ChevronDown, ChevronUp, Eye } from 'lucide-react';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../../ui/table';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { lcgSeedToHex } from '@/lib/utils/lcg-seed';
import type { InitialSeedResult } from '../../../types/search';
import type { SortField } from './ResultsControlCard';

interface ResultsCardProps {
  filteredAndSortedResults: InitialSeedResult[];
  searchResultsLength: number;
  sortField: SortField;
  sortOrder: 'asc' | 'desc';
  onSort: (field: SortField) => void;
  onShowDetails: (result: InitialSeedResult) => void;
}

export function ResultsCard({
  filteredAndSortedResults,
  searchResultsLength,
  sortField,
  sortOrder,
  onSort,
  onShowDetails,
}: ResultsCardProps) {
  const { lastSearchDuration } = useAppStore();
  const { isStack } = useResponsiveLayout();
  const formatDateTime = (date: Date): string => {
    return `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
  };

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) return null;
    return sortOrder === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />;
  };

  const handleSort = (field: SortField) => {
    onSort(field);
  };

  const filteredResultsCount = filteredAndSortedResults.length;

  return (
    <Card className={`py-2 flex flex-col ${isStack ? 'max-h-96' : 'h-full min-h-96'}`}>
      <CardHeader className="pb-0 flex-shrink-0">
        <CardTitle className="flex items-center gap-2 flex-wrap">
          <Eye size={20} className="flex-shrink-0 opacity-80" />
          <span className="flex-shrink-0">Search Results</span>
          <Badge variant="secondary" className="flex-shrink-0">
            {filteredResultsCount} result{filteredResultsCount !== 1 ? 's' : ''}
          </Badge>
          {lastSearchDuration !== null && (
            <Badge variant="outline" className="flex-shrink-0 text-xs">
              Search completed in {(lastSearchDuration / 1000).toFixed(1)}s
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col flex-1 min-h-0 p-0 overflow-y-auto">
        {filteredAndSortedResults.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            {searchResultsLength === 0 
              ? "No search results yet. Run a search to see results here."
              : "No results match the current filter criteria."
            }
          </div>
        ) : (
          <div className="overflow-y-auto flex-1">
            <Table className="table-auto min-w-full text-xs leading-tight">
              <TableHeader>
                <TableRow className="h-9">
                  <TableHead className="w-12 px-1 text-center"></TableHead>
                  <TableHead className="px-2 font-mono text-[11px] whitespace-nowrap min-w-[120px]">LCG Seed</TableHead>
                  <TableHead 
                    className="px-2 cursor-pointer select-none"
                    onClick={() => handleSort('datetime')}
                  >
                    <div className="flex items-center gap-1">
                      Date/Time {getSortIcon('datetime')}
                    </div>
                  </TableHead>
                  <TableHead 
                    className="px-2 cursor-pointer select-none"
                    onClick={() => handleSort('seed')}
                  >
                    <div className="flex items-center gap-1">
                      MT Seed {getSortIcon('seed')}
                    </div>
                  </TableHead>
                  <TableHead 
                    className="px-2 cursor-pointer select-none"
                    onClick={() => handleSort('timer0')}
                  >
                    <div className="flex items-center gap-1">
                      Timer0 {getSortIcon('timer0')}
                    </div>
                  </TableHead>
                  <TableHead 
                    className="px-2 cursor-pointer select-none"
                    onClick={() => handleSort('vcount')}
                  >
                    <div className="flex items-center gap-1">
                      VCount {getSortIcon('vcount')}
                    </div>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredAndSortedResults.map((result, index) => (
                  <TableRow key={index} className="h-9">
                    <TableCell className="px-1 py-1 text-center">
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => onShowDetails(result)}
                        className="h-7 w-7 p-0"
                        title="View Details"
                        aria-label="View search result details"
                      >
                        <Eye size={14} />
                      </Button>
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-nowrap min-w-[120px]">
                      {lcgSeedToHex(result.lcgSeed)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-normal">
                      {formatDateTime(result.datetime)}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-normal">
                      0x{result.seed.toString(16).toUpperCase().padStart(8, '0')}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-normal">
                      0x{result.timer0.toString(16).toUpperCase().padStart(4, '0')}
                    </TableCell>
                    <TableCell className="px-2 py-1 font-mono text-[11px] leading-tight whitespace-normal">
                      0x{result.vcount.toString(16).toUpperCase().padStart(2, '0')}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
