import { ChevronDown, ChevronUp, Eye, Hash } from 'lucide-react';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../../ui/table';
import { useAppStore } from '../../../store/app-store';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
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
    return sortOrder === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />;
  };

  const handleSort = (field: SortField) => {
    onSort(field);
  };

  const filteredResultsCount = filteredAndSortedResults.length;

  return (
    <Card className={`py-2 flex flex-col ${isStack ? 'max-h-96' : 'h-full min-h-96'}`}>
      <CardHeader className="pb-0 flex-shrink-0">
        <CardTitle className="flex items-center gap-2 flex-wrap">
          <Hash size={20} className="flex-shrink-0" />
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
          <div className="overflow-auto flex-1">
            <Table className="table-auto min-w-full">
              <TableHeader>
                <TableRow>
                  <TableHead 
                    className="cursor-pointer select-none whitespace-normal sm:whitespace-nowrap min-w-[80px]"
                    onClick={() => handleSort('seed')}
                  >
                    <div className="flex items-center gap-1">
                      Seed Value {getSortIcon('seed')}
                    </div>
                  </TableHead>
                  <TableHead 
                    className="cursor-pointer select-none whitespace-normal sm:whitespace-nowrap min-w-[100px]"
                    onClick={() => handleSort('datetime')}
                  >
                    <div className="flex items-center gap-1">
                      Date/Time {getSortIcon('datetime')}
                    </div>
                  </TableHead>
                  <TableHead 
                    className="cursor-pointer select-none whitespace-normal sm:whitespace-nowrap min-w-[70px]"
                    onClick={() => handleSort('timer0')}
                  >
                    <div className="flex items-center gap-1">
                      Timer0 {getSortIcon('timer0')}
                    </div>
                  </TableHead>
                  <TableHead 
                    className="cursor-pointer select-none whitespace-normal sm:whitespace-nowrap min-w-[70px]"
                    onClick={() => handleSort('vcount')}
                  >
                    <div className="flex items-center gap-1">
                      VCount {getSortIcon('vcount')}
                    </div>
                  </TableHead>
                  <TableHead className="whitespace-normal sm:whitespace-nowrap min-w-[80px]">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredAndSortedResults.map((result, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-mono whitespace-normal sm:whitespace-nowrap break-all sm:break-normal min-w-[80px]">
                      0x{result.seed.toString(16).toUpperCase().padStart(8, '0')}
                    </TableCell>
                    <TableCell className="whitespace-normal sm:whitespace-nowrap min-w-[100px]">
                      <span className="font-mono text-sm break-all sm:break-normal">
                        {formatDateTime(result.datetime)}
                      </span>
                    </TableCell>
                    <TableCell className="font-mono whitespace-normal sm:whitespace-nowrap break-all sm:break-normal min-w-[70px]">0x{result.timer0.toString(16).toUpperCase().padStart(4, '0')}</TableCell>
                    <TableCell className="font-mono whitespace-normal sm:whitespace-nowrap break-all sm:break-normal min-w-[70px]">0x{result.vcount.toString(16).toUpperCase().padStart(2, '0')}</TableCell>
                    <TableCell className="whitespace-normal sm:whitespace-nowrap min-w-[80px]">
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => onShowDetails(result)}
                        className="text-xs px-2 py-1"
                      >
                        <Eye size={14} className="mr-1" />
                        Details
                      </Button>
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
