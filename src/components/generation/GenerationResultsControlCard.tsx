import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/app-store';

export const GenerationResultsControlCard: React.FC = () => {
  const { filters, applyFilters } = useAppStore();
  return (
    <Card className="p-3 flex flex-col gap-2">
      <CardHeader className="py-0"><CardTitle className="text-sm">Results Control (Minimal)</CardTitle></CardHeader>
      <CardContent className="p-0 flex items-center gap-4 text-xs">
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={filters.shinyOnly} onChange={e=>applyFilters({ shinyOnly: e.target.checked })} />
          Shiny Only (view)
        </label>
      </CardContent>
    </Card>
  );
};
