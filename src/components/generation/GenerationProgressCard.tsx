import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/app-store';
import { selectThroughputEma, selectEtaFormatted, selectShinyCount } from '@/store/generation-store';

export const GenerationProgressCard: React.FC = () => {
  const { draftParams, progress } = useAppStore();
  const throughput = useAppStore(selectThroughputEma);
  const eta = useAppStore(selectEtaFormatted);
  const shinyCount = useAppStore(selectShinyCount);

  return (
    <Card className="p-3 flex flex-col gap-2">
      <CardHeader className="py-0"><CardTitle className="text-sm">Progress / Metrics</CardTitle></CardHeader>
      <CardContent className="p-0 text-xs grid grid-cols-2 md:grid-cols-5 gap-2">
        <div>Adv: {progress?.processedAdvances ?? 0}/{progress?.totalAdvances ?? draftParams.maxAdvances}</div>
        <div>Results: {useAppStore.getState().results.length}</div>
        <div>Shiny: {shinyCount}</div>
        <div>Thruput: {throughput?throughput.toFixed(1)+' adv/s':'--'}</div>
        <div>ETA: {eta ?? '--:--'}</div>
      </CardContent>
    </Card>
  );
};
