import React, { useRef } from 'react';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/store/app-store';
import { selectThroughputEma, selectEtaFormatted, selectShinyCount } from '@/store/generation-store';

/**
 * CompactProgressBadge (Phase1 placeholder)
 * - 現状: 小型表示の土台のみ。`hidden` で非表示。
 * - Phase2: 差分 2% / 5秒などの閾値で aria-live 更新 & tooltip 詳細
 */
export const CompactProgressBadge: React.FC = () => {
  const progress = useAppStore((s) => s.progress);
  const draftParams = useAppStore((s) => s.draftParams);
  const status = useAppStore((s) => s.status);
  const throughput = useAppStore(selectThroughputEma);
  const eta = useAppStore(selectEtaFormatted);
  const shiny = useAppStore(selectShinyCount);
  const lastAnnouncedPct = useRef<number | null>(null);

  const total = progress?.totalAdvances ?? draftParams.maxAdvances ?? 0;
  const done = progress?.processedAdvances ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;

  // 将来: pct 差分 >=2% の時のみ aria-live で announce
  const announceText = `${pct.toFixed(1)}% ${done}/${total} adv | ${throughput ? throughput.toFixed(1) + ' adv/s' : '--'} | ETA ${eta ?? '--:--'} | Shiny ${shiny}`;
  const shouldAnnounce = lastAnnouncedPct.current === null || Math.abs(pct - (lastAnnouncedPct.current ?? 0)) >= 2;
  if (shouldAnnounce) lastAnnouncedPct.current = pct;

  return (
    <div className="hidden lg:flex items-center gap-2 min-h-[32px]" data-testid="gen-progress-badge">
      <Badge variant="secondary" className="font-mono text-xs px-2 py-1">
        {pct ? pct.toFixed(1) : '0.0'}%
      </Badge>
      <div className="text-[11px] font-mono text-muted-foreground flex items-center gap-2">
        <span>{done}/{total}</span>
        <span>{throughput ? `${throughput.toFixed(1)} adv/s` : '--'}</span>
        <span>ETA {eta ?? '--:--'}</span>
        <span>Sh {shiny}</span>
        <span className="uppercase">{status}</span>
      </div>
      <div className="sr-only" aria-live="polite">
        {shouldAnnounce ? announceText : ''}
      </div>
    </div>
  );
};
