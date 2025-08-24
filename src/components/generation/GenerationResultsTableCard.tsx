import React from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Table } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { selectFilteredSortedResults, selectUiReadyResults } from '@/store/generation-store';
import { pidHex, natureName, shinyLabel } from '@/lib/utils/format-display';
import { useResponsiveLayout } from '@/hooks/use-mobile';

interface GenerationResultsTableCardProps { parentManagesScroll?: boolean; }
export const GenerationResultsTableCard: React.FC<GenerationResultsTableCardProps> = ({ parentManagesScroll }) => {
  // 元の生結果 (フィルタ/ソート適用済み GenerationResult)
  const rawResults = useAppStore(selectFilteredSortedResults);
  const total = useAppStore(s => s.results.length);
  // UI表示用解決結果 (locale: 今は固定 'ja')
  const uiResults = useAppStore(s => selectUiReadyResults({ ...s, results: rawResults } as any, 'ja'));
  const { isStack } = useResponsiveLayout();
  // スクロール方針
  // - モバイル(isStack): カード内でスクロール(overflow-y-auto)にして、ドキュメント高さの膨張を防ぐ
  // - デスクトップ: 呼び出し元の指定を尊重（既定はfalseでカード内スクロール）
  const effectiveParentManages = isStack ? false : !!parentManagesScroll;
  return (
    <Card className={`py-2 flex flex-col ${isStack ? '' : 'h-full min-h-64'}`} aria-labelledby="gen-results-table-title" role="region">
  <StandardCardHeader icon={<Table size={20} className="opacity-80" />} title={<span id="gen-results-table-title">Results ({rawResults.length}) / Total {total}</span>} />
      <StandardCardContent className="p-0" noScroll={effectiveParentManages}>
        <table className="min-w-full text-xs" aria-describedby="gen-results-table-desc">
          <caption id="gen-results-table-desc" className="sr-only">Filtered generation results list.</caption>
          <thead className="sticky top-0 bg-muted text-[11px]">
            <tr className="text-left">
              <th scope="col" className="px-2 py-1 font-medium w-14">Adv<span className="sr-only">ance</span></th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32">Species</th>
              <th scope="col" className="px-2 py-1 font-medium w-32">PID</th>
              <th scope="col" className="px-2 py-1 font-medium w-24">Nature</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32 hidden md:table-cell">Ability</th>
              <th scope="col" className="px-2 py-1 font-medium w-8">G<span className="sr-only">ender</span></th>
              <th scope="col" className="px-2 py-1 font-medium w-10">Lv</th>
              <th scope="col" className="px-2 py-1 font-medium w-16">Shiny</th>
            </tr>
          </thead>
          <tbody>
            {rawResults.map((r, idx) => {
              const u = uiResults[idx];
              return (
                <tr key={r.advance} className="odd:bg-background even:bg-muted/30">
                  <td className="px-2 py-1 font-mono tabular-nums">{r.advance}</td>
                  <td className="px-2 py-1 truncate max-w-[120px]" title={u?.speciesName || 'Unknown'}>{u?.speciesName || 'Unknown'}</td>
                  <td className="px-2 py-1 font-mono whitespace-nowrap">{pidHex(r.pid)}</td>
                  <td className="px-2 py-1 whitespace-nowrap">{natureName(r.nature)}</td>
                  <td className="px-2 py-1 truncate max-w-[120px] hidden md:table-cell" title={u?.abilityName || 'Unknown'}>{u?.abilityName || 'Unknown'}</td>
                  <td className="px-2 py-1">{u?.gender || '?'}</td>
                  <td className="px-2 py-1 tabular-nums">{u?.level ?? ''}</td>
                  <td className="px-2 py-1">{shinyLabel(r.shiny_type)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="sr-only" aria-live="polite">{rawResults.length} filtered results shown of {total} total.</div>
      </StandardCardContent>
    </Card>
  );
};
