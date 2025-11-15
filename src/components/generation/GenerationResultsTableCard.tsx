import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Table } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { selectFilteredSortedResults, selectUiReadyResults, type GenerationSlice } from '@/store/generation-store';
import { pidHex, natureName, shinyLabel, seedHex, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';

interface GenerationResultsTableCardProps { parentManagesScroll?: boolean; }
type AppStoreState = ReturnType<typeof useAppStore.getState>;
export const GenerationResultsTableCard: React.FC<GenerationResultsTableCardProps> = ({ parentManagesScroll }) => {
  const locale = useLocale();
  // 元の生結果 (フィルタ/ソート適用済み GenerationResult)
  const rawResults = useAppStore(selectFilteredSortedResults);
  const total = useAppStore(s => s.results.length);
  // UI表示用解決結果（ロケール依存）
  const uiResults = useAppStore((state: AppStoreState) => {
    const generationState: GenerationSlice = state;
    const overrideState: GenerationSlice = { ...generationState, results: rawResults };
    return selectUiReadyResults(overrideState, locale);
  });
  const { isStack } = useResponsiveLayout();
  // スクロール方針
  // - モバイル(isStack): カード内でスクロール(overflow-y-auto)にして、ドキュメント高さの膨張を防ぐ
  // - デスクトップ: 呼び出し元の指定を尊重（既定はfalseでカード内スクロール）
  const effectiveParentManages = isStack ? false : !!parentManagesScroll;
  return (
    <PanelCard
      icon={<Table size={20} className="opacity-80" />}
      title={<span id="gen-results-table-title">Results ({rawResults.length}) / Total {total}</span>}
      className={isStack ? 'max-h-96' : 'min-h-96'}
      fullHeight={!isStack}
      scrollMode={effectiveParentManages ? 'parent' : 'content'}
      padding="none"
      spacing="none"
      contentClassName="p-0"
      aria-labelledby="gen-results-table-title"
      role="region"
    >
      <table className="min-w-full text-xs" aria-describedby="gen-results-table-desc">
          <caption id="gen-results-table-desc" className="sr-only">Filtered generation results list.</caption>
          <thead className="sticky top-0 bg-muted text-[11px]">
            <tr className="text-left">
              <th scope="col" className="px-2 py-1 font-medium w-14">Adv<span className="sr-only">ance</span></th>
              <th scope="col" className="px-2 py-1 font-medium w-10">dir</th>
              <th scope="col" className="px-2 py-1 font-medium w-8">v</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32">Species</th>
              <th scope="col" className="px-2 py-1 font-medium w-24">Nature</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32 hidden md:table-cell">Ability</th>
              <th scope="col" className="px-2 py-1 font-medium w-8">G<span className="sr-only">ender</span></th>
              <th scope="col" className="px-2 py-1 font-medium w-10">Lv</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">HP</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">Atk</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">Def</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">SpA</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">SpD</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">Spe</th>
              <th scope="col" className="px-2 py-1 font-medium w-16">Shiny</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[120px] w-36">Seed</th>
              <th scope="col" className="px-2 py-1 font-medium w-32">PID</th>
            </tr>
          </thead>
          <tbody>
            {rawResults.map((r, idx) => {
              const u = uiResults[idx];
              const needleDir = calculateNeedleDirection(r.seed);
              const stats = u?.stats;
              const hpDisplay = stats ? stats.hp : '--';
              const atkDisplay = stats ? stats.attack : '--';
              const defDisplay = stats ? stats.defense : '--';
              const spaDisplay = stats ? stats.specialAttack : '--';
              const spdDisplay = stats ? stats.specialDefense : '--';
              const speDisplay = stats ? stats.speed : '--';
              const natureDisplay = u?.natureName ?? natureName(r.nature, locale);
              return (
                <tr key={r.advance} className="odd:bg-background even:bg-muted/30">
                  <td className="px-2 py-1 font-mono tabular-nums">{r.advance}</td>
                  <td className="px-2 py-1 text-center font-arrows">{needleDirectionArrow(needleDir)}</td>
                  <td className="px-2 py-1 font-mono tabular-nums">{needleDir}</td>
                  <td className="px-2 py-1 truncate max-w-[120px]" title={u?.speciesName || 'Unknown'}>{u?.speciesName || 'Unknown'}</td>
                  <td className="px-2 py-1 whitespace-nowrap">{natureDisplay}</td>
                  <td className="px-2 py-1 truncate max-w-[120px] hidden md:table-cell" title={u?.abilityName || 'Unknown'}>{u?.abilityName || 'Unknown'}</td>
                  <td className="px-2 py-1">{u?.gender || '?'}</td>
                  <td className="px-2 py-1 tabular-nums">{u?.level ?? ''}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{hpDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{atkDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{defDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{spaDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{spdDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{speDisplay}</td>
                  <td className="px-2 py-1">{shinyLabel(r.shiny_type, locale)}</td>
                  <td className="px-2 py-1 font-mono whitespace-nowrap">{seedHex(r.seed)}</td>
                  <td className="px-2 py-1 font-mono whitespace-nowrap">{pidHex(r.pid)}</td>
                </tr>
              );
            })}
          </tbody>
      </table>
      <div className="sr-only" aria-live="polite">{rawResults.length} filtered results shown of {total} total.</div>
    </PanelCard>
  );
};
