import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Table } from '@phosphor-icons/react';
import { useAppStore } from '@/store/app-store';
import { selectFilteredDisplayRows } from '@/store/generation-store';
import { shinyLabel, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatGenerationResultsTableAnnouncement,
  formatGenerationResultsTableTitle,
  generationResultsTableCaption,
  generationResultsTableUnknownLabel,
  resolveGenerationResultsTableHeaders,
} from '@/lib/i18n/strings/generation-results-table';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';

interface GenerationResultsTableCardProps { parentManagesScroll?: boolean; }
type AppStoreState = ReturnType<typeof useAppStore.getState>;
export const GenerationResultsTableCard: React.FC<GenerationResultsTableCardProps> = ({ parentManagesScroll }) => {
  const locale = useLocale();
  const rows = useAppStore((state: AppStoreState) => selectFilteredDisplayRows(state, locale));
  const total = useAppStore(s => s.results.length);
  const { isStack } = useResponsiveLayout();
  const headers = React.useMemo(() => resolveGenerationResultsTableHeaders(locale), [locale]);
  const panelTitle = formatGenerationResultsTableTitle(rows.length, total, locale);
  const caption = resolveLocaleValue(generationResultsTableCaption, locale);
  const unknownLabel = resolveLocaleValue(generationResultsTableUnknownLabel, locale);
  const announcement = formatGenerationResultsTableAnnouncement(rows.length, total, locale);
  // スクロール方針
  // - モバイル(isStack): カード内でスクロール(overflow-y-auto)にして、ドキュメント高さの膨張を防ぐ
  // - デスクトップ: 呼び出し元の指定を尊重（既定はfalseでカード内スクロール）
  const effectiveParentManages = isStack ? false : !!parentManagesScroll;
  return (
    <PanelCard
      icon={<Table size={20} className="opacity-80" />}
      title={<span id="gen-results-table-title">{panelTitle}</span>}
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
          <caption id="gen-results-table-desc" className="sr-only">{caption}</caption>
          <thead className="sticky top-0 bg-muted text-[11px]">
            <tr className="text-left">
              <th scope="col" className="px-2 py-1 font-medium w-14">
                {headers.advance.label}
                {headers.advance.sr ? <span className="sr-only">{headers.advance.sr}</span> : null}
              </th>
              <th scope="col" className="px-2 py-1 font-medium w-10">{headers.direction.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-8">{headers.directionValue.label}</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32">{headers.species.label}</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[90px] w-32">{headers.ability.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-8">
                {headers.gender.label}
                {headers.gender.sr ? <span className="sr-only">{headers.gender.sr}</span> : null}
              </th>
              <th scope="col" className="px-2 py-1 font-medium w-24">{headers.nature.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-16">{headers.shiny.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-10">{headers.level.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.hp.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.attack.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.defense.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.specialAttack.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.specialDefense.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-12 text-right">{headers.speed.label}</th>
              <th scope="col" className="px-2 py-1 font-medium min-w-[120px] w-36">{headers.seed.label}</th>
              <th scope="col" className="px-2 py-1 font-medium w-32">{headers.pid.label}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const needleDir = calculateNeedleDirection(row.seed);
              const stats = row.stats;
              const hpDisplay = stats ? stats.hp : '--';
              const atkDisplay = stats ? stats.attack : '--';
              const defDisplay = stats ? stats.defense : '--';
              const spaDisplay = stats ? stats.specialAttack : '--';
              const spdDisplay = stats ? stats.specialDefense : '--';
              const speDisplay = stats ? stats.speed : '--';
              const natureDisplay = row.natureName;
              return (
                <tr key={row.advance} className="odd:bg-background even:bg-muted/30">
                  <td className="px-2 py-1 font-mono tabular-nums">{row.advance}</td>
                  <td className="px-2 py-1 text-center font-arrows">{needleDirectionArrow(needleDir)}</td>
                  <td className="px-2 py-1 font-mono tabular-nums">{needleDir}</td>
                  <td className="px-2 py-1 truncate max-w-[120px]" title={row.speciesName || unknownLabel}>{row.speciesName || unknownLabel}</td>
                  <td className="px-2 py-1 truncate max-w-[120px]" title={row.abilityName || unknownLabel}>{row.abilityName || unknownLabel}</td>
                  <td className="px-2 py-1">{row.gender || '?'}</td>
                  <td className="px-2 py-1 whitespace-nowrap">{natureDisplay}</td>
                  <td className="px-2 py-1">{shinyLabel(row.shinyType, locale)}</td>
                  <td className="px-2 py-1 tabular-nums">{row.level ?? ''}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{hpDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{atkDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{defDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{spaDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{spdDisplay}</td>
                  <td className="px-2 py-1 font-mono tabular-nums text-right">{speDisplay}</td>
                  <td className="px-2 py-1 font-mono whitespace-nowrap">{row.seedHex}</td>
                  <td className="px-2 py-1 font-mono whitespace-nowrap">{row.pidHex}</td>
                </tr>
              );
            })}
          </tbody>
      </table>
      <div className="sr-only" aria-live="polite">{announcement}</div>
    </PanelCard>
  );
};
