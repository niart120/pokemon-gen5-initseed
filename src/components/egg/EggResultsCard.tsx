import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Table } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName, calculateNeedleDirection, needleDirectionArrow } from '@/lib/utils/format-display';
import { IV_UNKNOWN, type EnumeratedEggDataWithBootTiming } from '@/types/egg';
import {
  eggResultsPanelTitle,
  eggResultsEmptyMessage,
  getEggResultHeader,
  eggResultShinyLabels,
  eggResultGenderLabels,
  eggResultAbilityLabels,
  eggResultStableLabels,
  eggResultUnknownHp,
} from '@/lib/i18n/strings/egg-results';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';

/**
 * EggResultsCard
 * タマゴ生成結果表示カード
 */
export const EggResultsCard: React.FC = () => {
  const { results, draftParams, getFilteredResults } = useEggStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();

  const isBootTimingMode = draftParams.seedSourceMode === 'boot-timing';
  const filteredResults = getFilteredResults();

  // advanceでソート
  const sortedResults = React.useMemo(() => {
    return [...filteredResults].sort((a, b) => a.advance - b.advance);
  }, [filteredResults]);

  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;
  const shinyLabels = resolveLocaleValue(eggResultShinyLabels, locale);
  const genderLabels = resolveLocaleValue(eggResultGenderLabels, locale);
  const abilityLabels = resolveLocaleValue(eggResultAbilityLabels, locale);
  const stableLabels = resolveLocaleValue(eggResultStableLabels, locale);

  const formatIv = (iv: number): string => {
    return iv === IV_UNKNOWN ? '?' : String(iv);
  };

  const formatHiddenPower = (hp: { type: 'known'; hpType: number; power: number } | { type: 'unknown' }): string => {
    if (hp.type === 'unknown') {
      return eggResultUnknownHp[locale];
    }
    return `${hpTypeNames[hp.hpType] || '?'}/${hp.power}`;
  };

  const formatTimer0 = (row: EnumeratedEggDataWithBootTiming): string => {
    if (row.timer0 === undefined) return '-';
    return `0x${row.timer0.toString(16).toUpperCase().padStart(4, '0')}`;
  };

  const formatVcount = (row: EnumeratedEggDataWithBootTiming): string => {
    if (row.vcount === undefined) return '-';
    return `0x${row.vcount.toString(16).toUpperCase().padStart(2, '0')}`;
  };

  // Seedから方向を計算
  const getDirection = (row: EnumeratedEggDataWithBootTiming): { arrow: string; value: number } | null => {
    if (!row.seedSourceSeedHex) return null;
    try {
      const seed = BigInt(row.seedSourceSeedHex);
      const value = calculateNeedleDirection(seed);
      const arrow = needleDirectionArrow(value);
      return { arrow, value };
    } catch {
      return null;
    }
  };

  // Boot-Timing モード時の列数計算
  const baseColSpan = 15; // advance + dir + v + ability + gender + nature + shiny + 6 IVs + pid + hp
  const npcColSpan = draftParams.considerNpcConsumption ? 1 : 0;
  const bootTimingColSpan = isBootTimingMode ? 2 : 0;
  const totalColSpan = baseColSpan + npcColSpan + bootTimingColSpan;

  return (
    <PanelCard
      icon={<Table size={20} className="opacity-80" />}
      title={<span id="egg-results-title">{eggResultsPanelTitle[locale]} ({sortedResults.length}/{results.length})</span>}
      className={isStack ? 'min-h-64' : undefined}
      fullHeight={!isStack}
      scrollMode="content"
      aria-labelledby="egg-results-title"
    >
      <div className="overflow-x-auto" data-testid="egg-results-table">
        <table className="w-full text-xs border-collapse">
          <thead className="sticky top-0 bg-background z-10">
            <tr className="border-b">
              <th className="text-left py-1 px-2 font-bold">{getEggResultHeader('advance', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('dir', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('dirValue', locale)}</th>
              <th className="text-left py-1 px-2 font-bold">{getEggResultHeader('ability', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('gender', locale)}</th>
              <th className="text-left py-1 px-2 font-bold">{getEggResultHeader('nature', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('shiny', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('hp', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('atk', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('def', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('spa', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('spd', locale)}</th>
              <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('spe', locale)}</th>
              <th className="text-left py-1 px-2 font-bold font-mono">{getEggResultHeader('pid', locale)}</th>
              <th className="text-left py-1 px-2 font-bold">{getEggResultHeader('hiddenPower', locale)}</th>
              {isBootTimingMode && (
                <>
                  <th className="text-left py-1 px-2 font-bold font-mono">Timer0</th>
                  <th className="text-left py-1 px-2 font-bold font-mono">VCount</th>
                </>
              )}
              {draftParams.considerNpcConsumption && (
                <th className="text-center py-1 px-1 font-bold">{getEggResultHeader('stable', locale)}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {sortedResults.length === 0 ? (
              <tr>
                <td colSpan={totalColSpan} className="text-center py-8 text-muted-foreground">
                  {eggResultsEmptyMessage[locale]}
                </td>
              </tr>
            ) : (
              sortedResults.map((row, i) => (
                <tr key={i} className="border-b hover:bg-muted/50" data-testid="egg-result-row">
                  <td className="py-1 px-2 font-mono">{row.advance}</td>
                  <td className="text-center py-1 px-1">{getDirection(row)?.arrow ?? '-'}</td>
                  <td className="text-center py-1 px-1 font-mono">{getDirection(row)?.value ?? '-'}</td>
                  <td className="py-1 px-2">{abilityLabels[row.egg.ability as 0 | 1 | 2] || row.egg.ability}</td>
                  <td className="text-center py-1 px-1">{genderLabels[row.egg.gender] || row.egg.gender}</td>
                  <td className="py-1 px-2">{natureName(row.egg.nature, locale)}</td>
                  <td className="text-center py-1 px-1">
                    {row.egg.shiny > 0 ? (
                      <span className={row.egg.shiny === 2 ? 'text-yellow-500' : 'text-blue-500'}>
                        {shinyLabels[row.egg.shiny as 1 | 2]}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">{shinyLabels[0]}</span>
                    )}
                  </td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[0])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[1])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[2])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[3])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[4])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[5])}</td>
                  <td className="py-1 px-2 font-mono text-[10px]">
                    {row.egg.pid.toString(16).toUpperCase().padStart(8, '0')}
                  </td>
                  <td className="py-1 px-2">{formatHiddenPower(row.egg.hiddenPower)}</td>
                  {isBootTimingMode && (
                    <>
                      <td className="py-1 px-2 font-mono text-[10px]">{formatTimer0(row)}</td>
                      <td className="py-1 px-2 font-mono text-[10px]">{formatVcount(row)}</td>
                    </>
                  )}
                  {draftParams.considerNpcConsumption && (
                    <td className="text-center py-1 px-1">
                      {row.isStable ? stableLabels.yes : stableLabels.no}
                    </td>
                  )}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </PanelCard>
  );
};
