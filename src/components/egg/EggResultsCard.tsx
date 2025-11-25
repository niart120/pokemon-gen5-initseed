import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Table } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName } from '@/lib/utils/format-display';
import { IV_UNKNOWN } from '@/types/egg';
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

// めざパタイプ名
const HP_TYPE_NAMES = {
  ja: [
    'かくとう', 'ひこう', 'どく', 'じめん', 'いわ', 'むし', 'ゴースト', 'はがね',
    'ほのお', 'みず', 'くさ', 'でんき', 'エスパー', 'こおり', 'ドラゴン', 'あく',
  ],
  en: [
    'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
    'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark',
  ],
} as const;

/**
 * EggResultsCard
 * タマゴ生成結果表示カード
 */
export const EggResultsCard: React.FC = () => {
  const { results, draftParams } = useEggStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();

  const hpTypeNames = HP_TYPE_NAMES[locale] ?? HP_TYPE_NAMES.en;
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

  return (
    <PanelCard
      icon={<Table size={20} className="opacity-80" />}
      title={<span id="egg-results-title">{eggResultsPanelTitle[locale]} ({results.length}/{draftParams.count})</span>}
      className={isStack ? 'min-h-64' : undefined}
      fullHeight={!isStack}
      scrollMode="content"
      aria-labelledby="egg-results-title"
    >
      <div className="overflow-x-auto" data-testid="egg-results-table">
        <table className="w-full text-xs border-collapse">
          <thead className="sticky top-0 bg-background z-10">
            <tr className="border-b">
              <th className="text-left py-1 px-2 font-medium">{getEggResultHeader('advance', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('hp', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('atk', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('def', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('spa', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('spd', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('spe', locale)}</th>
              <th className="text-left py-1 px-2 font-medium">{getEggResultHeader('nature', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('gender', locale)}</th>
              <th className="text-left py-1 px-2 font-medium">{getEggResultHeader('ability', locale)}</th>
              <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('shiny', locale)}</th>
              <th className="text-left py-1 px-2 font-medium">{getEggResultHeader('hiddenPower', locale)}</th>
              <th className="text-left py-1 px-2 font-medium font-mono">{getEggResultHeader('pid', locale)}</th>
              {draftParams.considerNpcConsumption && (
                <th className="text-center py-1 px-1 font-medium">{getEggResultHeader('stable', locale)}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {results.length === 0 ? (
              <tr>
                <td colSpan={draftParams.considerNpcConsumption ? 14 : 13} className="text-center py-8 text-muted-foreground">
                  {eggResultsEmptyMessage[locale]}
                </td>
              </tr>
            ) : (
              results.map((row, i) => (
                <tr key={i} className="border-b hover:bg-muted/50" data-testid="egg-result-row">
                  <td className="py-1 px-2 font-mono">{row.advance}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[0])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[1])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[2])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[3])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[4])}</td>
                  <td className="text-center py-1 px-1 font-mono">{formatIv(row.egg.ivs[5])}</td>
                  <td className="py-1 px-2">{natureName(row.egg.nature, locale)}</td>
                  <td className="text-center py-1 px-1">{genderLabels[row.egg.gender] || row.egg.gender}</td>
                  <td className="py-1 px-2">{abilityLabels[row.egg.ability as 0 | 1 | 2] || row.egg.ability}</td>
                  <td className="text-center py-1 px-1">
                    {row.egg.shiny > 0 ? (
                      <span className={row.egg.shiny === 2 ? 'text-yellow-500' : 'text-blue-500'}>
                        {shinyLabels[row.egg.shiny as 1 | 2]}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">{shinyLabels[0]}</span>
                    )}
                  </td>
                  <td className="py-1 px-2">{formatHiddenPower(row.egg.hiddenPower)}</td>
                  <td className="py-1 px-2 font-mono text-[10px]">
                    {row.egg.pid.toString(16).toUpperCase().padStart(8, '0')}
                  </td>
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
