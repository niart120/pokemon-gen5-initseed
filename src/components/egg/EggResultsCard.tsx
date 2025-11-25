import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Table } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { IV_UNKNOWN } from '@/types/egg';

// 性格名配列
const NATURE_NAMES = [
  'がんばりや', 'さみしがり', 'ゆうかん', 'いじっぱり', 'やんちゃ',
  'ずぶとい', 'すなお', 'のんき', 'わんぱく', 'のうてんき',
  'おくびょう', 'せっかち', 'まじめ', 'ようき', 'むじゃき',
  'ひかえめ', 'おっとり', 'れいせい', 'てれや', 'うっかりや',
  'おだやか', 'おとなしい', 'なまいき', 'しんちょう', 'きまぐれ',
];

const HP_TYPE_NAMES = [
  'かくとう', 'ひこう', 'どく', 'じめん', 'いわ', 'むし', 'ゴースト', 'はがね',
  'ほのお', 'みず', 'くさ', 'でんき', 'エスパー', 'こおり', 'ドラゴン', 'あく',
];

const SHINY_LABELS = ['通常', '正方形', '星型'];

const GENDER_LABELS: Record<string, string> = {
  male: '♂',
  female: '♀',
  genderless: '-',
};

const ABILITY_LABELS = ['特性1', '特性2', '夢特性'];

/**
 * EggResultsCard
 * タマゴ生成結果表示カード
 */
export const EggResultsCard: React.FC = () => {
  const { results, draftParams } = useEggStore();
  const { isStack } = useResponsiveLayout();

  const formatIv = (iv: number): string => {
    return iv === IV_UNKNOWN ? '?' : String(iv);
  };

  const formatHiddenPower = (hp: { type: 'known'; hpType: number; power: number } | { type: 'unknown' }): string => {
    if (hp.type === 'unknown') {
      return '?/?';
    }
    return `${HP_TYPE_NAMES[hp.hpType] || '?'}/${hp.power}`;
  };

  return (
    <PanelCard
      icon={<Table size={20} className="opacity-80" />}
      title={<span id="egg-results-title">生成結果 ({results.length}/{draftParams.count})</span>}
      className={isStack ? 'min-h-64' : undefined}
      fullHeight={!isStack}
      scrollMode="content"
      aria-labelledby="egg-results-title"
    >
      <div className="overflow-x-auto" data-testid="egg-results-table">
        <table className="w-full text-xs border-collapse">
          <thead className="sticky top-0 bg-background z-10">
            <tr className="border-b">
              <th className="text-left py-1 px-2 font-medium">Adv</th>
              <th className="text-center py-1 px-1 font-medium">HP</th>
              <th className="text-center py-1 px-1 font-medium">Atk</th>
              <th className="text-center py-1 px-1 font-medium">Def</th>
              <th className="text-center py-1 px-1 font-medium">SpA</th>
              <th className="text-center py-1 px-1 font-medium">SpD</th>
              <th className="text-center py-1 px-1 font-medium">Spe</th>
              <th className="text-left py-1 px-2 font-medium">性格</th>
              <th className="text-center py-1 px-1 font-medium">性別</th>
              <th className="text-left py-1 px-2 font-medium">特性</th>
              <th className="text-center py-1 px-1 font-medium">色</th>
              <th className="text-left py-1 px-2 font-medium">めざパ</th>
              <th className="text-left py-1 px-2 font-medium font-mono">PID</th>
              {draftParams.considerNpcConsumption && (
                <th className="text-center py-1 px-1 font-medium">安定</th>
              )}
            </tr>
          </thead>
          <tbody>
            {results.length === 0 ? (
              <tr>
                <td colSpan={draftParams.considerNpcConsumption ? 14 : 13} className="text-center py-8 text-muted-foreground">
                  結果がありません
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
                  <td className="py-1 px-2">{NATURE_NAMES[row.egg.nature] || row.egg.nature}</td>
                  <td className="text-center py-1 px-1">{GENDER_LABELS[row.egg.gender] || row.egg.gender}</td>
                  <td className="py-1 px-2">{ABILITY_LABELS[row.egg.ability] || row.egg.ability}</td>
                  <td className="text-center py-1 px-1">
                    {row.egg.shiny > 0 ? (
                      <span className={row.egg.shiny === 2 ? 'text-yellow-500' : 'text-blue-500'}>
                        {SHINY_LABELS[row.egg.shiny]}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </td>
                  <td className="py-1 px-2">{formatHiddenPower(row.egg.hiddenPower)}</td>
                  <td className="py-1 px-2 font-mono text-[10px]">
                    {row.egg.pid.toString(16).toUpperCase().padStart(8, '0')}
                  </td>
                  {draftParams.considerNpcConsumption && (
                    <td className="text-center py-1 px-1">
                      {row.isStable ? '○' : '-'}
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
