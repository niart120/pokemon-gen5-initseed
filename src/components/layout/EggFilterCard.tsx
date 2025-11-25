import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Funnel } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { createDefaultEggFilter, type StatRange, type EggIndividualFilter } from '@/types/egg';

// 性格名配列
const NATURE_NAMES = [
  'がんばりや', 'さみしがり', 'ゆうかん', 'いじっぱり', 'やんちゃ',
  'ずぶとい', 'すなお', 'のんき', 'わんぱく', 'のうてんき',
  'おくびょう', 'せっかち', 'まじめ', 'ようき', 'むじゃき',
  'ひかえめ', 'おっとり', 'れいせい', 'てれや', 'うっかりや',
  'おだやか', 'おとなしい', 'なまいき', 'しんちょう', 'きまぐれ',
];

const STAT_NAMES = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'];

const HP_TYPE_NAMES = [
  'かくとう', 'ひこう', 'どく', 'じめん', 'いわ', 'むし', 'ゴースト', 'はがね',
  'ほのお', 'みず', 'くさ', 'でんき', 'エスパー', 'こおり', 'ドラゴン', 'あく',
];

const GENDER_OPTIONS = [
  { value: '', label: '指定なし' },
  { value: 'male', label: '♂' },
  { value: 'female', label: '♀' },
  { value: 'genderless', label: '無性別' },
];

const ABILITY_OPTIONS = [
  { value: '', label: '指定なし' },
  { value: '0', label: '特性1' },
  { value: '1', label: '特性2' },
  { value: '2', label: '夢特性' },
];

const SHINY_OPTIONS = [
  { value: '', label: '指定なし' },
  { value: '0', label: '通常' },
  { value: '1', label: '正方形色違い' },
  { value: '2', label: '星型色違い' },
];

/**
 * EggFilterCard
 * タマゴ個体フィルター設定カード
 */
export const EggFilterCard: React.FC = () => {
  const { draftParams, updateDraftParams, status } = useEggStore();
  const { isStack } = useResponsiveLayout();
  const disabled = status === 'running' || status === 'starting';

  const filter = draftParams.filter || createDefaultEggFilter();

  const updateFilter = (updates: Partial<EggIndividualFilter>) => {
    updateDraftParams({
      filter: { ...filter, ...updates },
    });
  };

  const handleIvRangeChange = (
    statIndex: number,
    minMax: 'min' | 'max',
    value: number
  ) => {
    const newRanges = [...filter.ivRanges] as [StatRange, StatRange, StatRange, StatRange, StatRange, StatRange];
    newRanges[statIndex] = {
      ...newRanges[statIndex],
      [minMax]: Math.max(0, Math.min(32, value)),
    };
    updateFilter({ ivRanges: newRanges });
  };

  const clearFilter = () => {
    updateDraftParams({ filter: null });
  };

  const enableFilter = () => {
    updateDraftParams({ filter: createDefaultEggFilter() });
  };

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={<span id="egg-filter-title">フィルター設定</span>}
      className={isStack ? 'max-h-96' : undefined}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="egg-filter-title"
      role="form"
    >
      {/* フィルター有効/無効 */}
      <div className="flex items-center gap-2 mb-3">
        <Checkbox
          id="egg-filter-enabled"
          checked={draftParams.filter !== null}
          onCheckedChange={(checked) => {
            if (checked) {
              enableFilter();
            } else {
              clearFilter();
            }
          }}
          disabled={disabled}
        />
        <Label htmlFor="egg-filter-enabled" className="text-xs">フィルターを有効にする</Label>
      </div>

      {draftParams.filter && (
        <>
          {/* IV範囲フィルター */}
          <section className="space-y-2" role="group">
            <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">個体値範囲</h4>
            <div className="space-y-2">
              {STAT_NAMES.map((stat, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs w-8">{stat}</span>
                  <Input
                    type="number"
                    min={0}
                    max={32}
                    value={filter.ivRanges[i].min}
                    onChange={(e) => handleIvRangeChange(i, 'min', parseInt(e.target.value) || 0)}
                    disabled={disabled}
                    className="text-xs h-7 w-14 text-center"
                    placeholder="min"
                  />
                  <span className="text-xs">~</span>
                  <Input
                    type="number"
                    min={0}
                    max={32}
                    value={filter.ivRanges[i].max}
                    onChange={(e) => handleIvRangeChange(i, 'max', parseInt(e.target.value) || 32)}
                    disabled={disabled}
                    className="text-xs h-7 w-14 text-center"
                    placeholder="max"
                  />
                </div>
              ))}
            </div>
          </section>

          {/* 性格フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">性格</Label>
            <Select
              value={filter.nature !== undefined ? String(filter.nature) : ''}
              onValueChange={(v) => updateFilter({ nature: v ? Number(v) : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder="指定なし" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="" className="text-xs">指定なし</SelectItem>
                {NATURE_NAMES.map((name, i) => (
                  <SelectItem key={i} value={String(i)} className="text-xs">
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 性別フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">性別</Label>
            <Select
              value={filter.gender || ''}
              onValueChange={(v) => updateFilter({ gender: v ? v as 'male' | 'female' | 'genderless' : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder="指定なし" />
              </SelectTrigger>
              <SelectContent>
                {GENDER_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value} className="text-xs">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 特性フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">特性</Label>
            <Select
              value={filter.ability !== undefined ? String(filter.ability) : ''}
              onValueChange={(v) => updateFilter({ ability: v ? Number(v) as 0 | 1 | 2 : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder="指定なし" />
              </SelectTrigger>
              <SelectContent>
                {ABILITY_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value} className="text-xs">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 色違いフィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">色違い</Label>
            <Select
              value={filter.shiny !== undefined ? String(filter.shiny) : ''}
              onValueChange={(v) => updateFilter({ shiny: v ? Number(v) as 0 | 1 | 2 : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder="指定なし" />
              </SelectTrigger>
              <SelectContent>
                {SHINY_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value} className="text-xs">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* めざパタイプフィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">めざパタイプ</Label>
            <Select
              value={filter.hiddenPowerType !== undefined ? String(filter.hiddenPowerType) : ''}
              onValueChange={(v) => updateFilter({ hiddenPowerType: v ? Number(v) : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder="指定なし" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="" className="text-xs">指定なし</SelectItem>
                {HP_TYPE_NAMES.map((name, i) => (
                  <SelectItem key={i} value={String(i)} className="text-xs">
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* めざパ威力フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">めざパ威力 (30-70)</Label>
            <Input
              type="number"
              min={30}
              max={70}
              value={filter.hiddenPowerPower ?? ''}
              onChange={(e) => {
                const v = e.target.value;
                updateFilter({ hiddenPowerPower: v ? Math.max(30, Math.min(70, parseInt(v))) : undefined });
              }}
              disabled={disabled}
              placeholder="指定なし"
              className="text-xs"
            />
          </div>
        </>
      )}
    </PanelCard>
  );
};
