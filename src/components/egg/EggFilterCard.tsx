import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Funnel } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName } from '@/lib/utils/format-display';
import { DOMAIN_NATURE_COUNT } from '@/types/domain';
import { createDefaultEggFilter, type StatRange, type EggIndividualFilter } from '@/types/egg';
import {
  eggFilterPanelTitle,
  eggFilterEnabledLabel,
  eggFilterIvRangeTitle,
  eggFilterNatureLabel,
  eggFilterGenderLabel,
  eggFilterAbilityLabel,
  eggFilterShinyLabel,
  eggFilterHpTypeLabel,
  eggFilterHpPowerLabel,
  eggFilterNoSelection,
  eggFilterGenderOptions,
  eggFilterAbilityOptions,
  eggFilterShinyOptions,
} from '@/lib/i18n/strings/egg-filter';

const STAT_NAMES = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'];

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
 * EggFilterCard
 * タマゴ個体フィルター設定カード
 */
export const EggFilterCard: React.FC = () => {
  const { draftParams, updateDraftParams, status } = useEggStore();
  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
  const disabled = status === 'running' || status === 'starting';

  const filter = draftParams.filter || createDefaultEggFilter();

  // Localized options
  const genderOptions = resolveLocaleValue(eggFilterGenderOptions, locale);
  const abilityOptions = resolveLocaleValue(eggFilterAbilityOptions, locale);
  const shinyOptions = resolveLocaleValue(eggFilterShinyOptions, locale);
  const hpTypeNames = HP_TYPE_NAMES[locale] ?? HP_TYPE_NAMES.en;

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
      title={<span id="egg-filter-title">{eggFilterPanelTitle[locale]}</span>}
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
        <Label htmlFor="egg-filter-enabled" className="text-xs">{eggFilterEnabledLabel[locale]}</Label>
      </div>

      {draftParams.filter && (
        <>
          {/* IV範囲フィルター */}
          <section className="space-y-2" role="group">
            <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{eggFilterIvRangeTitle[locale]}</h4>
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
            <Label className="text-xs">{eggFilterNatureLabel[locale]}</Label>
            <Select
              value={filter.nature !== undefined ? String(filter.nature) : ''}
              onValueChange={(v) => updateFilter({ nature: v ? Number(v) : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={eggFilterNoSelection[locale]} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="" className="text-xs">{eggFilterNoSelection[locale]}</SelectItem>
                {Array.from({ length: DOMAIN_NATURE_COUNT }, (_, i) => (
                  <SelectItem key={i} value={String(i)} className="text-xs">
                    {natureName(i, locale)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 性別フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">{eggFilterGenderLabel[locale]}</Label>
            <Select
              value={filter.gender || ''}
              onValueChange={(v) => updateFilter({ gender: v ? v as 'male' | 'female' | 'genderless' : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={eggFilterNoSelection[locale]} />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(genderOptions).map(([value, label]) => (
                  <SelectItem key={value} value={value} className="text-xs">
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 特性フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">{eggFilterAbilityLabel[locale]}</Label>
            <Select
              value={filter.ability !== undefined ? String(filter.ability) : ''}
              onValueChange={(v) => updateFilter({ ability: v ? Number(v) as 0 | 1 | 2 : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={eggFilterNoSelection[locale]} />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(abilityOptions).map(([value, label]) => (
                  <SelectItem key={value} value={value} className="text-xs">
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 色違いフィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">{eggFilterShinyLabel[locale]}</Label>
            <Select
              value={filter.shiny !== undefined ? String(filter.shiny) : ''}
              onValueChange={(v) => updateFilter({ shiny: v ? Number(v) as 0 | 1 | 2 : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={eggFilterNoSelection[locale]} />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(shinyOptions).map(([value, label]) => (
                  <SelectItem key={value} value={value} className="text-xs">
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* めざパタイプフィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">{eggFilterHpTypeLabel[locale]}</Label>
            <Select
              value={filter.hiddenPowerType !== undefined ? String(filter.hiddenPowerType) : ''}
              onValueChange={(v) => updateFilter({ hiddenPowerType: v ? Number(v) : undefined })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={eggFilterNoSelection[locale]} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="" className="text-xs">{eggFilterNoSelection[locale]}</SelectItem>
                {hpTypeNames.map((name, i) => (
                  <SelectItem key={i} value={String(i)} className="text-xs">
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* めざパ威力フィルター */}
          <div className="flex flex-col gap-1 mt-3">
            <Label className="text-xs">{eggFilterHpPowerLabel[locale]}</Label>
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
              placeholder={eggFilterNoSelection[locale]}
              className="text-xs"
            />
          </div>
        </>
      )}
    </PanelCard>
  );
};
