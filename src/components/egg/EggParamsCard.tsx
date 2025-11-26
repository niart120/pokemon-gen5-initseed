import React, { useMemo } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Gear } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { useLocale } from '@/lib/i18n/locale-context';
import { natureName } from '@/lib/utils/format-display';
import { DOMAIN_NATURE_COUNT } from '@/types/domain';
import type { IvSet, EggSeedSourceMode } from '@/types/egg';
import { EggBootTimingControls, type EggBootTimingLabels } from './EggBootTimingControls';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  eggParamsPanelTitle,
  eggParamsSectionTitles,
  eggParamsBaseSeedLabel,
  eggParamsBaseSeedPlaceholder,
  eggParamsUserOffsetLabel,
  eggParamsCountLabel,
  eggParentsMaleLabel,
  eggParentsFemaleLabel,
  eggParamsUsesDittoLabel,
  eggParamsEverstoneLabel,
  eggParamsEverstoneNone,
  eggParamsGenderRatioLabel,
  eggGenderRatioPresets,
  eggParamsNidoranFlagLabel,
  eggParamsFemaleAbilityLabel,
  eggParamsFemaleAbilityOptions,
  eggParamsMasudaMethodLabel,
  eggParamsNpcConsumptionLabel,
  eggParamsIvUnknownLabel,
  eggSeedSourceModeLabel,
  eggSeedSourceModeOptions,
  eggBootTimingLabels,
  eggParamsStatNames,
} from '@/lib/i18n/strings/egg-params';

// 16進数入力の正規化
function normalizeHexInput(value: string): string {
  return value.replace(/[^0-9a-fA-F]/g, '').toUpperCase();
}

/**
 * EggParamsCard
 * タマゴ生成パラメータ入力カード
 */
export const EggParamsCard: React.FC = () => {
  const {
    draftParams,
    updateDraftParams,
    updateDraftConditions,
    updateDraftParentsMale,
    updateDraftParentsFemale,
    status,
  } = useEggStore();

  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
  const disabled = status === 'running' || status === 'starting';

  const femaleAbilityOptions = resolveLocaleValue(eggParamsFemaleAbilityOptions, locale);
  const statNames = resolveLocaleValue(eggParamsStatNames, locale);

  // Boot-Timing ラベル解決
  const bootTimingLabelsResolved: EggBootTimingLabels = useMemo(() => ({
    timestamp: eggBootTimingLabels.timestamp[locale],
    timestampPlaceholder: eggBootTimingLabels.timestampPlaceholder[locale],
    keyInput: eggBootTimingLabels.keyInput[locale],
    profile: eggBootTimingLabels.profile[locale],
    configure: eggBootTimingLabels.configure[locale],
    dialogTitle: eggBootTimingLabels.dialogTitle[locale],
    reset: eggBootTimingLabels.reset[locale],
    apply: eggBootTimingLabels.apply[locale],
  }), [locale]);

  // 性別比プリセットの選択値を計算
  const genderRatioValue = useMemo(() => {
    const { threshold, genderless } = draftParams.conditions.genderRatio;
    const preset = eggGenderRatioPresets.find(
      p => p.threshold === threshold && p.genderless === genderless
    );
    return preset ? `${preset.threshold}-${preset.genderless}` : 'custom';
  }, [draftParams.conditions.genderRatio]);

  // 親IV変更ハンドラ（入力中はバリデーションなし）
  const handleIvChange = (
    parent: 'male' | 'female',
    index: number,
    value: string
  ) => {
    // 入力中はそのまま保存（空入力は0として扱う）
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const newIvs = [...currentIvs] as IvSet;
    const numValue = parseInt(value, 10);
    newIvs[index] = Number.isNaN(numValue) ? 0 : numValue;

    if (parent === 'male') {
      updateDraftParentsMale(newIvs);
    } else {
      updateDraftParentsFemale(newIvs);
    }
  };

  // 親IVフォーカスアウト時のバリデーション
  const handleIvBlur = (
    parent: 'male' | 'female',
    index: number
  ) => {
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const currentValue = currentIvs[index];
    
    // 0-31にクランプ
    const clampedValue = Math.min(31, Math.max(0, currentValue));
    if (clampedValue !== currentValue) {
      const newIvs = [...currentIvs] as IvSet;
      newIvs[index] = clampedValue;
      if (parent === 'male') {
        updateDraftParentsMale(newIvs);
      } else {
        updateDraftParentsFemale(newIvs);
      }
    }
  };

  const handleIvUnknownChange = (
    parent: 'male' | 'female',
    index: number,
    isUnknown: boolean
  ) => {
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const newIvs = [...currentIvs] as IvSet;
    // unknown時は32を設定、チェック解除時は0に戻す
    newIvs[index] = isUnknown ? 32 : 0;

    if (parent === 'male') {
      updateDraftParentsMale(newIvs);
    } else {
      updateDraftParentsFemale(newIvs);
    }
  };

  const handleGenderRatioChange = (value: string) => {
    const preset = eggGenderRatioPresets.find(
      p => `${p.threshold}-${p.genderless}` === value
    );
    if (preset) {
      updateDraftConditions({
        genderRatio: {
          threshold: preset.threshold,
          genderless: preset.genderless,
        },
      });
    }
  };

  return (
    <PanelCard
      icon={<Gear size={20} className="opacity-80" />}
      title={<span id="egg-params-title">{eggParamsPanelTitle[locale]}</span>}
      className={isStack ? 'max-h-200' : 'min-h-64'}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="egg-params-title"
      role="form"
    >
      {/* 基本設定セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{eggParamsSectionTitles.basic[locale]}</h4>

        <div className="space-y-3">
          {/* Seed入力モード切り替え */}
          <div className="flex flex-col gap-1">
            <Label className="text-xs">{eggSeedSourceModeLabel[locale]}</Label>
            <ToggleGroup
              type="single"
              value={draftParams.seedSourceMode}
              onValueChange={(value) => {
                if (value) updateDraftParams({ seedSourceMode: value as EggSeedSourceMode });
              }}
              disabled={disabled}
              className="justify-start"
            >
              <ToggleGroupItem value="lcg" aria-label="LCG mode" className="text-xs px-3 h-8">
                {eggSeedSourceModeOptions.lcg[locale]}
              </ToggleGroupItem>
              <ToggleGroupItem value="boot-timing" aria-label="Boot-Timing mode" className="text-xs px-3 h-8">
                {eggSeedSourceModeOptions['boot-timing'][locale]}
              </ToggleGroupItem>
            </ToggleGroup>
          </div>

          {/* LCGモード: 初期Seed入力 */}
          {draftParams.seedSourceMode === 'lcg' && (
            <div className="grid grid-cols-3 gap-1">
              <div className="flex flex-col gap-1">
                <Label className="text-xs" htmlFor="egg-base-seed">{eggParamsBaseSeedLabel[locale]}</Label>
                <Input
                  id="egg-base-seed"
                  data-testid="egg-base-seed"
                  value={draftParams.baseSeedHex}
                  onChange={(e) => updateDraftParams({ baseSeedHex: normalizeHexInput(e.target.value) })}
                  disabled={disabled}
                  placeholder={eggParamsBaseSeedPlaceholder[locale]}
                  className="font-mono text-xs"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs" htmlFor="egg-user-offset">{eggParamsUserOffsetLabel[locale]}</Label>
                <Input
                  id="egg-user-offset"
                  data-testid="egg-user-offset"
                  type="number"
                  min={0}
                  value={parseInt(draftParams.userOffsetHex, 16) || 0}
                  onChange={(e) => {
                    updateDraftParams({ userOffsetHex: e.target.value });
                  }}
                  onBlur={(e) => {
                    const num = Math.max(0, parseInt(e.target.value) || 0);
                    updateDraftParams({ userOffsetHex: num.toString(16).toUpperCase() });
                  }}
                  disabled={disabled}
                  className="text-xs"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs" htmlFor="egg-count">{eggParamsCountLabel[locale]}</Label>
                <Input
                  id="egg-count"
                  data-testid="egg-count"
                  type="number"
                  min={1}
                  max={100000}
                  value={draftParams.count}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    updateDraftParams({ count: Number.isNaN(v) ? 0 : v });
                  }}
                  onBlur={() => {
                    const num = Math.max(1, Math.min(100000, draftParams.count || 1));
                    updateDraftParams({ count: num });
                  }}
                  disabled={disabled}
                  className="text-xs"
                />
              </div>
            </div>
          )}

          {/* Boot-Timingモード: 起動時間パラメータ入力 */}
          {draftParams.seedSourceMode === 'boot-timing' && (
            <div className="space-y-3">
              <EggBootTimingControls
                locale={locale}
                disabled={disabled}
                isActive={draftParams.seedSourceMode === 'boot-timing'}
                labels={bootTimingLabelsResolved}
              />
              <div className="grid grid-cols-2 gap-1">
                <div className="flex flex-col gap-1">
                  <Label className="text-xs" htmlFor="egg-user-offset-bt">{eggParamsUserOffsetLabel[locale]}</Label>
                  <Input
                    id="egg-user-offset-bt"
                    data-testid="egg-user-offset-bt"
                    type="number"
                    min={0}
                    value={parseInt(draftParams.userOffsetHex, 16) || 0}
                    onChange={(e) => {
                      updateDraftParams({ userOffsetHex: e.target.value });
                    }}
                    onBlur={(e) => {
                      const num = Math.max(0, parseInt(e.target.value) || 0);
                      updateDraftParams({ userOffsetHex: num.toString(16).toUpperCase() });
                    }}
                    disabled={disabled}
                    className="text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs" htmlFor="egg-count-bt">{eggParamsCountLabel[locale]}</Label>
                  <Input
                    id="egg-count-bt"
                    data-testid="egg-count-bt"
                    type="number"
                    min={1}
                    max={100000}
                    value={draftParams.count}
                    onChange={(e) => {
                      const v = parseInt(e.target.value, 10);
                      updateDraftParams({ count: Number.isNaN(v) ? 0 : v });
                    }}
                    onBlur={() => {
                      const num = Math.max(1, Math.min(100000, draftParams.count || 1));
                      updateDraftParams({ count: num });
                    }}
                    disabled={disabled}
                    className="text-xs"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <Separator className="my-3" />

      {/* 親個体情報セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{eggParamsSectionTitles.parents[locale]}</h4>

        {/* ♂親IV */}
        <div className="space-y-1">
          <Label className="text-xs">{eggParentsMaleLabel[locale]}</Label>
          <div className="grid grid-cols-6 gap-1">
            {statNames.map((stat, i) => {
              const isUnknown = draftParams.parents.male[i] === 32;
              return (
                <div key={i} className="flex flex-col items-center">
                  <span className="text-[10px] text-muted-foreground">{stat}</span>
                  <Input
                    type="number"
                    min={0}
                    max={31}
                    data-testid={`egg-male-iv-${i}`}
                    value={isUnknown ? '' : draftParams.parents.male[i]}
                    onChange={(e) => handleIvChange('male', i, e.target.value)}
                    onBlur={() => handleIvBlur('male', i)}
                    disabled={disabled || isUnknown}
                    className="text-xs text-center h-7 px-1"
                    placeholder={isUnknown ? '?' : undefined}
                  />
                  <div className="flex items-center gap-1 mt-1">
                    <Checkbox
                      id={`egg-male-iv-unknown-${i}`}
                      checked={isUnknown}
                      onCheckedChange={(checked) => handleIvUnknownChange('male', i, !!checked)}
                      disabled={disabled}
                      className="h-3 w-3"
                    />
                    <Label htmlFor={`egg-male-iv-unknown-${i}`} className="text-[9px] text-muted-foreground cursor-pointer">
                      {eggParamsIvUnknownLabel[locale]}
                    </Label>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ♀親IV */}
        <div className="space-y-1">
          <Label className="text-xs">{eggParentsFemaleLabel[locale]}</Label>
          <div className="grid grid-cols-6 gap-1">
            {statNames.map((stat, i) => {
              const isUnknown = draftParams.parents.female[i] === 32;
              return (
                <div key={i} className="flex flex-col items-center">
                  <span className="text-[10px] text-muted-foreground">{stat}</span>
                  <Input
                    type="number"
                    min={0}
                    max={31}
                    data-testid={`egg-female-iv-${i}`}
                    value={isUnknown ? '' : draftParams.parents.female[i]}
                    onChange={(e) => handleIvChange('female', i, e.target.value)}
                    onBlur={() => handleIvBlur('female', i)}
                    disabled={disabled || isUnknown}
                    className="text-xs text-center h-7 px-1"
                    placeholder={isUnknown ? '?' : undefined}
                  />
                  <div className="flex items-center gap-1 mt-1">
                    <Checkbox
                      id={`egg-female-iv-unknown-${i}`}
                      checked={isUnknown}
                      onCheckedChange={(checked) => handleIvUnknownChange('female', i, !!checked)}
                      disabled={disabled}
                      className="h-3 w-3"
                    />
                    <Label htmlFor={`egg-female-iv-unknown-${i}`} className="text-[9px] text-muted-foreground cursor-pointer">
                      {eggParamsIvUnknownLabel[locale]}
                    </Label>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <Separator className="my-3" />

      {/* 生成条件セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">{eggParamsSectionTitles.conditions[locale]}</h4>

        {/* セレクト系: 横2列 */}
        <div className="grid grid-cols-3 gap-2">
          {/* 性別比 */}
          <div className="flex flex-col gap-1">
            <Label className="text-xs">{eggParamsGenderRatioLabel[locale]}</Label>
            <Select
              value={genderRatioValue}
              onValueChange={handleGenderRatioChange}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {eggGenderRatioPresets.map((preset) => (
                  <SelectItem
                    key={`${preset.threshold}-${preset.genderless}`}
                    value={`${preset.threshold}-${preset.genderless}`}
                    className="text-xs"
                  >
                    {preset.label[locale]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* ♀親の特性 */}
          <div className="flex flex-col gap-1">
            <Label className="text-xs">{eggParamsFemaleAbilityLabel[locale]}</Label>
            <Select
              value={String(draftParams.conditions.femaleParentAbility)}
              onValueChange={(v) => updateDraftConditions({ femaleParentAbility: Number(v) as 0 | 1 | 2 })}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {([0, 1, 2] as const).map((ability) => (
                  <SelectItem key={ability} value={String(ability)} className="text-xs">
                    {femaleAbilityOptions[ability]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {/* かわらずのいし */}
          <div className="flex flex-col gap-1">
            <Label className="text-xs">{eggParamsEverstoneLabel[locale]}</Label>
            <Select
              value={draftParams.conditions.everstone.type === 'none' ? 'none' : `fixed-${(draftParams.conditions.everstone as { type: 'fixed'; nature: number }).nature}`}
              onValueChange={(v) => {
                if (v === 'none') {
                  updateDraftConditions({ everstone: { type: 'none' } });
                } else {
                  const nature = parseInt(v.replace('fixed-', ''));
                  updateDraftConditions({ everstone: { type: 'fixed', nature } });
                }
              }}
              disabled={disabled}
            >
              <SelectTrigger className="text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none" className="text-xs">{eggParamsEverstoneNone[locale]}</SelectItem>
                {Array.from({ length: DOMAIN_NATURE_COUNT }, (_, i) => (
                  <SelectItem key={i} value={`fixed-${i}`} className="text-xs">
                    {natureName(i, locale)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        {/* チェックボックス群: 2列グリッド */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          {/* メタモン利用 */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="egg-uses-ditto"
              checked={draftParams.conditions.usesDitto}
              onCheckedChange={(checked) => updateDraftConditions({ usesDitto: !!checked })}
              disabled={disabled}
            />
            <Label htmlFor="egg-uses-ditto" className="text-xs">{eggParamsUsesDittoLabel[locale]}</Label>
          </div>

          {/* 国際孵化 */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="egg-masuda"
              checked={draftParams.conditions.masudaMethod}
              onCheckedChange={(checked) => updateDraftConditions({ masudaMethod: !!checked })}
              disabled={disabled}
            />
            <Label htmlFor="egg-masuda" className="text-xs">{eggParamsMasudaMethodLabel[locale]}</Label>
          </div>

          {/* ニドラン系 */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="egg-nidoran"
              checked={draftParams.conditions.hasNidoranFlag}
              onCheckedChange={(checked) => updateDraftConditions({ hasNidoranFlag: !!checked })}
              disabled={disabled}
            />
            <Label htmlFor="egg-nidoran" className="text-xs">{eggParamsNidoranFlagLabel[locale]}</Label>
          </div>

          {/* NPC消費考慮 */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="egg-npc-consumption"
              checked={draftParams.considerNpcConsumption}
              onCheckedChange={(checked) => updateDraftParams({ considerNpcConsumption: !!checked })}
              disabled={disabled}
            />
            <Label htmlFor="egg-npc-consumption" className="text-xs">{eggParamsNpcConsumptionLabel[locale]}</Label>
          </div>
        </div>
      </section>
    </PanelCard>
  );
};
