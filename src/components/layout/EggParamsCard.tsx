import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { Gear } from '@phosphor-icons/react';
import { useEggStore } from '@/store/egg-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { EggGameMode, type IvSet } from '@/types/egg';

// 性格名配列
const NATURE_NAMES = [
  'がんばりや', 'さみしがり', 'ゆうかん', 'いじっぱり', 'やんちゃ',
  'ずぶとい', 'すなお', 'のんき', 'わんぱく', 'のうてんき',
  'おくびょう', 'せっかち', 'まじめ', 'ようき', 'むじゃき',
  'ひかえめ', 'おっとり', 'れいせい', 'てれや', 'うっかりや',
  'おだやか', 'おとなしい', 'なまいき', 'しんちょう', 'きまぐれ',
];

const STAT_NAMES = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'];

const GAME_MODE_OPTIONS = [
  { value: EggGameMode.BwNew, label: 'BW 新規' },
  { value: EggGameMode.BwContinue, label: 'BW 続きから' },
  { value: EggGameMode.Bw2New, label: 'BW2 新規' },
  { value: EggGameMode.Bw2Continue, label: 'BW2 続きから' },
];

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
  const disabled = status === 'running' || status === 'starting';

  const handleIvChange = (
    parent: 'male' | 'female',
    index: number,
    value: string
  ) => {
    const numValue = Math.min(32, Math.max(0, parseInt(value) || 0));
    const currentIvs = parent === 'male' ? draftParams.parents.male : draftParams.parents.female;
    const newIvs = [...currentIvs] as IvSet;
    newIvs[index] = numValue;

    if (parent === 'male') {
      updateDraftParentsMale(newIvs);
    } else {
      updateDraftParentsFemale(newIvs);
    }
  };

  return (
    <PanelCard
      icon={<Gear size={20} className="opacity-80" />}
      title={<span id="egg-params-title">タマゴ生成パラメータ</span>}
      className={isStack ? 'max-h-200' : 'min-h-64'}
      fullHeight={!isStack}
      scrollMode={isStack ? 'parent' : 'content'}
      aria-labelledby="egg-params-title"
      role="form"
    >
      {/* 基本設定セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">基本設定</h4>

        {/* 初期Seed */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="egg-base-seed">初期Seed</Label>
          <Input
            id="egg-base-seed"
            data-testid="egg-base-seed"
            value={draftParams.baseSeedHex}
            onChange={(e) => updateDraftParams({ baseSeedHex: normalizeHexInput(e.target.value) })}
            disabled={disabled}
            placeholder="1234567890ABCDEF"
            className="font-mono text-xs"
          />
        </div>

        {/* 開始advance */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="egg-user-offset">開始advance</Label>
          <Input
            id="egg-user-offset"
            data-testid="egg-user-offset"
            value={draftParams.userOffsetHex}
            onChange={(e) => updateDraftParams({ userOffsetHex: normalizeHexInput(e.target.value) })}
            disabled={disabled}
            placeholder="0"
            className="font-mono text-xs"
          />
        </div>

        {/* 列挙上限 */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="egg-count">列挙上限</Label>
          <Input
            id="egg-count"
            data-testid="egg-count"
            type="number"
            min={1}
            max={100000}
            value={draftParams.count}
            onChange={(e) => updateDraftParams({ count: Math.max(1, Math.min(100000, parseInt(e.target.value) || 1)) })}
            disabled={disabled}
            className="text-xs"
          />
        </div>

        {/* GameMode */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="egg-game-mode">ゲームモード</Label>
          <Select
            value={String(draftParams.gameMode)}
            onValueChange={(v) => updateDraftParams({ gameMode: Number(v) as EggGameMode })}
            disabled={disabled}
          >
            <SelectTrigger id="egg-game-mode" className="text-xs">
              <SelectValue placeholder="選択" />
            </SelectTrigger>
            <SelectContent>
              {GAME_MODE_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={String(opt.value)} className="text-xs">
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </section>

      <Separator className="my-3" />

      {/* 親個体情報セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">親個体情報</h4>

        {/* ♂親IV */}
        <div className="space-y-1">
          <Label className="text-xs">♂親 IV</Label>
          <div className="grid grid-cols-6 gap-1">
            {STAT_NAMES.map((stat, i) => (
              <div key={i} className="flex flex-col items-center">
                <span className="text-[10px] text-muted-foreground">{stat}</span>
                <Input
                  type="number"
                  min={0}
                  max={32}
                  data-testid={`egg-male-iv-${i}`}
                  value={draftParams.parents.male[i]}
                  onChange={(e) => handleIvChange('male', i, e.target.value)}
                  disabled={disabled}
                  className="text-xs text-center h-7 px-1"
                />
              </div>
            ))}
          </div>
        </div>

        {/* ♀親IV */}
        <div className="space-y-1">
          <Label className="text-xs">♀親 IV</Label>
          <div className="grid grid-cols-6 gap-1">
            {STAT_NAMES.map((stat, i) => (
              <div key={i} className="flex flex-col items-center">
                <span className="text-[10px] text-muted-foreground">{stat}</span>
                <Input
                  type="number"
                  min={0}
                  max={32}
                  data-testid={`egg-female-iv-${i}`}
                  value={draftParams.parents.female[i]}
                  onChange={(e) => handleIvChange('female', i, e.target.value)}
                  disabled={disabled}
                  className="text-xs text-center h-7 px-1"
                />
              </div>
            ))}
          </div>
        </div>

        {/* メタモン利用 */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="egg-uses-ditto"
            checked={draftParams.conditions.usesDitto}
            onCheckedChange={(checked) => updateDraftConditions({ usesDitto: !!checked })}
            disabled={disabled}
          />
          <Label htmlFor="egg-uses-ditto" className="text-xs">メタモン利用</Label>
        </div>
      </section>

      <Separator className="my-3" />

      {/* 生成条件セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">生成条件</h4>

        {/* かわらずのいし */}
        <div className="space-y-1">
          <Label className="text-xs">かわらずのいし</Label>
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
              <SelectValue placeholder="選択" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none" className="text-xs">なし</SelectItem>
              {NATURE_NAMES.map((name, i) => (
                <SelectItem key={i} value={`fixed-${i}`} className="text-xs">
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* 性別比 */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">性別比閾値 (0-255)</Label>
          <Input
            type="number"
            min={0}
            max={255}
            value={draftParams.conditions.genderRatio.threshold}
            onChange={(e) => updateDraftConditions({
              genderRatio: {
                ...draftParams.conditions.genderRatio,
                threshold: Math.max(0, Math.min(255, parseInt(e.target.value) || 127)),
              }
            })}
            disabled={disabled}
            className="text-xs"
          />
        </div>

        {/* 無性別 */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="egg-genderless"
            checked={draftParams.conditions.genderRatio.genderless}
            onCheckedChange={(checked) => updateDraftConditions({
              genderRatio: {
                ...draftParams.conditions.genderRatio,
                genderless: !!checked,
              }
            })}
            disabled={disabled}
          />
          <Label htmlFor="egg-genderless" className="text-xs">無性別</Label>
        </div>

        {/* ニドラン系 */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="egg-nidoran"
            checked={draftParams.conditions.hasNidoranFlag}
            onCheckedChange={(checked) => updateDraftConditions({ hasNidoranFlag: !!checked })}
            disabled={disabled}
          />
          <Label htmlFor="egg-nidoran" className="text-xs">ニドラン系/バルビート系</Label>
        </div>

        {/* 夢特性許可 */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="egg-allow-hidden"
            checked={draftParams.conditions.allowHiddenAbility}
            onCheckedChange={(checked) => updateDraftConditions({ allowHiddenAbility: !!checked })}
            disabled={disabled}
          />
          <Label htmlFor="egg-allow-hidden" className="text-xs">夢特性許可</Label>
        </div>

        {/* 親♀夢特性 */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="egg-female-hidden"
            checked={draftParams.conditions.femaleParentHasHidden}
            onCheckedChange={(checked) => updateDraftConditions({ femaleParentHasHidden: !!checked })}
            disabled={disabled}
          />
          <Label htmlFor="egg-female-hidden" className="text-xs">親♀が夢特性を持つ</Label>
        </div>

        {/* 国際孵化リロール */}
        <div className="flex flex-col gap-1">
          <Label className="text-xs">国際孵化リロール回数 (0-5)</Label>
          <Input
            type="number"
            min={0}
            max={5}
            value={draftParams.conditions.rerollCount}
            onChange={(e) => updateDraftConditions({ rerollCount: Math.max(0, Math.min(5, parseInt(e.target.value) || 0)) })}
            disabled={disabled}
            className="text-xs"
          />
        </div>

        {/* TID/SID */}
        <div className="grid grid-cols-2 gap-2">
          <div className="flex flex-col gap-1">
            <Label className="text-xs">TID</Label>
            <Input
              type="number"
              min={0}
              max={65535}
              value={draftParams.conditions.tid}
              onChange={(e) => updateDraftConditions({ tid: Math.max(0, Math.min(65535, parseInt(e.target.value) || 0)) })}
              disabled={disabled}
              className="text-xs"
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-xs">SID</Label>
            <Input
              type="number"
              min={0}
              max={65535}
              value={draftParams.conditions.sid}
              onChange={(e) => updateDraftConditions({ sid: Math.max(0, Math.min(65535, parseInt(e.target.value) || 0)) })}
              disabled={disabled}
              className="text-xs"
            />
          </div>
        </div>
      </section>

      <Separator className="my-3" />

      {/* その他設定セクション */}
      <section className="space-y-3" role="group">
        <h4 className="text-xs font-medium text-muted-foreground tracking-wide uppercase">その他設定</h4>

        {/* NPC消費考慮 */}
        <div className="flex items-center gap-2">
          <Checkbox
            id="egg-npc-consumption"
            checked={draftParams.considerNpcConsumption}
            onCheckedChange={(checked) => updateDraftParams({ considerNpcConsumption: !!checked })}
            disabled={disabled}
          />
          <Label htmlFor="egg-npc-consumption" className="text-xs">NPC消費を考慮</Label>
        </div>
      </section>
    </PanelCard>
  );
};
