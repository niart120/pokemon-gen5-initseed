import type { LocaleText, LocaleMap } from './types';
import type { EggGameMode } from '@/types/egg';

export const eggParamsPanelTitle: LocaleText = {
  ja: 'タマゴ生成パラメータ',
  en: 'Egg Generation Parameters',
};

export const eggParamsSectionTitles = {
  basic: {
    ja: '基本設定',
    en: 'Basic Settings',
  } satisfies LocaleText,
  parents: {
    ja: '親個体情報',
    en: 'Parent Information',
  } satisfies LocaleText,
  conditions: {
    ja: '生成条件',
    en: 'Generation Conditions',
  } satisfies LocaleText,
  other: {
    ja: 'その他設定',
    en: 'Other Settings',
  } satisfies LocaleText,
};

export const eggParamsBaseSeedLabel: LocaleText = {
  ja: '初期Seed',
  en: 'Base Seed',
};

export const eggParamsUserOffsetLabel: LocaleText = {
  ja: '開始advance',
  en: 'Start Advance',
};

export const eggParamsCountLabel: LocaleText = {
  ja: '列挙上限',
  en: 'Enumeration Limit',
};

export const eggParamsGameModeLabel: LocaleText = {
  ja: 'ゲームモード',
  en: 'Game Mode',
};

export const eggParamsGameModeOptions: LocaleMap<Record<EggGameMode, string>> = {
  ja: {
    0: 'BW 新規',
    1: 'BW 続きから',
    2: 'BW2 新規',
    3: 'BW2 続きから',
  },
  en: {
    0: 'BW New Game',
    1: 'BW Continue',
    2: 'BW2 New Game',
    3: 'BW2 Continue',
  },
};

export const eggParentsMaleLabel: LocaleText = {
  ja: '♂親 IV',
  en: '♂ Parent IV',
};

export const eggParentsFemaleLabel: LocaleText = {
  ja: '♀親 IV',
  en: '♀ Parent IV',
};

export const eggParamsUsesDittoLabel: LocaleText = {
  ja: 'メタモン利用',
  en: 'Using Ditto',
};

export const eggParamsEverstoneLabel: LocaleText = {
  ja: 'かわらずのいし',
  en: 'Everstone',
};

export const eggParamsEverstoneNone: LocaleText = {
  ja: 'なし',
  en: 'None',
};

export const eggParamsGenderRatioLabel: LocaleText = {
  ja: '性別比閾値 (0-255)',
  en: 'Gender Ratio Threshold (0-255)',
};

export const eggParamsGenderlessLabel: LocaleText = {
  ja: '無性別',
  en: 'Genderless',
};

export const eggParamsNidoranFlagLabel: LocaleText = {
  ja: 'ニドラン系/バルビート系',
  en: 'Nidoran/Volbeat Family',
};

export const eggParamsAllowHiddenLabel: LocaleText = {
  ja: '夢特性許可',
  en: 'Allow Hidden Ability',
};

export const eggParamsFemaleHiddenLabel: LocaleText = {
  ja: '親♀が夢特性を持つ',
  en: 'Female Parent Has Hidden Ability',
};

export const eggParamsRerollCountLabel: LocaleText = {
  ja: '国際孵化リロール回数 (0-5)',
  en: 'Masuda Method Reroll Count (0-5)',
};

export const eggParamsTidLabel: LocaleText = {
  ja: 'TID',
  en: 'TID',
};

export const eggParamsSidLabel: LocaleText = {
  ja: 'SID',
  en: 'SID',
};

export const eggParamsNpcConsumptionLabel: LocaleText = {
  ja: 'NPC消費を考慮',
  en: 'Consider NPC Consumption',
};
