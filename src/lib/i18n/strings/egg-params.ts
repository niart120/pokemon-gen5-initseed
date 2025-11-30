import type { LocaleText, LocaleMap } from './types';

export const eggParamsPanelTitle: LocaleText = {
  ja: 'Generation Parameters',
  en: 'Generation Parameters',
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

export const eggParamsBaseSeedPlaceholder: LocaleText = {
  ja: '例: 1234567890ABCDEF',
  en: 'e.g. 1234567890ABCDEF',
};

export const eggParamsUserOffsetLabel: LocaleText = {
  ja: '開始消費',
  en: 'Start Frame',
};

export const eggParamsCountLabel: LocaleText = {
  ja: '最大消費数',
  en: 'Max Frames',
};

export const eggParentsMaleLabel: LocaleText = {
  ja: '♂親 個体値',
  en: '♂ Parent IV',
};

export const eggParentsFemaleLabel: LocaleText = {
  ja: '♀親 個体値',
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
  ja: '性別比',
  en: 'Gender Ratio',
};

/**
 * 性別比選択肢
 * 閾値は実際のゲーム内判定に基づく
 */
export type GenderRatioPreset = {
  threshold: number;
  genderless: boolean;
  label: LocaleText;
};

export const eggGenderRatioPresets: GenderRatioPreset[] = [
  { threshold: 127, genderless: false, label: { ja: '♂1:♀1', en: '♂1:♀1' } },
  { threshold: 64, genderless: false, label: { ja: '♂3:♀1', en: '♂3:♀1' } },
  { threshold: 31, genderless: false, label: { ja: '♂7:♀1', en: '♂7:♀1' } },
  { threshold: 191, genderless: false, label: { ja: '♂1:♀3', en: '♂1:♀3' } },
  { threshold: 0, genderless: true, label: { ja: '性別不明', en: 'Genderless' } },
];

export const eggParamsNidoranFlagLabel: LocaleText = {
  ja: 'ニドラン♂♀ / イルミーゼ・バルビート',
  en: 'Nidoran♂♀ / Illumise・Volbeat',
};

export const eggParamsFemaleAbilityLabel: LocaleText = {
  ja: '♀親の特性',
  en: 'Female Parent Ability',
};

export const eggParamsFemaleAbilityOptions: LocaleMap<Record<0 | 1 | 2, string>> = {
  ja: {
    0: '特性1',
    1: '特性2',
    2: '隠れ特性',
  },
  en: {
    0: 'Ability 1',
    1: 'Ability 2',
    2: 'Hidden Ability',
  },
};

export const eggParamsMasudaMethodLabel: LocaleText = {
  ja: '国際孵化',
  en: 'Masuda Method',
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

export const eggParamsIvUnknownLabel: LocaleText = {
  ja: '不明',
  en: 'Unknown',
};

// === Boot-Timing Labels ===

export const eggSeedSourceModeLabel: LocaleText = {
  ja: 'Seed入力モード',
  en: 'Seed Input Mode',
};

export const eggSeedSourceModeOptions = {
  lcg: {
    ja: 'LCG Seed',
    en: 'LCG Seed',
  } satisfies LocaleText,
  'boot-timing': {
    ja: 'Boot-Timing',
    en: 'Boot-Timing',
  } satisfies LocaleText,
};

/**
 * ステータス名（親IV入力用）
 * JPN: H/A/B/C/D/S 表記
 * ENG: HP/Atk/Def/SpA/SpD/Spe 表記
 */
export const eggParamsStatNames: LocaleMap<[string, string, string, string, string, string]> = {
  ja: ['H', 'A', 'B', 'C', 'D', 'S'],
  en: ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'],
};

export const eggBootTimingLabels = {
  timestamp: {
    ja: '起動時刻',
    en: 'Boot Time',
  } satisfies LocaleText,
  timestampPlaceholder: {
    ja: '起動時刻を入力',
    en: 'Enter boot time',
  } satisfies LocaleText,
  keyInput: {
    ja: 'キー入力',
    en: 'Key Input',
  } satisfies LocaleText,
  profile: {
    ja: 'プロファイル情報',
    en: 'Profile Info',
  } satisfies LocaleText,
  configure: {
    ja: '設定',
    en: 'Configure',
  } satisfies LocaleText,
  dialogTitle: {
    ja: 'キー入力設定',
    en: 'Key Input Settings',
  } satisfies LocaleText,
  reset: {
    ja: 'リセット',
    en: 'Reset',
  } satisfies LocaleText,
  apply: {
    ja: '適用',
    en: 'Apply',
  } satisfies LocaleText,
};
