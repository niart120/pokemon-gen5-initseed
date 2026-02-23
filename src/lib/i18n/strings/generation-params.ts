import type { SeedSourceMode } from '@/types/generation';
import type { LocaleMap, LocaleText } from './types';

type AbilityModeValue = 'none' | 'sync' | 'compound';

export const generationParamsPanelTitle: LocaleText = {
  ja: 'Generation Parameters',
  en: 'Generation Parameters',
};

export const generationParamsSectionTitles = {
  target: {
    ja: '基本設定',
    en: 'Basic Settings',
  } satisfies LocaleText,
  encounter: {
    ja: 'エンカウント',
    en: 'Encounter',
  } satisfies LocaleText,
  stopConditions: {
    ja: '停止条件',
    en: 'Stop Conditions',
  } satisfies LocaleText,
};

export const generationParamsBaseSeedLabel: LocaleText = {
  ja: '初期Seed',
  en: 'Base Seed',
};

export const generationParamsBaseSeedPlaceholder: LocaleText = {
  ja: '1a2b3c4d5e6f7890',
  en: '1a2b3c4d5e6f7890',
};

export const generationParamsSeedSourceLabel: LocaleText = {
  ja: 'Seed入力モード',
  en: 'Seed Input Mode',
};

export const generationParamsSeedSourceOptionLabels: LocaleMap<Record<SeedSourceMode, string>> = {
  ja: {
    lcg: 'LCG Seed',
    'boot-timing': 'Boot-Timing',
  },
  en: {
    lcg: 'LCG Seed',
    'boot-timing': 'Boot-Timing',
  },
};

export const generationParamsMinAdvanceLabel: LocaleText = {
  ja: '最小消費数',
  en: 'Min Advance',
};

export const generationParamsMaxAdvanceLabel: LocaleText = {
  ja: '最大消費数',
  en: 'Max Advance',
};

export const generationParamsEncounterCategoryLabel: LocaleText = {
  ja: 'カテゴリ',
  en: 'Category',
};

export const generationParamsEncounterTypeLabel: LocaleText = {
  ja: '分類',
  en: 'Type',
};

export const generationParamsEncounterFieldLabel: LocaleText = {
  ja: 'フィールド',
  en: 'Field',
};

export const generationParamsEncounterSpeciesLabel: LocaleText = {
  ja: 'ポケモン',
  en: 'Species',
};

export const generationParamsAbilityLabel: LocaleText = {
  ja: '特性',
  en: 'Ability',
};

export const generationParamsSyncNatureLabel: LocaleText = {
  ja: 'シンクロ性格',
  en: 'Sync Nature',
};

export const generationParamsTypeUnavailablePlaceholder: LocaleText = {
  ja: '選択不可',
  en: 'Unavailable',
};

export const generationParamsSelectOptionPlaceholder: LocaleText = {
  ja: '選択してください',
  en: 'Select...',
};

export const generationParamsNotApplicablePlaceholder: LocaleText = {
  ja: '該当なし',
  en: 'N/A',
};

export const generationParamsSelectSpeciesPlaceholder: LocaleText = {
  ja: 'ポケモンを選択',
  en: 'Select species',
};

export const generationParamsDataUnavailablePlaceholder: LocaleText = {
  ja: 'データなし',
  en: 'Data unavailable',
};

export const generationParamsStaticDataPendingLabel: LocaleText = {
  ja: 'データはまだ利用できません',
  en: 'Data not yet available',
};

export const generationParamsNoTypesAvailableLabel: LocaleText = {
  ja: '選択可能なタイプがありません',
  en: 'No types available',
};

export const generationParamsStopFirstShinyLabel: LocaleText = {
  ja: '最初の色違いで停止',
  en: 'Stop at First Shiny',
};

export const generationParamsStopOnCapLabel: LocaleText = {
  ja: '上限到達で停止',
  en: 'Stop On Cap',
};

export const generationParamsBootTimingTimestampLabel: LocaleText = {
  ja: '起動日時 (ローカル)',
  en: 'Boot Time (Local)',
};

export const generationParamsBootTimingTimestampPlaceholder: LocaleText = {
  ja: 'YYYY-MM-DD hh:mm:ss',
  en: 'YYYY-MM-DD hh:mm:ss',
};

export const generationParamsBootTimingKeyInputLabel: LocaleText = {
  ja: 'キー入力',
  en: 'Key Input',
};

export const generationParamsBootTimingConfigureLabel: LocaleText = {
  ja: 'Configure',
  en: 'Configure',
};

export const generationParamsBootTimingKeyDialogTitle: LocaleText = {
  ja: 'キー入力の設定',
  en: 'Key Input Configuration',
};

export const generationParamsBootTimingKeyResetLabel: LocaleText = {
  ja: 'Reset',
  en: 'Reset',
};

export const generationParamsBootTimingKeyApplyLabel: LocaleText = {
  ja: 'Apply',
  en: 'Apply',
};

export const generationParamsBootTimingProfileLabel: LocaleText = {
  ja: 'Device Profile',
  en: 'Device Profile',
};

export const generationParamsScreenReaderAnnouncement: LocaleText = {
  ja: '生成パラメータ設定です。生成の実行中は編集できません。',
  en: 'Generation parameters configuration. Editing is disabled while generation is active.',
};

export const generationParamsAbilityOptionLabels: LocaleMap<Record<AbilityModeValue, string>> = {
  ja: {
    none: '-',
    sync: 'シンクロ',
    compound: 'ふくがん (WIP)',
  },
  en: {
    none: '-',
    sync: 'Sync',
    compound: 'Compound (WIP)',
  },
};
