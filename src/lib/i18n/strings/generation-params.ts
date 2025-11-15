import type { LocaleMap, LocaleText } from './types';

type AbilityModeValue = 'none' | 'sync' | 'compound';

export const generationParamsPanelTitle: LocaleText = {
  ja: 'Generation Parameters',
  en: 'Generation Parameters',
};

export const generationParamsSectionTitles = {
  target: {
    ja: '目標',
    en: 'Target',
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

export const generationParamsMinAdvanceLabel: LocaleText = {
  ja: '最小消費数',
  en: 'Min Advance',
};

export const generationParamsMaxAdvancesLabel: LocaleText = {
  ja: '最大消費数',
  en: 'Max Advances',
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
