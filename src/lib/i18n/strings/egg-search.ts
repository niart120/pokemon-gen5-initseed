/**
 * i18n strings for Egg Search Panel
 * Based on: spec/agent/pr_egg_boot_timing_search/I18N_DESIGN.md
 */

import type { SupportedLocale } from '@/types/i18n';
import { resolveLocaleValue, type LocaleMap, type LocaleText } from './types';
import type { EggBootTimingSearchStatus } from '@/store/egg-boot-timing-search-store';

// NOTE: BCP47_BY_LOCALE should be moved to common.ts in a future refactor
const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

// === Panel Titles ===

export const eggSearchPanelTitle: LocaleText = {
  ja: 'Search(Egg)',
  en: 'Search(Egg)',
};

// === Run Card ===

export const eggSearchRunCardTitle: LocaleText = {
  ja: '検索制御',
  en: 'Search Control',
};

export const eggSearchStatusPrefix: LocaleText = {
  ja: 'ステータス',
  en: 'Status',
};

export const eggSearchFoundLabel: LocaleText = {
  ja: '発見数',
  en: 'Found',
};

export const eggSearchElapsedLabel: LocaleText = {
  ja: '経過時間',
  en: 'Elapsed',
};

export const eggSearchProgressLabel: LocaleText = {
  ja: '進捗',
  en: 'Progress',
};

export const eggSearchButtonLabels = {
  start: {
    ja: '検索開始',
    en: 'Start Search',
  } satisfies LocaleText,
  stop: {
    ja: '停止',
    en: 'Stop',
  } satisfies LocaleText,
  stopping: {
    ja: '停止中...',
    en: 'Stopping...',
  } satisfies LocaleText,
};

export const eggSearchStatusLabels: LocaleMap<Record<EggBootTimingSearchStatus, string>> = {
  ja: {
    idle: 'アイドル',
    starting: '開始中',
    running: '検索中',
    stopping: '停止中',
    completed: '完了',
    error: 'エラー',
  },
  en: {
    idle: 'Idle',
    starting: 'Starting',
    running: 'Searching',
    stopping: 'Stopping',
    completed: 'Completed',
    error: 'Error',
  },
};

// === Params Card ===

export const eggSearchParamsCardTitle: LocaleText = {
  ja: '検索条件',
  en: 'Search Parameters',
};

export const eggSearchParamsSectionTitles = {
  dateTime: {
    ja: '日時範囲',
    en: 'Date/Time Range',
  } satisfies LocaleText,
  parents: {
    ja: '親個体情報',
    en: 'Parent Information',
  } satisfies LocaleText,
  conditions: {
    ja: '生成条件',
    en: 'Generation Conditions',
  } satisfies LocaleText,
  advance: {
    ja: '消費範囲',
    en: 'Advance Range',
  } satisfies LocaleText,
};

export const eggSearchParamsLabels = {
  startDate: {
    ja: '開始日',
    en: 'Start Date',
  } satisfies LocaleText,
  endDate: {
    ja: '終了日',
    en: 'End Date',
  } satisfies LocaleText,
  timeRange: {
    ja: '時間範囲',
    en: 'Time Range',
  } satisfies LocaleText,
  hour: {
    ja: '時',
    en: 'Hour',
  } satisfies LocaleText,
  minute: {
    ja: '分',
    en: 'Minute',
  } satisfies LocaleText,
  second: {
    ja: '秒',
    en: 'Second',
  } satisfies LocaleText,
  keyInput: {
    ja: 'キー入力',
    en: 'Key Input',
  } satisfies LocaleText,
  keyInputConfigure: {
    ja: '設定',
    en: 'Configure',
  } satisfies LocaleText,
  maleParentIv: {
    ja: '♂親 個体値',
    en: '♂ Parent IV',
  } satisfies LocaleText,
  femaleParentIv: {
    ja: '♀親 個体値',
    en: '♀ Parent IV',
  } satisfies LocaleText,
  ivUnknown: {
    ja: '不明',
    en: 'Unknown',
  } satisfies LocaleText,
  genderRatio: {
    ja: '性別比',
    en: 'Gender Ratio',
  } satisfies LocaleText,
  femaleAbility: {
    ja: '♀親の特性',
    en: 'Female Parent Ability',
  } satisfies LocaleText,
  everstone: {
    ja: 'かわらずのいし',
    en: 'Everstone',
  } satisfies LocaleText,
  everstoneNone: {
    ja: 'なし',
    en: 'None',
  } satisfies LocaleText,
  usesDitto: {
    ja: 'メタモン利用',
    en: 'Using Ditto',
  } satisfies LocaleText,
  masudaMethod: {
    ja: '国際孵化',
    en: 'Masuda Method',
  } satisfies LocaleText,
  nidoranFlag: {
    ja: 'ニドラン♂♀ / イルミーゼ・バルビート',
    en: 'Nidoran♂♀ / Illumise・Volbeat',
  } satisfies LocaleText,
  npcConsumption: {
    ja: 'NPC消費を考慮',
    en: 'Consider NPC Consumption',
  } satisfies LocaleText,
  userOffset: {
    ja: '開始消費',
    en: 'Start Advance',
  } satisfies LocaleText,
  advanceCount: {
    ja: '検索消費数',
    en: 'Advance Count',
  } satisfies LocaleText,
};

export const eggSearchFemaleAbilityOptions: LocaleMap<Record<0 | 1 | 2, string>> = {
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

/**
 * ステータス名（親IV入力用）
 */
export const eggSearchStatNames: LocaleMap<[string, string, string, string, string, string]> = {
  ja: ['H', 'A', 'B', 'C', 'D', 'S'],
  en: ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'],
};

/**
 * 性別比選択肢
 */
export type GenderRatioPreset = {
  threshold: number;
  genderless: boolean;
  label: LocaleText;
};

export const eggSearchGenderRatioPresets: GenderRatioPreset[] = [
  { threshold: 127, genderless: false, label: { ja: '♂1:♀1', en: '♂1:♀1' } },
  { threshold: 64, genderless: false, label: { ja: '♂3:♀1', en: '♂3:♀1' } },
  { threshold: 31, genderless: false, label: { ja: '♂7:♀1', en: '♂7:♀1' } },
  { threshold: 191, genderless: false, label: { ja: '♂1:♀3', en: '♂1:♀3' } },
  { threshold: 0, genderless: true, label: { ja: '性別不明', en: 'Genderless' } },
];

// === Filter Card ===

export const eggSearchFilterCardTitle: LocaleText = {
  ja: 'フィルター',
  en: 'Filter',
};

export const eggSearchFilterLabels = {
  disabled: {
    ja: 'フィルターを無効にする',
    en: 'Disable Filter',
  } satisfies LocaleText,
  ivRange: {
    ja: '個体値範囲',
    en: 'IV Range',
  } satisfies LocaleText,
  ivUnknown: {
    ja: '任意',
    en: 'Any',
  } satisfies LocaleText,
  nature: {
    ja: '性格',
    en: 'Nature',
  } satisfies LocaleText,
  gender: {
    ja: '性別',
    en: 'Gender',
  } satisfies LocaleText,
  ability: {
    ja: '特性',
    en: 'Ability',
  } satisfies LocaleText,
  shiny: {
    ja: '色違い',
    en: 'Shiny',
  } satisfies LocaleText,
  shinyOnly: {
    ja: '色違いのみ',
    en: 'Shiny Only',
  } satisfies LocaleText,
  shinyHint: {
    ja: '色違いの結果のみ表示',
    en: 'Show only shiny results',
  } satisfies LocaleText,
  hpType: {
    ja: 'めざパ(タイプ)',
    en: 'Hidden Power (Type)',
  } satisfies LocaleText,
  hpPower: {
    ja: 'めざパ(威力)',
    en: 'Hidden Power (Power)',
  } satisfies LocaleText,
  noSelection: {
    ja: '指定なし',
    en: 'Any',
  } satisfies LocaleText,
};

export const eggSearchGenderOptions: LocaleMap<Record<'none' | 'male' | 'female' | 'genderless', string>> = {
  ja: {
    'none': '指定なし',
    male: '♂',
    female: '♀',
    genderless: '無性別',
  },
  en: {
    'none': 'Any',
    male: '♂',
    female: '♀',
    genderless: 'Genderless',
  },
};

export const eggSearchAbilityOptions: LocaleMap<Record<'none' | '0' | '1' | '2', string>> = {
  ja: {
    'none': '指定なし',
    '0': '特性1',
    '1': '特性2',
    '2': '夢特性',
  },
  en: {
    'none': 'Any',
    '0': 'Ability 1',
    '1': 'Ability 2',
    '2': 'Hidden',
  },
};

export const eggSearchShinyOptions: LocaleMap<Record<'none' | '0' | '1' | '2', string>> = {
  ja: {
    'none': '指定なし',
    '0': '-',
    '1': '◇',
    '2': '☆',
  },
  en: {
    'none': 'Any',
    '0': '-',
    '1': '◇',
    '2': '☆',
  },
};

// === Results Card ===

export const eggSearchResultsCardTitle: LocaleText = {
  ja: '検索結果',
  en: 'Search Results',
};

export const eggSearchResultsEmpty: LocaleText = {
  ja: '結果がありません',
  en: 'No results',
};

export const eggSearchResultsCountLabel: LocaleText = {
  ja: '件',
  en: 'results',
};

export const eggSearchResultsTableHeaders = {
  bootTime: {
    ja: '起動時間',
    en: 'Boot Time',
  } satisfies LocaleText,
  timer0: {
    ja: 'Timer0',
    en: 'Timer0',
  } satisfies LocaleText,
  vcount: {
    ja: 'VCount',
    en: 'VCount',
  } satisfies LocaleText,
  lcgSeed: {
    ja: 'LCG Seed',
    en: 'LCG Seed',
  } satisfies LocaleText,
  advance: {
    ja: 'Advance',
    en: 'Advance',
  } satisfies LocaleText,
  nature: {
    ja: '性格',
    en: 'Nature',
  } satisfies LocaleText,
  ivs: {
    ja: '個体値',
    en: 'IVs',
  } satisfies LocaleText,
  shiny: {
    ja: '色違い',
    en: 'Shiny',
  } satisfies LocaleText,
  stable: {
    ja: '安定',
    en: 'Stable',
  } satisfies LocaleText,
};

// === Helper Functions ===

function getNumberFormatter(locale: SupportedLocale): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
}

export function getEggSearchStatusLabel(
  status: EggBootTimingSearchStatus,
  locale: SupportedLocale
): string {
  const labels = resolveLocaleValue(eggSearchStatusLabels, locale);
  return labels[status] ?? status;
}

export function formatEggSearchElapsed(ms: number, locale: SupportedLocale): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return locale === 'ja' ? `${seconds}秒` : `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return locale === 'ja'
    ? `${minutes}分${remainingSeconds}秒`
    : `${minutes}m ${remainingSeconds}s`;
}

export function formatEggSearchResultsCount(
  count: number,
  locale: SupportedLocale
): string {
  const formatter = getNumberFormatter(locale);
  const countStr = formatter.format(count);
  const suffix = eggSearchResultsCountLabel[locale];
  // Japanese doesn't use space before counters (e.g., '100件' not '100 件')
  const separator = locale === 'ja' ? '' : ' ';
  return `${countStr}${separator}${suffix}`;
}
