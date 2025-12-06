import type { LocaleText } from './types';

export const rangeKeySectionLabels = {
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
  configure: {
    ja: '設定',
    en: 'Configure',
  } satisfies LocaleText,
} as const;

export const rangeKeyDialogLabels = {
  title: {
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
} as const;
