import type { LocaleMap } from './types';

/**
 * Hidden Power type names (めざめるパワー)
 * Index corresponds to the calculated HP type (0-15)
 */
export const hiddenPowerTypeNames: LocaleMap<readonly string[]> = {
  ja: [
    'かくとう', 'ひこう', 'どく', 'じめん', 'いわ', 'むし', 'ゴースト', 'はがね',
    'ほのお', 'みず', 'くさ', 'でんき', 'エスパー', 'こおり', 'ドラゴン', 'あく',
  ],
  en: [
    'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
    'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark',
  ],
};
