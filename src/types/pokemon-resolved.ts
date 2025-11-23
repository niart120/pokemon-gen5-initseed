import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import type { IndividualValues } from '@/lib/utils/individual-values';
import type { CalculatedStats } from '@/lib/utils/pokemon-stats';
import type { KeyName } from '@/lib/utils/key-input';

export interface ResolutionContext {
  encounterTable?: EncounterTable;
  genderRatios?: Map<number, GenderRatio>;
  abilityCatalog?: Map<number, string[]>;
}

export interface SerializedResolutionContext {
  encounterTable?: EncounterTable;
  genderRatios?: Array<[number, GenderRatio]>;
  abilityCatalog?: Array<[number, string[]]>;
}

export type ResolvedPokemonData = Readonly<{
  seed: bigint;
  pid: number;
  advance: number;
  natureId: number;
  shinyType: number;
  speciesId?: number;
  level?: number;
  gender?: 'M' | 'F' | 'N';
  abilityIndex?: 0 | 1 | 2;
  encounterType: number;
}>;

export interface UiResolutionOptions {
  locale?: 'ja' | 'en';
  version?: 'B' | 'W' | 'B2' | 'W2';
  baseSeed?: bigint;
}

export interface UiReadyPokemonData {
  advance: number;
  seed: bigint;
  seedHex: string;
  pid: number;
  pidHex: string;
  speciesId?: number;
  speciesName: string;
  natureId: number;
  natureName: string;
  abilityIndex?: 0 | 1 | 2;
  abilityName: string;
  genderCode?: 'M' | 'F' | 'N';
  gender: 'M' | 'F' | '-' | '?';
  level?: number;
  shinyType: number;
  shinyStatus: 'normal' | 'square' | 'star';
  encounterType: number;
  stats?: CalculatedStats;
  ivs?: IndividualValues;
  seedSourceMode?: 'lcg' | 'boot-timing';
  derivedSeedIndex?: number;
  seedSourceSeedHex?: string;
  timer0?: number;
  vcount?: number;
  bootTimestampIso?: string;
  keyInputDisplay?: string;
  keyInputNames?: KeyName[];
}
