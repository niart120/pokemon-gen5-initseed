import type { DomainEncounterTypeName } from '@/types/domain';

export interface EncounterSlotJson {
  speciesId: number;
  rate: number;
  levelRange: { min: number; max: number };
}

export interface EncounterLocationsJson {
  version: 'B' | 'W' | 'B2' | 'W2';
  method: DomainEncounterTypeName;
  source: { name: string; url: string; retrievedAt: string };
  locations: Record<string, { displayNameKey: string; slots: EncounterSlotJson[] }>;
}

export interface EncounterSpeciesEntryJson {
  id: string;
  displayNameKey: string;
  speciesId: number;
  level: number;
  gender?: 'male' | 'female';
  isHiddenAbility?: boolean;
  isShinyLocked?: boolean;
}

export interface EncounterSpeciesJson {
  version: 'B' | 'W' | 'B2' | 'W2';
  method: DomainEncounterTypeName;
  source: { name: string; url: string; retrievedAt: string };
  entries: EncounterSpeciesEntryJson[];
}
