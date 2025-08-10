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
  locations: Record<string, { displayName: string; slots: EncounterSlotJson[] }>;
}
