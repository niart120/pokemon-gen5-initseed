import type { ROMVersion } from '@/types/rom';
import type { DomainEncounterType } from '@/types/domain';
import { DomainEncounterType as EncounterTypeEnum } from '@/types/domain';
import { listRegistryLocations, getEncounterFromRegistry } from './loader';

export interface EncounterLocationOption {
  key: string; // normalized key
  displayName: string; // original display name
}

export interface EncounterSpeciesOption {
  speciesId: number;
  firstSlotIndex: number;
  appearances: number;
  totalRate: number;
  minLevel: number;
  maxLevel: number;
}

// Location based encounter types
const LOCATION_BASED: Set<number> = new Set([
  EncounterTypeEnum.Normal,
  EncounterTypeEnum.Surfing,
  EncounterTypeEnum.Fishing,
  EncounterTypeEnum.ShakingGrass,
  EncounterTypeEnum.DustCloud,
  EncounterTypeEnum.PokemonShadow,
  EncounterTypeEnum.SurfingBubble,
  EncounterTypeEnum.FishingBubble,
]);

export function isLocationBasedEncounter(method: DomainEncounterType): boolean {
  return LOCATION_BASED.has(method as number);
}

const cacheLocations = new Map<string, EncounterLocationOption[]>();
const cacheSpecies = new Map<string, EncounterSpeciesOption[]>();

function locKey(version: ROMVersion, method: DomainEncounterType) {
  return `L|${version}|${method}`;
}
function speciesKey(version: ROMVersion, method: DomainEncounterType, locationKey?: string) {
  return `S|${version}|${method}|${locationKey ?? '-'}`;
}

export function listEncounterLocations(version: ROMVersion, method: DomainEncounterType): EncounterLocationOption[] {
  if (!isLocationBasedEncounter(method)) return [];
  const key = locKey(version, method);
  const hit = cacheLocations.get(key);
  if (hit) return hit;
  const rows = listRegistryLocations(version, method).map(r => ({ key: r.key, displayName: r.displayName }));
  cacheLocations.set(key, rows);
  return rows;
}

export function listEncounterSpeciesOptions(version: ROMVersion, method: DomainEncounterType, locationKey?: string): EncounterSpeciesOption[] {
  const k = speciesKey(version, method, locationKey);
  const hit = cacheSpecies.get(k);
  if (hit) return hit;
  let rows: EncounterSpeciesOption[] = [];
  if (isLocationBasedEncounter(method)) {
    if (!locationKey) return [];
    const table = getEncounterFromRegistry(version, locationKey, method as any);
    if (table) {
      const map = new Map<number, EncounterSpeciesOption>();
      table.slots.forEach((slot, idx) => {
        const ex = map.get(slot.speciesId);
        if (!ex) {
          map.set(slot.speciesId, {
            speciesId: slot.speciesId,
            firstSlotIndex: idx,
            appearances: 1,
            totalRate: slot.rate,
            minLevel: slot.levelRange.min,
            maxLevel: slot.levelRange.max,
          });
        } else {
          ex.appearances += 1;
            ex.totalRate += slot.rate;
            if (slot.levelRange.min < ex.minLevel) ex.minLevel = slot.levelRange.min;
            if (slot.levelRange.max > ex.maxLevel) ex.maxLevel = slot.levelRange.max;
        }
      });
      rows = Array.from(map.values()).sort((a,b)=> b.totalRate - a.totalRate || a.speciesId - b.speciesId);
    }
  } else {
    // Static encounter placeholder (WIP)
    // TODO(static encounters): load static species catalog
    rows = [];
  }
  cacheSpecies.set(k, rows);
  return rows;
}
