/**
 * WASM boundary enum re-exports and conversions.
 *
 * - Keep the single source of truth for application enums in src/types/domain.ts
 * - This module exposes the WASM-flavored enums and conversion helpers.
 */

import { DomainGameVersion, DomainEncounterType, DomainGameMode } from '../../types/domain';

// These mirrors the numeric values of WASM enums and provide TS names used at the boundary
export const WasmGameVersion = DomainGameVersion;
export type WasmGameVersion = DomainGameVersion;

export const WasmEncounterType = DomainEncounterType;
export type WasmEncounterType = DomainEncounterType;

export const WasmGameMode = DomainGameMode;
export type WasmGameMode = DomainGameMode;

// Converters between domain-side strings or inputs and WASM numeric enums
import type { ROMVersion } from '../../types/rom';
import { ConversionError } from './wasm-service';

export function romVersionToGameVersion(romVersion: ROMVersion): WasmGameVersion {
  switch (romVersion) {
    case 'B':
    case 'W':
      return WasmGameVersion.BlackWhite;
    case 'B2':
    case 'W2':
      return WasmGameVersion.BlackWhite2;
    default:
      throw new ConversionError(`Invalid ROM version: ${romVersion}`, romVersion);
  }
}

export function stringToEncounterType(encounterType: string): WasmEncounterType {
  const normalized = encounterType.toUpperCase().replace(/[_-]/g, '');
  switch (normalized) {
    case 'NORMAL':
      return WasmEncounterType.Normal;
    case 'SURFING':
      return WasmEncounterType.Surfing;
    case 'FISHING':
      return WasmEncounterType.Fishing;
    case 'SHAKINGGRASS':
      return WasmEncounterType.ShakingGrass;
    case 'DUSTCLOUD':
      return WasmEncounterType.DustCloud;
    case 'POKEMONSHADOW':
      return WasmEncounterType.PokemonShadow;
    case 'SURFINGBUBBLE':
      return WasmEncounterType.SurfingBubble;
    case 'FISHINGBUBBLE':
      return WasmEncounterType.FishingBubble;
    case 'STATICSYMBOL':
      return WasmEncounterType.StaticSymbol;
    case 'STATICSTARTER':
      return WasmEncounterType.StaticStarter;
    case 'STATICFOSSIL':
      return WasmEncounterType.StaticFossil;
    case 'STATICEVENT':
      return WasmEncounterType.StaticEvent;
    case 'ROAMING':
      return WasmEncounterType.Roaming;
    default:
      throw new ConversionError(`Invalid encounter type: ${encounterType}`, encounterType);
  }
}

export function configToGameMode(
  romVersion: ROMVersion,
  hasExistingSave: boolean,
  isNewGame: boolean,
  hasMemoryLink?: boolean
): WasmGameMode {
  const isBW2 = romVersion === 'B2' || romVersion === 'W2';
  if (isBW2) {
    if (isNewGame) {
      if (hasExistingSave) {
        return hasMemoryLink
          ? WasmGameMode.Bw2NewGameWithMemoryLinkSave
          : WasmGameMode.Bw2NewGameNoMemoryLinkSave;
      } else {
        return WasmGameMode.Bw2NewGameNoSave;
      }
    } else {
      return hasMemoryLink
        ? WasmGameMode.Bw2ContinueWithMemoryLink
        : WasmGameMode.Bw2ContinueNoMemoryLink;
    }
  } else {
    if (isNewGame) {
      return hasExistingSave ? WasmGameMode.BwNewGameWithSave : WasmGameMode.BwNewGameNoSave;
    } else {
      return WasmGameMode.BwContinue;
    }
  }
}
