/**
 * WASM Service Wrapper (module functions version)
 * - Input validation, enum conversion, and error handling for WASM operations
 * - Uses IntegratedSeedSearcher as the sole WASM interface
 */

import { initWasm, getWasm, isWasmReady } from '../core/wasm-interface';
import type { WasmModule } from '../core/wasm-interface';
import type { WasmSearchResult } from '../core/wasm-interface';
import type { ROMVersion, Hardware } from '../../types/rom';
import type { SearchConditions } from '../../types/search';
import {
  WasmGameVersion as WasmGameVersionEnum,
  WasmEncounterType as WasmEncounterTypeEnum,
  WasmGameMode as WasmGameModeEnum,
  romVersionToGameVersion as convertRomVersionToGameVersion,
  stringToEncounterType as convertStringToEncounterType,
  configToGameMode as convertConfigToGameMode,
} from './wasm-enums';

// Re-export enums for backward compatibility
export const WasmGameVersion = WasmGameVersionEnum;
export type WasmGameVersion = WasmGameVersionEnum;
export const WasmEncounterType = WasmEncounterTypeEnum;
export type WasmEncounterType = WasmEncounterTypeEnum;
export const WasmGameMode = WasmGameModeEnum;
export type WasmGameMode = WasmGameModeEnum;

// Error types
export class WasmServiceError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = 'WasmServiceError';
  }
}

export class ValidationError extends WasmServiceError {
  constructor(message: string, public readonly field?: string) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
  }
}

export class ConversionError extends WasmServiceError {
  constructor(message: string, public readonly value?: unknown) {
    super(message, 'CONVERSION_ERROR');
    this.name = 'ConversionError';
  }
}

export class WasmInitializationError extends WasmServiceError {
  constructor(message: string, public readonly cause?: Error) {
    super(message, 'WASM_INIT_ERROR');
    this.name = 'WasmInitializationError';
  }
}

// Conversion utilities (function-based)
export function romVersionToGameVersion(romVersion: ROMVersion): WasmGameVersion {
  return convertRomVersionToGameVersion(romVersion);
}

export function stringToEncounterType(encounterType: string): WasmEncounterType {
  return convertStringToEncounterType(encounterType);
}

export function configToGameMode(
  romVersion: ROMVersion,
  hasExistingSave: boolean,
  isNewGame: boolean,
  hasMemoryLink?: boolean
): WasmGameMode {
  return convertConfigToGameMode(romVersion, hasExistingSave, isNewGame, hasMemoryLink);
}

export function validateHardware(hardware: Hardware): string {
  switch (hardware) {
    case 'DS':
      return 'DS';
    case 'DS_LITE':
      return 'DS_LITE';
    case '3DS':
      return '3DS';
    default:
      throw new ConversionError(`Invalid hardware type: ${hardware}`, hardware);
  }
}

// Parameter validation utilities (function-based)
export function validateMacAddress(macAddress: number[]): Uint8Array {
  if (!Array.isArray(macAddress)) {
    throw new ValidationError('MAC address must be an array', 'macAddress');
  }
  if (macAddress.length !== 6) {
    throw new ValidationError('MAC address must be exactly 6 bytes', 'macAddress');
  }
  for (let i = 0; i < macAddress.length; i++) {
    const byte = macAddress[i];
    if (!Number.isInteger(byte) || byte < 0 || byte > 255) {
      throw new ValidationError(`MAC address byte ${i} must be 0-255, got: ${byte}`, 'macAddress');
    }
  }
  return new Uint8Array(macAddress);
}

export function validateNazo(nazo: number[]): Uint32Array {
  if (!Array.isArray(nazo)) {
    throw new ValidationError('Nazo must be an array', 'nazo');
  }
  if (nazo.length !== 5) {
    throw new ValidationError('Nazo must be exactly 5 32-bit values', 'nazo');
  }
  for (let i = 0; i < nazo.length; i++) {
    const value = nazo[i];
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new ValidationError(`Nazo value ${i} must be 0-4294967295, got: ${value}`, 'nazo');
    }
  }
  return new Uint32Array(nazo);
}

export function validateKeyInput(keyInput: number): number {
  if (!Number.isInteger(keyInput) || keyInput < 0 || keyInput > 0xfff) {
    throw new ValidationError(`Key input must be 0-4095, got: ${keyInput}`, 'keyInput');
  }
  return keyInput;
}

export function validateRange(min: number, max: number, fieldName: string): { min: number; max: number } {
  if (!Number.isInteger(min) || !Number.isInteger(max)) {
    throw new ValidationError(`${fieldName} min and max must be integers`, fieldName);
  }
  if (min < 0 || max < 0) {
    throw new ValidationError(`${fieldName} values must be non-negative`, fieldName);
  }
  if (min > max) {
    throw new ValidationError(`${fieldName} min must be less than or equal to max`, fieldName);
  }
  const maxValue = fieldName.toLowerCase().includes('timer0') ? 0xffff : 0xff;
  if (min > maxValue || max > maxValue) {
    throw new ValidationError(`${fieldName} values must not exceed ${maxValue}`, fieldName);
  }
  return { min, max };
}

export function validateDateTime(
  year: number,
  month: number,
  date: number,
  hour: number,
  minute: number,
  second: number
): void {
  if (!Number.isInteger(year) || year < 2000 || year > 2099) {
    throw new ValidationError(`Year must be 2000-2099, got: ${year}`, 'year');
  }
  if (!Number.isInteger(month) || month < 1 || month > 12) {
    throw new ValidationError(`Month must be 1-12, got: ${month}`, 'month');
  }
  if (!Number.isInteger(date) || date < 1 || date > 31) {
    throw new ValidationError(`Date must be 1-31, got: ${date}`, 'date');
  }
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
    throw new ValidationError(`Hour must be 0-23, got: ${hour}`, 'hour');
  }
  if (!Number.isInteger(minute) || minute < 0 || minute > 59) {
    throw new ValidationError(`Minute must be 0-59, got: ${minute}`, 'minute');
  }
  if (!Number.isInteger(second) || second < 0 || second > 59) {
    throw new ValidationError(`Second must be 0-59, got: ${second}`, 'second');
  }

  try {
    const testDate = new Date(year, month - 1, date, hour, minute, second);
    if (
      testDate.getFullYear() !== year ||
      testDate.getMonth() !== month - 1 ||
      testDate.getDate() !== date ||
      testDate.getHours() !== hour ||
      testDate.getMinutes() !== minute ||
      testDate.getSeconds() !== second
    ) {
      throw new ValidationError('Invalid date/time combination', 'dateTime');
    }
  } catch {
    throw new ValidationError('Invalid date/time combination', 'dateTime');
  }
}

export function validateTargetSeeds(targetSeeds: number[]): Uint32Array {
  if (!Array.isArray(targetSeeds)) {
    throw new ValidationError('Target seeds must be an array', 'targetSeeds');
  }
  if (targetSeeds.length === 0) {
    throw new ValidationError('Target seeds array cannot be empty', 'targetSeeds');
  }
  if (targetSeeds.length > 10000) {
    throw new ValidationError('Target seeds array cannot exceed 10000 elements', 'targetSeeds');
  }
  for (let i = 0; i < targetSeeds.length; i++) {
    const seed = targetSeeds[i];
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new ValidationError(`Target seed ${i} must be 0-4294967295, got: ${seed}`, 'targetSeeds');
    }
  }
  return new Uint32Array(targetSeeds);
}

// Module-level initialization state
let initialized = false;

// WASM統合検索のインスタンス型
type SeedSearcher = InstanceType<WasmModule['IntegratedSeedSearcher']>;

export async function initializeWasmService(): Promise<void> {
  if (initialized) return;
  try {
    await initWasm();
    initialized = true;
  } catch (error) {
    throw new WasmInitializationError(
      `Failed to initialize WebAssembly module: ${error}`,
      error instanceof Error ? error : undefined
    );
  }
}

export function isWasmServiceReady(): boolean {
  return initialized && isWasmReady();
}

// Test helper: reset state (used only in tests)
export function resetWasmServiceStateForTests(): void {
  initialized = false;
}

export function createSearcher(
  macAddress: number[],
  nazo: number[],
  hardware: Hardware,
  keyInput: number,
  frame: number = 0
): SeedSearcher {
  if (!isWasmServiceReady()) {
    throw new WasmInitializationError('WASM module not initialized');
  }

  try {
  const validatedMac = validateMacAddress(macAddress);
  const validatedNazo = validateNazo(nazo);
  const validatedKeyInput = validateKeyInput(keyInput);
  validateHardware(hardware);

    if (!Number.isInteger(frame) || frame < 0 || frame > 0xffffffff) {
      throw new ValidationError(`Frame must be 0-4294967295, got: ${frame}`, 'frame');
    }

    const wasm = getWasm();
    return new wasm.IntegratedSeedSearcher(
      validatedMac,
      validatedNazo,
      hardware,
      validatedKeyInput,
      frame
    );
  } catch (error) {
    if (error instanceof WasmServiceError) throw error;
    throw new WasmServiceError(`Failed to create searcher: ${error}`, 'SEARCHER_CREATION_ERROR');
  }
}

export function searchSeeds(
  searcher: SeedSearcher,
  startDateTime: {
    year: number;
    month: number;
    date: number;
    hour: number;
    minute: number;
    second: number;
  },
  rangeSeconds: number,
  timer0Range: { min: number; max: number },
  vcountRange: { min: number; max: number },
  targetSeeds: number[]
): WasmSearchResult[] {
  if (!isWasmServiceReady()) {
    throw new WasmInitializationError('WASM module not initialized');
  }
  if (!searcher) {
    throw new ValidationError('Searcher instance is required', 'searcher');
  }

  try {
  validateDateTime(
      startDateTime.year,
      startDateTime.month,
      startDateTime.date,
      startDateTime.hour,
      startDateTime.minute,
      startDateTime.second
    );
    if (!Number.isInteger(rangeSeconds) || rangeSeconds < 1 || rangeSeconds > 86400) {
      throw new ValidationError('Range seconds must be 1-86400', 'rangeSeconds');
    }
  const validatedTimer0Range = validateRange(
      timer0Range.min,
      timer0Range.max,
      'timer0Range'
    );
  const validatedVcountRange = validateRange(
      vcountRange.min,
      vcountRange.max,
      'vcountRange'
    );
  const validatedTargetSeeds = validateTargetSeeds(targetSeeds);

  return searcher.search_seeds_integrated_simd(
      startDateTime.year,
      startDateTime.month,
      startDateTime.date,
      startDateTime.hour,
      startDateTime.minute,
      startDateTime.second,
      rangeSeconds,
      validatedTimer0Range.min,
      validatedTimer0Range.max,
      validatedVcountRange.min,
      validatedVcountRange.max,
      validatedTargetSeeds
    );
  } catch (error) {
    if (error instanceof WasmServiceError) throw error;
    throw new WasmServiceError(`Search failed: ${error}`, 'SEARCH_ERROR');
  }
}

export async function searchWithConditions(
  conditions: SearchConditions,
  targetSeeds: number[],
  nazo?: number[]
): Promise<WasmSearchResult[]> {
  await initializeWasmService();

  try {
  validateKeyInput(conditions.keyInput);
    const nazovalues = nazo || [0, 0, 0, 0, 0];

    const searcher = createSearcher(
      conditions.macAddress,
      nazovalues,
      conditions.hardware,
      conditions.keyInput
    );

    try {
  const results = searchSeeds(
        searcher,
        {
          year: conditions.dateRange.startYear,
          month: conditions.dateRange.startMonth,
          date: conditions.dateRange.startDay,
          hour: conditions.dateRange.startHour,
          minute: conditions.dateRange.startMinute,
          second: conditions.dateRange.startSecond,
        },
        Math.max(
          1,
          Math.floor(
            (new Date(
              conditions.dateRange.endYear,
              conditions.dateRange.endMonth - 1,
              conditions.dateRange.endDay,
              conditions.dateRange.endHour,
              conditions.dateRange.endMinute,
              conditions.dateRange.endSecond
            ).getTime() -
              new Date(
                conditions.dateRange.startYear,
                conditions.dateRange.startMonth - 1,
                conditions.dateRange.startDay,
                conditions.dateRange.startHour,
                conditions.dateRange.startMinute,
                conditions.dateRange.startSecond
              ).getTime()) /
              1000
          )
        ),
        conditions.timer0VCountConfig.timer0Range,
        conditions.timer0VCountConfig.vcountRange,
        targetSeeds
      );
      return results;
    } finally {
      if (searcher && typeof searcher.free === 'function') {
        searcher.free();
      }
    }
  } catch (error) {
    if (error instanceof WasmServiceError) throw error;
    throw new WasmServiceError(`Search with conditions failed: ${error}`, 'CONDITIONS_SEARCH_ERROR');
  }
}
