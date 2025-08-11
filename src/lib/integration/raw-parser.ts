/**
 * WASM入力アダプタ（唯一の境界）
 * wasm-bindgen の RawPokemonData インスタンスから snake_case の RawPokemonData を生成する。
 * 互換性拡張は行わず、get_* のアクセサ（readonlyプロパティ）または0引数関数のみを許可する。
 */

import { RawPokemonData } from '@/types/pokemon-raw';

/**
 * WASMライク入力アダプタ: wasm-bindgenインスタンス or 同等のgetterを持つオブジェクトから
 * RawPokemonData（snake_case）を生成する低レベルアダプタ。
 *
 * 役割: インテグレーション境界で入力のゆらぎを吸収し、以降は型安全な RawPokemonData に揃える。
 * 想定利用先: 上位のUI層アダプタ（例: parseRawPokemonData）など。
 */
export function parseFromWasmRaw(wasmData: unknown): RawPokemonData {
  if (!wasmData) {
    throw new Error('WASM data is null or undefined');
  }

  const obj = wasmData as Record<string, unknown>;

  // wasm-bindgen の get_* は JS 側では readonly アクセサとして公開されるため、
  // 関数であれば呼び出し、値であればそのまま読む。
  const readField = (key: string) => {
    const v = obj[key];
    if (typeof v === 'undefined') {
      throw new Error(`Missing required property: ${key}`);
    }
    if (typeof v === 'function') {
      try {
        const getter = v as (this: unknown) => unknown;
        return getter.call(obj);
      } catch (e) {
        throw new Error(`Failed to call getter ${key}: ${e}`);
      }
    }
    return v;
  };

  const toBigInt = (v: unknown): bigint => {
    if (typeof v === 'bigint') return v;
    if (typeof v === 'number') return BigInt(Math.trunc(v));
    if (typeof v === 'string') return BigInt(v);
    if (typeof v === 'boolean') return BigInt(v ? 1 : 0);
    throw new Error(`Invalid bigint-like value: ${String(v)}`);
  };

  const toNumber = (v: unknown): number => {
    if (typeof v === 'number') return v;
    if (typeof v === 'bigint') return Number(v);
    if (typeof v === 'string') {
      const n = Number(v);
      if (!Number.isFinite(n)) throw new Error(`Invalid number string: ${v}`);
      return n;
    }
    if (typeof v === 'boolean') return v ? 1 : 0;
    throw new Error(`Invalid number-like value: ${String(v)}`);
  };

  try {
  const seedVal = readField('get_seed');
  const pid = readField('get_pid');
  const nature = readField('get_nature');
  const syncApplied = readField('get_sync_applied');
  const abilitySlot = readField('get_ability_slot');
  const genderValue = readField('get_gender_value');
  const encounterSlotValue = readField('get_encounter_slot_value');
  const encounterType = readField('get_encounter_type');
  const levelRandValue = readField('get_level_rand_value');
  const shinyType = readField('get_shiny_type');

    return {
      seed: toBigInt(seedVal),
      pid: toNumber(pid),
      nature: toNumber(nature),
      sync_applied: Boolean(syncApplied),
      ability_slot: toNumber(abilitySlot),
      gender_value: toNumber(genderValue),
      encounter_slot_value: toNumber(encounterSlotValue),
      encounter_type: toNumber(encounterType),
      level_rand_value: toNumber(levelRandValue),
      shiny_type: toNumber(shinyType),
    };
  } catch (error) {
    throw new Error(`Failed to adapt WASM RawPokemonData: ${error}`);
  }
}
