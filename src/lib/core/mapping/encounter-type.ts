/**
 * EncounterType mapping layer (Domain <-> WASM)
 *
 * 目的:
 * - DomainEncounterType と wasm EncounterType の値整合性を起動時に検証
 * - 双方向変換 API を提供（将来的に差異が出た場合でも境界変更を局所化）
 * - 既存の `as unknown as any` キャストを除去
 */

import { DomainEncounterType, DomainEncounterTypeNames } from '@/types/domain';
import { getWasm, isWasmReady } from '@/lib/core/wasm-interface';

const DOMAIN_KEY_ALIAS: Partial<Record<typeof DomainEncounterTypeNames[number], keyof typeof DomainEncounterType>> = {
  StaticLegendary: 'StaticSymbol',
};

const domainToWasmNumeric = new Map<number, number>();
const wasmToDomainNumeric = new Map<number, number>();

let verified = false;

function verifyAlignment(): void {
  if (verified) return;
  if (!isWasmReady()) {
    throw new Error('[EncounterTypeMapping] WASM not initialized. Call initWasm() before using mapping functions.');
  }
  const wasmEnum = getWasm().EncounterType as unknown as Record<string, number>;

  const mismatches: string[] = [];
  domainToWasmNumeric.clear();
  wasmToDomainNumeric.clear();

  for (const key of DomainEncounterTypeNames) {
    const domainValue = (DomainEncounterType as Record<string, number>)[key];
    const aliasKey = DOMAIN_KEY_ALIAS[key] ?? key;
    const wasmValue = wasmEnum[aliasKey as string];
    if (typeof wasmValue !== 'number') {
      mismatches.push(`${key}: missing in WASM`);
      continue;
    }

    if (!DOMAIN_KEY_ALIAS[key] && domainValue !== wasmValue) {
      mismatches.push(`${key}: domain=${domainValue} wasm=${wasmValue}`);
    }

    domainToWasmNumeric.set(domainValue, wasmValue);
    if (!wasmToDomainNumeric.has(wasmValue)) {
      wasmToDomainNumeric.set(wasmValue, (DomainEncounterType as Record<string, number>)[aliasKey]);
    }
  }

  if (mismatches.length) {
    throw new Error('[EncounterTypeMapping] Value mismatch\n' + mismatches.join('\n'));
  }
  verified = true;
}

/** Domain -> WASM (数値恒等写像だが整合性検証を通す) */
export function domainEncounterTypeToWasm(v: DomainEncounterType): number {
  verifyAlignment();
  const mapped = domainToWasmNumeric.get(v as number);
  if (mapped === undefined) {
    throw new Error(`[EncounterTypeMapping] Unmapped domain value: ${v}`);
  }
  return mapped;
}

/** WASM -> Domain */
export function wasmEncounterTypeToDomain(v: number): DomainEncounterType {
  verifyAlignment();
  const mapped = wasmToDomainNumeric.get(v);
  if (mapped === undefined) {
    throw new Error(`[EncounterTypeMapping] Unknown numeric value: ${v}`);
  }
  return mapped as DomainEncounterType;
}

export function isExactDomainEncounterType(v: DomainEncounterType): boolean {
  verifyAlignment();
  return domainToWasmNumeric.get(v as number) === (v as number);
}

/** 起動後など任意タイミングでの明示的検証用 */
export function ensureEncounterTypeAlignment(): void {
  verifyAlignment();
}
