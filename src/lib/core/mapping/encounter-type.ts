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

let verified = false;

function verifyAlignment(): void {
  if (verified) return;
  if (!isWasmReady()) {
    throw new Error('[EncounterTypeMapping] WASM not initialized. Call initWasm() before using mapping functions.');
  }
  const wasmEnum = getWasm().EncounterType as unknown as Record<string, number>;

  const mismatches: string[] = [];
  // enum の文字キーのみ（数値キーは逆引き用エントリなので除外）
  for (const key of DomainEncounterTypeNames) {
    const dv = (DomainEncounterType as Record<string, number>)[key];
    const wv = wasmEnum[key];
    if (typeof wv !== 'number') {
      mismatches.push(`${key}: missing in WASM`);
      continue;
    }
    if (dv !== wv) {
      mismatches.push(`${key}: domain=${dv} wasm=${wv}`);
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
  return v as number;
}

/** WASM -> Domain */
export function wasmEncounterTypeToDomain(v: number): DomainEncounterType {
  verifyAlignment();
  // const オブジェクト化したため値集合チェックのみ
  const values = Object.values(DomainEncounterType) as number[];
  if (!values.includes(v)) {
    throw new Error(`[EncounterTypeMapping] Unknown numeric value: ${v}`);
  }
  return v as DomainEncounterType;
}

/** 起動後など任意タイミングでの明示的検証用 */
export function ensureEncounterTypeAlignment(): void {
  verifyAlignment();
}
