import { DomainGameMode } from '@/types/domain';
import { getWasm, isWasmReady } from '@/lib/core/wasm-interface';

let verified = false;

function verifyAlignment(): void {
  if (verified) return;
  if (!isWasmReady()) {
    throw new Error('[GameModeMapping] WASM not initialized. Call initWasm() before using game mode mappings.');
  }
  const wasmEnum = getWasm().GameMode as unknown as Record<string, number>;
  const domainEnum = DomainGameMode as unknown as Record<string, number>;

  const expected: Array<[string, number]> = Object.keys(domainEnum)
    .filter(key => Number.isNaN(Number(key)))
    .map(key => [key, domainEnum[key]]);

  const mismatches: string[] = [];
  for (const [key, value] of expected) {
    const wasmValue = wasmEnum[key];
    if (typeof wasmValue !== 'number') {
      mismatches.push(`${key}: missing in WASM`);
      continue;
    }
    if (wasmValue !== value) {
      mismatches.push(`${key}: domain=${value} wasm=${wasmValue}`);
    }
  }

  if (mismatches.length) {
    throw new Error('[GameModeMapping] Value mismatch\n' + mismatches.join('\n'));
  }
  verified = true;
}

export function domainGameModeToWasm(mode: DomainGameMode): number {
  verifyAlignment();
  return mode as number;
}

export function ensureGameModeAlignment(): void {
  verifyAlignment();
}
