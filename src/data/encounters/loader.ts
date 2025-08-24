import type { EncounterLocationsJson, EncounterSlotJson } from './schema';
import type { DomainEncounterType as EncounterType } from '@/types/domain';
import type { ROMVersion } from '@/types/rom';
import { DomainEncounterType, getDomainEncounterTypeName } from '@/types/domain';

function normalizeLocationKey(location: string): string {
  return location.trim().replace(/[\u3000\s]+/g, '').replace(/[‐‑‒–—−\-_.]/g, '');
}

// 英語名→日本語キーの簡易エイリアス
function applyLocationAlias(input: string): string {
  const s = input.trim();
  // Route N → N番道路
  const m = s.match(/^route\s*(\d+)$/i);
  if (m) return `${parseInt(m[1], 10)}番道路`;
  return s;
}

// Local resolver: numeric value -> canonical encounter type name
const methodName = (method: EncounterType): keyof typeof DomainEncounterType => {
  const name = getDomainEncounterTypeName(method);
  if (!name) throw new Error(`Unknown encounter method value: ${method}`);
  return name as keyof typeof DomainEncounterType;
};

export type EncounterRegistry = Record<string, { displayName: string; slots: EncounterSlotJson[] }>

let registry: Record<string, EncounterRegistry> | null = null; // key: `${version}_${method}`

// 同期初期化（ビルド時取り込み済みJSONのみ）
(function initRegistry() {
  const modules = import.meta.glob('./generated/v1/**/**/*.json', { eager: true }) as Record<string, { default: EncounterLocationsJson } | EncounterLocationsJson>;
  const acc: Record<string, EncounterRegistry> = {};
  for (const [, mod] of Object.entries(modules)) {
    const data: EncounterLocationsJson = (('default' in (mod as object)) ? (mod as { default: EncounterLocationsJson }).default : (mod as EncounterLocationsJson));
    const key = `${data.version}_${data.method}`;
    if (!acc[key]) acc[key] = {};
    for (const [locKey, payload] of Object.entries(data.locations)) {
      acc[key][normalizeLocationKey(locKey)] = payload;
    }
  }
  registry = acc;
})();

export function ensureEncounterRegistryLoaded(): void {
  if (!registry) throw new Error('Encounter registry not initialized.');
}

export function getEncounterFromRegistry(version: ROMVersion, location: string, method: EncounterType) {
  ensureEncounterRegistryLoaded();
  const key = `${version}_${methodName(method)}`;
  // 入力ロケーションに英語→日本語の簡易エイリアスを適用してから正規化
  const loc = normalizeLocationKey(applyLocationAlias(location));
  const hit = registry![key]?.[loc];
  return hit ?? null;
}

/**
 * List normalized location entries for a given version & method.
 * Order: JSON 定義読み込み順 (registry 格納順) を維持。
 */
export function listRegistryLocations(version: ROMVersion, method: EncounterType): { key: string; displayName: string }[] {
  ensureEncounterRegistryLoaded();
  const key = `${version}_${methodName(method)}`;
  const bucket = registry![key];
  if (!bucket) return [];
  const out: { key: string; displayName: string }[] = [];
  for (const [locKey, payload] of Object.entries(bucket)) {
    out.push({ key: locKey, displayName: payload.displayName });
  }
  return out;
}
