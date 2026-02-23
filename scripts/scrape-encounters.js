#!/usr/bin/env node
/*
  Encounter tables scraper for BW/BW2
  - Primary source: Pokebook (静的HTML表)
  - Output: src/data/encounters/generated/v1/<ROMVersion>/<Method>.json
  - Schema: EncounterLocationsJson (see src/data/encounters/schema.ts)

  Notes
  - This script is for data acquisition in development. Do NOT fetch at runtime.
  - Always commit generated JSON for reproducible builds.
*/

import fs from 'node:fs/promises';
import path from 'node:path';
import fetch from 'node-fetch';
import { load as loadHtml } from 'cheerio';

// 未知のJP種名を収集するセット（実行中のみ）
let MISSING_SPECIES = new Set();

const I18N_DIR = path.resolve('src/data/encounters/i18n');
const DISPLAY_NAME_DICTIONARY_PATH = path.join(I18N_DIR, 'display-names.json');

let displayNameDictionary = { locations: {}, static: {}, categories: {}, types: {} };

// ターゲットとするメソッド（濃い草むらはスコープ外）
const METHODS = [
  'Normal', // 草むら, 洞窟
  'ShakingGrass', // 揺れる草むら
  'DustCloud', // 土煙（洞窟系）
  'Surfing', // なみのり（通常）
  'SurfingBubble', // なみのり（泡）
  'Fishing', // つり（通常）
  'FishingBubble', // つり（泡）
];

const VERSIONS = ['B', 'W', 'B2', 'W2'];

function normalizeLocationKey(location) {
  return location.trim().replace(/[\u3000\s]+/g, '').replace(/[‐‑‒–—−\-_.]/g, '');
}

async function loadDisplayNameDictionary() {
  try {
    const text = await fs.readFile(DISPLAY_NAME_DICTIONARY_PATH, 'utf8');
    const json = JSON.parse(text);
    return {
      locations: json.locations ?? {},
      static: json.static ?? {},
      categories: json.categories ?? {},
      types: json.types ?? {},
    };
  } catch {
    return { locations: {}, static: {}, categories: {}, types: {} };
  }
}

function upsertDictionaryEntry(bucket, key, jaValue, enValue) {
  if (!key) return;
  if (!bucket[key]) {
    bucket[key] = {};
  }
  const entry = bucket[key];
  if (jaValue && !entry.ja) entry.ja = jaValue;
  if (enValue && !entry.en) entry.en = enValue;
}

function upsertLocationDisplayName(key, displayName) {
  const ja = displayName || key;
  upsertDictionaryEntry(displayNameDictionary.locations, key, ja, ja);
}

function sortRecord(record) {
  return Object.fromEntries(Object.keys(record).sort().map(k => [k, record[k]]));
}

async function persistDisplayNameDictionary() {
  const sorted = {
    locations: sortRecord(displayNameDictionary.locations),
    static: sortRecord(displayNameDictionary.static),
    categories: sortRecord(displayNameDictionary.categories),
    types: sortRecord(displayNameDictionary.types),
  };
  await ensureDir(I18N_DIR);
  await fs.writeFile(DISPLAY_NAME_DICTIONARY_PATH, JSON.stringify(sorted, null, 2), 'utf8');
}

function todayISO() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function toInt(n, def = 0) {
  const x = parseInt(String(n).trim(), 10);
  return Number.isFinite(x) ? x : def;
}

function parseLevelRangeFromText(text) {
  const m = String(text).match(/(\d+)\D+(\d+)/);
  if (m) {
    const min = toInt(m[1]);
    const max = toInt(m[2]);
    if (min && max) return { min, max };
  }
  const single = String(text).match(/(\d+)/);
  if (single) {
    const v = toInt(single[1]);
    return { min: v, max: v };
  }
  return { min: 1, max: 1 };
}

// name alias → canonical English name（未使用）
const NAME_ALIASES = new Map([
  // ['ヨーテリー', 'Lillipup'],
]);

// 実URL: Pokebook 各バージョンのエンカウントテーブル（各メソッドは同一ページ内セクションに存在）
const SOURCE_MAP = {
  B: {
    Normal: 'https://pokebook.jp/data/sp5/enc_b',
    ShakingGrass: 'https://pokebook.jp/data/sp5/enc_b',
    DustCloud: 'https://pokebook.jp/data/sp5/enc_b',
    Surfing: 'https://pokebook.jp/data/sp5/enc_b',
    SurfingBubble: 'https://pokebook.jp/data/sp5/enc_b',
    Fishing: 'https://pokebook.jp/data/sp5/enc_b',
    FishingBubble: 'https://pokebook.jp/data/sp5/enc_b',
  },
  W: {
    Normal: 'https://pokebook.jp/data/sp5/enc_w',
    ShakingGrass: 'https://pokebook.jp/data/sp5/enc_w',
    DustCloud: 'https://pokebook.jp/data/sp5/enc_w',
    Surfing: 'https://pokebook.jp/data/sp5/enc_w',
    SurfingBubble: 'https://pokebook.jp/data/sp5/enc_w',
    Fishing: 'https://pokebook.jp/data/sp5/enc_w',
    FishingBubble: 'https://pokebook.jp/data/sp5/enc_w',
  },
  B2: {
    Normal: 'https://pokebook.jp/data/sp5/enc_b2',
    ShakingGrass: 'https://pokebook.jp/data/sp5/enc_b2',
    DustCloud: 'https://pokebook.jp/data/sp5/enc_b2',
    Surfing: 'https://pokebook.jp/data/sp5/enc_b2',
    SurfingBubble: 'https://pokebook.jp/data/sp5/enc_b2',
    Fishing: 'https://pokebook.jp/data/sp5/enc_b2',
    FishingBubble: 'https://pokebook.jp/data/sp5/enc_b2',
  },
  W2: {
    Normal: 'https://pokebook.jp/data/sp5/enc_w2',
    ShakingGrass: 'https://pokebook.jp/data/sp5/enc_w2',
    DustCloud: 'https://pokebook.jp/data/sp5/enc_w2',
    Surfing: 'https://pokebook.jp/data/sp5/enc_w2',
    SurfingBubble: 'https://pokebook.jp/data/sp5/enc_w2',
    Fishing: 'https://pokebook.jp/data/sp5/enc_w2',
    FishingBubble: 'https://pokebook.jp/data/sp5/enc_w2',
  },
};

// スロット率プリセット
const SLOT_RATE_PRESETS = {
  Normal: [20, 20, 10, 10, 10, 10, 5, 5, 4, 4, 1, 1],
  ShakingGrass: [20, 20, 10, 10, 10, 10, 5, 5, 4, 4, 1, 1],
  DustCloud: [20, 20, 10, 10, 10, 10, 5, 5, 4, 4, 1, 1],
  Surfing: [60, 30, 5, 4, 1],
  SurfingBubble: [60, 30, 5, 4, 1],
  Fishing: [60, 30, 5, 4, 1],
  FishingBubble: [60, 30, 5, 4, 1],
};

const WATER_SINGLE_ROW_LOCATIONS = new Map([
  ['チャンピオンロード', {}],
  ['ジャイアントホール', {}],
  ['地下水脈の穴', {}],
  ['ヒウン下水道', {}],
  ['サンギ牧場', {}],
  ['ヤーコンロード', {}],
  ['4番道路', {}],
]);

const DUPLICATE_SUFFIX_RULES_BW = Object.freeze({
  'ヤグルマの森': ['外部', '内部'],
  'リゾートデザート': ['外部', '内部'],
  '古代の城': ['1F,B1F', 'B2F-B6F', '最下層', '小部屋'],
  '電気石の洞穴': ['1F', 'B1F', 'B2F'],
  'ネジ山(春)': null,
  'ネジ山(夏)': null,
  'ネジ山(秋)': null,
  'ネジ山(冬)': null,
  'リュウラセンの塔(春)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔(夏)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔(秋)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔(冬)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔': ['1F', '2F'],
  'チャンピオンロード': ['外部', '内部1F(中央・右)', '内部1F(右),2F,3F', '内部4F(中央)', '内部4F(左・右),5F-7F'],
  'ジャイアントホール': ['外部', 'B1F', '地底森林', '最奥部'],
  '地下水脈の穴': null,
  'フキヨセの洞穴': null,
  'タワーオブヘブン': ['2F', '3F', '4F', '5F'],
  '修行の岩屋': ['1F', 'B1F', 'B2F'],
  '10番道路': ['外部', '内部'],
});

const DUPLICATE_SUFFIX_RULES_B2W2 = Object.freeze({
  'ヤグルマの森': ['外部', '内部'],
  'リゾートデザート': ['外部', '内部'],
  '古代の城': ['1F,B1F', '最下層', '小部屋'],
  '電気石の洞穴': ['1F', 'B1F', 'B2F'],
  'ネジ山(春)': null,
  'ネジ山(夏)': null,
  'ネジ山(秋)': null,
  'ネジ山(冬)': null,
  'リュウラセンの塔(春)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔(夏)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔(秋)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔(冬)': ['外部(南)', '外部(北東)'],
  'リュウラセンの塔': ['1F', '2F'],
  'チャンピオンロード': ['内部1F', '内部2F(前部)', '内部2F(後部)', '内部3F', '内部4F', '樹林', '外壁(下層)', '外壁(上層)'],
  'ジャイアントホール': ['外部', 'B1F', '地底森林', '最奥部'],
  'ヒウン下水道': null,
  'タチワキコンビナート': ['北部', '南部'],
  'リバースマウンテン': ['外部', '内部', '小部屋'],
  'ストレンジャーハウス': ['入口,B1F', '小部屋'],
  '古代の抜け道': ['南部', '北部', '中央部'],
  'ヤーコンロード': null,
  '地底遺跡': null,
  '海辺の洞穴': ['1F', 'B1F'],
  '4番道路': null,
  '地下水脈の穴': null,
  'フキヨセの洞穴': null,
  'タワーオブヘブン': ['2F', '3F', '4F', '5F'],
});

const DUPLICATE_SUFFIX_RULES = Object.freeze({
  B: DUPLICATE_SUFFIX_RULES_BW,
  W: DUPLICATE_SUFFIX_RULES_BW,
  B2: DUPLICATE_SUFFIX_RULES_B2W2,
  W2: DUPLICATE_SUFFIX_RULES_B2W2,
});

async function fetchHtml(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'encounter-scraper/1.0' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return await res.text();
}

async function loadSpeciesNameToId() {
  const file = path.resolve('src/data/pokemon-species.ts');
  const txt = await fs.readFile(file, 'utf8');

  // Prefer fast path: inline comment after ID, e.g., "509: { // Purrloin"
  const quick = new Map();
  for (const line of txt.split(/\r?\n/)) {
    const m = line.match(/^\s*(\d+):\s*\{\s*\/\/\s*([^\r\n]+)/);
    if (m) quick.set(m[2].trim(), parseInt(m[1], 10));
  }

  if (quick.size > 0) return quick;

  // Fallback: scan blocks to capture nationalDex and name fields
  const map = new Map();
  const idRe = /nationalDex:\s*(\d+)/;
  const nameRe = /name:\s*'([^']+)'/;
  let currentId = null;
  for (const line of txt.split(/\r?\n/)) {
    const idM = line.match(idRe);
    if (idM) currentId = parseInt(idM[1], 10);
    const nameM = line.match(nameRe);
    if (currentId && nameM) {
      map.set(nameM[1], currentId);
      currentId = null;
    }
  }
  return map;
}

async function loadSpeciesAliasJa() {
  const aliasPath = path.resolve('src/data/encounters/aliases/species-ja.json');
  try {
    const txt = await fs.readFile(aliasPath, 'utf8');
    const obj = JSON.parse(txt);
    const m = new Map();
    for (const [k, v] of Object.entries(obj)) m.set(k, v);
    return m;
  } catch (_) {
    return new Map();
  }
}

function canonicalizeSpeciesName(raw) {
  const t = String(raw).trim();
  // Remove bracketed notes and spaces
  return t.replace(/\s*\([^)]*\)\s*$/, '').replace(/[\u3000\s]+/g, '');
}

function findSectionTables($, method) {
  // 見出しテキストでセクション境界を抽出
  const headers = $('h1,h2,h3,h4,h5').toArray();
  const textOf = (el) => ($(el).text() || '').replace(/[\s\u3000]+/g, '');
  const matchers = {
    Normal: (t) => t.includes('草むら') && t.includes('洞窟') && !t.includes('濃い草むら'),
    ShakingGrass: (t) => t.includes('揺れる草むら') || t.includes('土煙'),
    DustCloud: (t) => t.includes('揺れる草むら') || t.includes('土煙'),
    Surfing: (t) => t.includes('なみのり') || t.includes('つり'),
    SurfingBubble: (t) => t.includes('なみのり') || t.includes('つり'),
    Fishing: (t) => t.includes('なみのり') || t.includes('つり'),
    FishingBubble: (t) => t.includes('なみのり') || t.includes('つり'),
  };
  const isHeader = (el) => /^h[1-6]$/i.test(el.tagName || el.name || '');

  let startIdx = -1;
  for (let i = 0; i < headers.length; i++) {
    const t = textOf(headers[i]);
    if (matchers[method]?.(t)) {
      // 濃い草むらは除外（完全一致を避ける）
      if (method === 'Normal' && t.includes('濃い草むら')) continue;
      startIdx = i; break;
    }
  }
  if (startIdx === -1) return [];

  const tables = [];
  for (let i = startIdx + 1; i < headers.length; i++) {
    const el = headers[i];
    // 次の同レベル以上の見出しが来たら終了
    if (isHeader(el)) break;
  }
  // start headerから次のheader直前までの兄弟要素を走査
  let node = $(headers[startIdx]).next();
  while (node && node.length) {
    const name = node[0].name || node[0].tagName || '';
    if (/^h[1-6]$/i.test(name)) break;
    if (name === 'table') tables.push(node);
    node = node.next();
  }
  return tables;
}

function parseWideRowIntoSlots($, tr, method, aliasJa) {
  const tds = $(tr).find('td');
  if (tds.length < 13) return null; // 1列:ロケーション + 12枠
  const locText = $(tds[0]).text().trim();
  const locMatch = locText.match(/^\[([^\]]+)\]\s*(.*)$/);
  const baseName = (locMatch ? locMatch[2] : locText.replace(/^\[[^\]]+\]\s*/, '')).trim();

  const rates = SLOT_RATE_PRESETS[method] || SLOT_RATE_PRESETS.Normal;
  const slots = [];
  for (let i = 1; i <= 12 && i < tds.length; i++) {
    const cell = $(tds[i]).text().trim();
    if (!cell) continue;
    // 形式: 名前(レベル) または 名前(最小～最大)
    const nameMatch = cell.match(/^([^()]+)\(([^)]+)\)$/);
    if (!nameMatch) continue;
    const rawName = canonicalizeSpeciesName(nameMatch[1]);
    const lvlText = nameMatch[2];
    const levelRange = parseLevelRangeFromText(lvlText);

    const speciesId = aliasJa.get(rawName);
    if (!speciesId) {
      MISSING_SPECIES.add(rawName);
      continue;
    }

    const rate = rates[i - 1] ?? 1;
    slots.push({ speciesId, rate, levelRange });
  }
  if (!slots.length) return null;
  return { baseName, slots };
}

function rowLooksLikeDust($, tr) {
  // 種族にモグリュー/ドリュウズが含まれる行は「土煙」寄りとみなす
  const tds = $(tr).find('td');
  for (let i = 1; i < tds.length; i++) {
    const txt = $(tds[i]).text();
    if (/モグリュー|ドリュウズ/.test(txt)) return true;
  }
  return false;
}

// 5枠の水用行パース（先頭はロケーション名(場合あり) + 5枠）
function parseWaterRowSlots($, tr, method, aliasJa) {
  const tds = $(tr).find('td');
  const startIdx = tds.length === 6 ? 1 : 0; // locセル付き=6、なし=5
  const rates = SLOT_RATE_PRESETS[method] || SLOT_RATE_PRESETS.Surfing;
  const slots = [];
  for (let i = 0; i < 5 && startIdx + i < tds.length; i++) {
    const cell = $(tds[startIdx + i]).text().trim();
    if (!cell) continue;
    const m = cell.match(/^([^()]+)\(([^)]+)\)$/);
    if (!m) continue;
    const rawName = canonicalizeSpeciesName(m[1]);
    const levelRange = parseLevelRangeFromText(m[2]);
    const speciesId = aliasJa.get(rawName);
    if (!speciesId) {
      MISSING_SPECIES.add(rawName);
      continue;
    }
    slots.push({ speciesId, rate: rates[i] ?? 1, levelRange });
  }
  return slots;
}

function makeSlotSignature(slot) {
  if (!slot) return '';
  const speciesId = slot.speciesId ?? 'null';
  const rate = slot.rate ?? 'null';
  const min = slot.levelRange?.min ?? 'null';
  const max = slot.levelRange?.max ?? 'null';
  return `${speciesId}:${rate}:${min}-${max}`;
}

function makeRowSignature(slots) {
  if (!slots || !slots.length) return '';
  return slots.map(makeSlotSignature).join('|');
}

function uniqueRowsBySignature(rows) {
  const seen = new Set();
  const uniques = [];
  for (const row of rows) {
    const signature = makeRowSignature(row.slots);
    if (!signature) continue;
    if (seen.has(signature)) continue;
    seen.add(signature);
    uniques.push(row);
  }
  return uniques;
}

function resolveLocationGroup(baseName, rows, { version, method, suffixRules }) {
  const uniques = uniqueRowsBySignature(rows);
  if (!uniques.length) return [];

  const plan = suffixRules ? suffixRules[baseName] : undefined;
  if (plan === undefined) {
    const merged = [];
    for (const row of rows) merged.push(...row.slots);
    return merged.length ? [{ displayName: baseName, slots: merged }] : [];
  }

  if (plan === null) {
    if (uniques.length > 1) {
      console.warn(
        `[warn] Expected identical rows for ${version}/${method}/${baseName}, found ${uniques.length} variants; using first variant.`
      );
    }
    return [{ displayName: baseName, slots: uniques[0].slots }];
  }

  if (!Array.isArray(plan) || !plan.length) {
    console.warn(`[warn] No valid suffix plan for ${version}/${method}/${baseName}; merging rows.`);
    const merged = [];
    for (const row of rows) merged.push(...row.slots);
    return merged.length ? [{ displayName: baseName, slots: merged }] : [];
  }

  if (uniques.length > plan.length) {
    console.warn(
      `[warn] Insufficient suffix entries for ${version}/${method}/${baseName}: need ${uniques.length}, have ${plan.length}. Extra variants reuse last suffix.`
    );
  }

  const result = [];
  for (let i = 0; i < uniques.length; i++) {
    const suffix = plan[Math.min(i, plan.length - 1)] ?? null;
    const displayName = suffix ? `${baseName} ${suffix}` : baseName;
    result.push({ displayName, slots: uniques[i].slots });
  }
  return result;
}

function isLocationRow($, tr) {
  const td0 = $(tr).find('td').first();
  if (!td0.length) return false;
  const txt = td0.text().trim();
  return /^\[[^\]]+\]/.test(txt) || /\S/.test(txt) && ($(tr).find('td').length >= 6);
}

function extractDisplayNameFromRow($, tr) {
  const td0 = $(tr).find('td').first();
  const locText = td0.text().trim();
  return locText.replace(/^\[[^\]]+\]\s*/, '').trim();
}

function parseWaterEncounterPage(html, { version, method, url, aliasJa }) {
  const $ = loadHtml(html);
  const rawLocations = new Map();
  const tables = findSectionTables($, method);
  if (!tables.length)
    return { version, method, source: { name: 'Pokebook', url, retrievedAt: todayISO() }, locations: {} };

  for (const tbl of tables) {
    const rows = $(tbl).find('tbody tr, tr').toArray();
    for (let i = 0; i < rows.length; i++) {
      const tr = rows[i];
      if (!isLocationRow($, tr)) continue;
      const displayName = extractDisplayNameFromRow($, tr);
      const group = [tr, rows[i + 1], rows[i + 2], rows[i + 3]].filter(Boolean);
      // 行インデックス→メソッド
      const indexToMethod = ['Surfing', 'SurfingBubble', 'Fishing', 'FishingBubble'];
      for (let gi = 0; gi < group.length; gi++) {
        const targetMethod = indexToMethod[gi];
        if (targetMethod !== method) continue;
        const slots = parseWaterRowSlots($, group[gi], method, aliasJa);
        if (!slots.length) continue;
        if (!rawLocations.has(displayName)) rawLocations.set(displayName, []);
        rawLocations.get(displayName).push(slots);
      }
      i += Math.max(0, group.length - 1); // グループ分スキップ
    }
  }

  const locations = {};
  for (const [displayName, rows] of rawLocations) {
    const seen = new Set();
    const mergedSlots = [];
    const options = WATER_SINGLE_ROW_LOCATIONS.get(displayName);
    const limitToOne = options && options.keepAll !== true;
    const enforceUnique = !options || options.keepAll !== true;
    for (const rowSlots of rows) {
      const signature = makeRowSignature(rowSlots);
      if (enforceUnique) {
        if (signature && seen.has(signature)) continue;
        if (signature) seen.add(signature);
      }
      mergedSlots.push(...rowSlots);
      if (limitToOne) break;
    }
    if (mergedSlots.length) {
      const normalizedKey = normalizeLocationKey(displayName);
      locations[normalizedKey] = { displayNameKey: normalizedKey, slots: mergedSlots };
      upsertLocationDisplayName(normalizedKey, displayName);
    }
  }

  return { version, method, source: { name: 'Pokebook', url, retrievedAt: todayISO() }, locations };
}

// セクションごとのパース
function parseEncounterPage(html, { version, method, url, aliasJa }) {
  const waterMethods = new Set(['Surfing', 'SurfingBubble', 'Fishing', 'FishingBubble']);
  if (waterMethods.has(method)) {
    return parseWaterEncounterPage(html, { version, method, url, aliasJa });
  }

  const $ = loadHtml(html);
  const locations = {};
  const parsedRows = [];

  const tables = findSectionTables($, method);
  if (!tables.length) {
    return { version, method, source: { name: 'Pokebook', url, retrievedAt: todayISO() }, locations };
  }

  for (const tbl of tables) {
    $(tbl)
      .find('tbody tr, tr')
      .each((_, tr) => {
        // 揺れる草/土煙は同一セクションのため、メソッドごとにフィルタ
        if (method === 'ShakingGrass' && rowLooksLikeDust($, tr)) return;
        if (method === 'DustCloud' && !rowLooksLikeDust($, tr)) return;

        const parsed = parseWideRowIntoSlots($, tr, method, aliasJa);
        if (!parsed) return;
        parsedRows.push(parsed);
      });
  }

  const suffixRules = ['Normal', 'ShakingGrass', 'DustCloud'].includes(method)
    ? DUPLICATE_SUFFIX_RULES[version]
    : undefined;
  const grouped = new Map();
  for (const row of parsedRows) {
    const key = row.baseName;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }

  for (const [baseName, rows] of grouped) {
    const resolved = resolveLocationGroup(baseName, rows, { version, method, suffixRules });
    for (const entry of resolved) {
      if (!entry.displayName || !entry.slots?.length) continue;
      const normalizedKey = normalizeLocationKey(entry.displayName);
      locations[normalizedKey] = { displayNameKey: normalizedKey, slots: entry.slots };
      upsertLocationDisplayName(normalizedKey, entry.displayName);
    }
  }

  return {
    version,
    method,
    source: { name: 'Pokebook', url, retrievedAt: todayISO() },
    locations,
  };
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function writeJson(file, data) {
  await ensureDir(path.dirname(file));
  await fs.writeFile(file, JSON.stringify(data, null, 2), 'utf8');
}

async function scrapeVersionMethod(version, method, overrideUrl) {
  // 各実行ごとに未知種名セットをリセット
  MISSING_SPECIES = new Set();
  const url = overrideUrl || SOURCE_MAP[version]?.[method];
  if (!url) {
    console.warn(`[skip] No source URL for ${version}/${method}`);
    return;
  }
  if (!METHODS.includes(method)) {
    console.warn(`[skip] Method ${method} not implemented yet`);
    return;
  }
  console.log(`[fetch] ${version}/${method} → ${url}`);
  const html = await fetchHtml(url);
  const aliasJa = await loadSpeciesAliasJa();
  const json = parseEncounterPage(html, { version, method, url, aliasJa });
  const outPath = path.resolve(
    'src/data/encounters/generated/v1',
    version,
    `${method}.json`
  );
  await writeJson(outPath, json);
  console.log(`[ok] wrote ${outPath} (${Object.keys(json.locations).length} locations)`);
  if (MISSING_SPECIES.size) {
    console.warn(`[warn] Unknown JP species (${MISSING_SPECIES.size}) for ${version}/${method}: ${[...MISSING_SPECIES].join(', ')}`);
  }
}

function parseArgs() {
  const get = (k) => process.argv.find((a) => a.startsWith(`--${k}=`))?.split('=')[1];
  return {
    version: get('version'),
    method: get('method'),
    url: get('url'),
  };
}

async function main() {
  displayNameDictionary = await loadDisplayNameDictionary();

  const args = parseArgs();
  const versions = args.version ? [args.version] : VERSIONS;
  const methods = args.method ? [args.method] : METHODS;

  for (const v of versions) {
    for (const m of methods) {
      try {
        await scrapeVersionMethod(v, m, args.url);
      } catch (e) {
        console.error(`[error] ${v}/${m}:`, e.message);
      }
    }
  }

  await persistDisplayNameDictionary();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
