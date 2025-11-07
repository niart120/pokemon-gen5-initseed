#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const GENERATED_ROOT = path.join(ROOT, 'src', 'data', 'encounters', 'generated', 'v1');
const STATIC_ROOT = path.join(ROOT, 'src', 'data', 'encounters', 'static', 'v1');
const I18N_DIR = path.join(ROOT, 'src', 'data', 'encounters', 'i18n');
const DICTIONARY_JSON = path.join(I18N_DIR, 'display-names.json');

/**
 * Normalize location keys to match loader behavior.
 */
function normalizeLocationKey(input) {
  return input
    .trim()
    .replace(/[\u3000\s]+/g, '')
    .replace(/[‐‑‒–—−\-_.]/g, '');
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function readJson(file) {
  const text = await fs.readFile(file, 'utf8');
  return JSON.parse(text);
}

async function writeJson(file, data) {
  const text = JSON.stringify(data, null, 2);
  await fs.writeFile(file, `${text}\n`, 'utf8');
}

async function listJsonFiles(root) {
  const out = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (entry.isFile() && entry.name.endsWith('.json')) {
        out.push(full);
      }
    }
  }
  await walk(root);
  return out;
}

function sortKeys(obj) {
  return Object.fromEntries(Object.keys(obj).sort().map(key => [key, obj[key]]));
}

function upsertDictionaryBucket(bucket, key, jaValue, enValue) {
  if (!bucket[key]) {
    bucket[key] = { ja: jaValue, en: enValue ?? jaValue };
    return;
  }
  const entry = bucket[key];
  if (!entry.ja) entry.ja = jaValue;
  if (!entry.en) entry.en = enValue ?? entry.ja ?? jaValue;
}

async function migrateLocations(dictionary) {
  const files = await listJsonFiles(GENERATED_ROOT);
  for (const file of files) {
    const data = await readJson(file);
    if (!data || !data.locations) continue;
    const nextLocations = {};
    for (const [rawKey, payload] of Object.entries(data.locations)) {
      if (!payload || !payload.slots) continue;
      const normalizedKey = normalizeLocationKey(rawKey);
      const displayName = payload.displayName ?? rawKey;
      const displayNameKey = payload.displayNameKey ?? normalizedKey;
      upsertDictionaryBucket(dictionary.locations, displayNameKey, displayName, displayName);
      nextLocations[normalizedKey] = {
        displayNameKey,
        slots: payload.slots,
      };
    }
    data.locations = nextLocations;
    await writeJson(file, data);
  }
}

async function migrateStatic(dictionary) {
  const files = await listJsonFiles(STATIC_ROOT);
  for (const file of files) {
    const data = await readJson(file);
    if (!data || !Array.isArray(data.entries)) continue;
    for (const entry of data.entries) {
      if (!entry) continue;
      const displayName = entry.displayName ?? entry.id;
      const displayNameKey = entry.displayNameKey ?? entry.id;
      upsertDictionaryBucket(dictionary.static, displayNameKey, displayName, displayName);
      entry.displayNameKey = displayNameKey;
      delete entry.displayName;
    }
    await writeJson(file, data);
  }
}

async function loadExistingDictionary() {
  try {
    const json = await readJson(DICTIONARY_JSON);
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

async function main() {
  await ensureDir(I18N_DIR);
  const dictionary = await loadExistingDictionary();
  await migrateLocations(dictionary);
  await migrateStatic(dictionary);
  dictionary.locations = sortKeys(dictionary.locations);
  dictionary.static = sortKeys(dictionary.static);
  if (dictionary.categories) dictionary.categories = sortKeys(dictionary.categories);
  if (dictionary.types) dictionary.types = sortKeys(dictionary.types);
  await writeJson(DICTIONARY_JSON, dictionary);
  console.log('Encounter display name migration completed.');
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
