// Generate species database (Gen1–Gen5) with localization and held items from PokéAPI
// Output: src/data/species/generated/gen5-species.json
// Node 18+ required (global fetch)

import { writeFile, mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const START_ID = 1;   // Gen 1 start
const END_ID = 649;   // Genesect (end of Gen 5)
const OUT_DIR = path.resolve(process.cwd(), 'src/data/species/generated');
const OUT_PATH = path.join(OUT_DIR, 'gen5-species.json');

// Gen 5 versions we care about for held items
const GEN5_VERSIONS = new Set(['black', 'white', 'black-2', 'white-2']);

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function titleCase(name) {
  return name
    .split('-')
    .map(w => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

function pickLang(names, lang) {
  return names?.find(n => n.language?.name === lang)?.name;
}

function getLocalized(namesArr) {
  // Prefer ja, fallback ja-Hrkt, else en
  const ja = pickLang(namesArr, 'ja') || pickLang(namesArr, 'ja-Hrkt');
  const en = pickLang(namesArr, 'en');
  return { en: en || null, ja: ja || null };
}

function deriveGenderSpec(genderRate) {
  // genderRate: -1 (genderless) or 0..8 = female rate in eighths
  if (genderRate === -1) return { type: 'genderless' };
  if (genderRate === 0) return { type: 'fixed', fixed: 'male' };
  if (genderRate === 8) return { type: 'fixed', fixed: 'female' };
  const femaleThreshold = Math.floor((genderRate / 8) * 256);
  return { type: 'ratio', femaleThreshold };
}

async function fetchJson(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'gen5-species-fetch-script' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function fetchAbilityNames(keys) {
  const map = {};
  for (const key of keys) {
    try {
      const data = await fetchJson(`https://pokeapi.co/api/v2/ability/${key}`);
      const names = getLocalized(data.names);
      // Use API canonical key, keep original key
      map[key] = { key, names };
      await sleep(50);
    } catch (e) {
      map[key] = { key, names: { en: titleCase(key), ja: null } };
      await sleep(100);
    }
  }
  return map;
}

async function fetchItemNames(keys) {
  const map = {};
  for (const key of keys) {
    try {
      const data = await fetchJson(`https://pokeapi.co/api/v2/item/${key}`);
      const names = getLocalized(data.names);
      map[key] = { key, names };
      await sleep(50);
    } catch (e) {
      map[key] = { key, names: { en: titleCase(key), ja: null } };
      await sleep(100);
    }
  }
  return map;
}

async function main() {
  const db = {};

  // For localization of ability/item keys
  const abilityKeys = new Set();
  const itemKeys = new Set();

  for (let id = START_ID; id <= END_ID; id++) {
    try {
      const speciesUrl = `https://pokeapi.co/api/v2/pokemon-species/${id}`;
      const pokemonUrl = `https://pokeapi.co/api/v2/pokemon/${id}`;

      const species = await fetchJson(speciesUrl);
      const pokemon = await fetchJson(pokemonUrl);

      const speciesNames = getLocalized(species.names);

      const gender = deriveGenderSpec(species.gender_rate);

      // Base stats from pokemon.stats
      const stats = Object.fromEntries(
        (pokemon.stats || []).map(s => [s.stat?.name, s.base_stat])
      );
      const baseStats = {
        hp: stats['hp'] ?? null,
        attack: stats['attack'] ?? null,
        defense: stats['defense'] ?? null,
        specialAttack: stats['special-attack'] ?? null,
        specialDefense: stats['special-defense'] ?? null,
        speed: stats['speed'] ?? null,
      };

      // Abilities: capture keys first
      const abilities = pokemon.abilities || [];
      const normals = abilities
        .filter(a => !a.is_hidden)
        .sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99));
      const hidden = abilities.find(a => a.is_hidden) || null;

      const ability1Key = normals[0]?.ability?.name || null;
      const ability2Key = normals[1]?.ability?.name || null;
      const hiddenKey = hidden?.ability?.name || null;

      if (ability1Key) abilityKeys.add(ability1Key);
      if (ability2Key) abilityKeys.add(ability2Key);
      if (hiddenKey) abilityKeys.add(hiddenKey);

      // Held items per Gen 5 versions
      const held = { 'black': [], 'white': [], 'black-2': [], 'white-2': [] };
      for (const hi of pokemon.held_items || []) {
        const key = hi.item?.name;
        if (!key) continue;
        // version_details: [{ rarity, version: { name } }]
        for (const vd of hi.version_details || []) {
          const vname = vd.version?.name;
          if (GEN5_VERSIONS.has(vname)) {
            held[vname].push({ key, rarity: vd.rarity ?? null });
            itemKeys.add(key);
          }
        }
      }

      db[id] = {
        nationalDex: id,
        names: {
          en: speciesNames.en ? titleCase(speciesNames.en.replace(/_/g, '-')) : null,
          ja: speciesNames.ja || null,
        },
        gender,
        baseStats,
        abilities: {
          ability1: ability1Key,
          ability2: ability2Key,
          hidden: hiddenKey,
        },
        heldItems: held,
      };

      // Polite delay between species/pokemon pair
      await sleep(60);
    } catch (err) {
      console.error(`Failed id ${id}:`, err?.message || err);
      db[id] = db[id] || {
        nationalDex: id,
        names: { en: `ID-${id}`, ja: null },
        gender: { type: 'genderless' },
        baseStats: { hp: null, attack: null, defense: null, specialAttack: null, specialDefense: null, speed: null },
        abilities: { ability1: null, ability2: null, hidden: null },
        heldItems: { 'black': [], 'white': [], 'black-2': [], 'white-2': [] },
      };
      await sleep(120);
    }
  }

  // Fetch localization for abilities and items
  const abilityMap = await fetchAbilityNames(abilityKeys);
  const itemMap = await fetchItemNames(itemKeys);

  // Enrich db with localized ability/item names
  for (let id = START_ID; id <= END_ID; id++) {
    const entry = db[id];
    if (!entry) continue;

    const a1 = entry.abilities.ability1 ? abilityMap[entry.abilities.ability1] : null;
    const a2 = entry.abilities.ability2 ? abilityMap[entry.abilities.ability2] : null;
    const ah = entry.abilities.hidden ? abilityMap[entry.abilities.hidden] : null;

    entry.abilities = {
      ability1: a1 ? { key: a1.key, names: a1.names } : null,
      ability2: a2 ? { key: a2.key, names: a2.names } : null,
      hidden: ah ? { key: ah.key, names: ah.names } : null,
    };

    for (const ver of Object.keys(entry.heldItems)) {
      entry.heldItems[ver] = entry.heldItems[ver].map(it => {
        const meta = itemMap[it.key] || { key: it.key, names: { en: titleCase(it.key), ja: null } };
        return { key: meta.key, names: meta.names, rarity: it.rarity };
      });
    }
  }

  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(OUT_PATH, JSON.stringify(db, null, 2), 'utf-8');
  console.log(`✔ Wrote ${OUT_PATH}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
