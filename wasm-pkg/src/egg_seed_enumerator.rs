use crate::egg_iv::{
    derive_pending_egg_with_state, matches_filter, resolve_egg_iv, resolve_npc_advance, EggIvError,
    GenerationConditions, GenerationConditionsJs, IVResolutionConditions, IndividualFilter,
    IndividualFilterJs, IvSet, ResolvedEgg,
};
use crate::mt19937::Mt19937;
use crate::offset_calculator::{calculate_game_offset, GameMode};
use crate::personality_rng::PersonalityRNG;
use wasm_bindgen::prelude::*;

const NPC_FRAME_THRESHOLD: u8 = 96;
const NPC_FRAME_SLACK: u8 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentsIVs {
    pub male: IvSet,
    pub female: IvSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnumeratedEggData {
    pub advance: u64,
    pub egg: ResolvedEgg,
    pub is_stable: bool,
}

impl EnumeratedEggData {
    fn new(advance: u64, egg: ResolvedEgg, is_stable: bool) -> Self {
        EnumeratedEggData {
            advance,
            egg,
            is_stable,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EggSeedEnumerator {
    current_seed: u64,
    next_advance: u64,
    target_count: u32,
    produced: u32,
    conditions: GenerationConditions,
    iv_sources: IVResolutionConditions,
    consider_npc_consumption: bool,
    filter: Option<IndividualFilter>,
}

impl EggSeedEnumerator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_seed: u64,
        user_offset: u64,
        count: u32,
        conditions: GenerationConditions,
        parents: ParentsIVs,
        filter: Option<IndividualFilter>,
        consider_npc_consumption: bool,
        game_mode: GameMode,
    ) -> EggSeedEnumerator {
        let game_offset = calculate_game_offset(base_seed, game_mode) as u64;
        let (combined_offset, overflowed) = game_offset.overflowing_add(user_offset);
        let total_offset = if overflowed {
            u64::MAX
        } else {
            combined_offset
        };
        let (mul, add) = PersonalityRNG::lcg_affine_for_steps(total_offset);
        let current_seed = PersonalityRNG::lcg_apply(base_seed, mul, add);

        let iv_sources = build_iv_sources(base_seed, parents);

        EggSeedEnumerator {
            current_seed,
            next_advance: user_offset,
            target_count: count,
            produced: 0,
            conditions,
            iv_sources,
            consider_npc_consumption,
            filter,
        }
    }

    pub fn next_egg(&mut self) -> Result<Option<EnumeratedEggData>, EggIvError> {
        if self.produced >= self.target_count {
            return Ok(None);
        }

        loop {
            if self.produced >= self.target_count {
                return Ok(None);
            }

            let current_advance = self.next_advance;
            self.next_advance = self.next_advance.saturating_add(1);
            let (seed_after_npc, is_stable) = if self.consider_npc_consumption {
                let (next_seed, _consumed, stable) =
                    resolve_npc_advance(self.current_seed, NPC_FRAME_THRESHOLD, NPC_FRAME_SLACK);
                (next_seed, stable)
            } else {
                (self.current_seed, false)
            };

            let (pending, final_seed) =
                derive_pending_egg_with_state(seed_after_npc, &self.conditions);
            let resolved = resolve_egg_iv(&pending, &self.iv_sources)?;

            self.current_seed = PersonalityRNG::next_seed(self.current_seed);
            self.produced = self.produced.saturating_add(1);

            let passes = self
                .filter
                .as_ref()
                .map_or(true, |filter| matches_filter(&resolved, filter));

            if passes {
                return Ok(Some(EnumeratedEggData::new(
                    current_advance,
                    resolved,
                    is_stable,
                )));
            }

            if self.produced >= self.target_count {
                return Ok(None);
            }
        }
    }

    pub fn remaining(&self) -> u32 {
        self.target_count.saturating_sub(self.produced)
    }
}

fn build_iv_sources(base_seed: u64, parents: ParentsIVs) -> IVResolutionConditions {
    let mt_seed = derive_mt_seed(base_seed);
    let rng = generate_rng_iv_set(mt_seed);
    IVResolutionConditions {
        male: parents.male,
        female: parents.female,
        rng,
    }
}

pub fn derive_mt_seed(base_seed: u64) -> u32 {
    let mut rng = PersonalityRNG::new(base_seed);
    rng.next()
}

pub(crate) fn generate_rng_iv_set(mt_seed: u32) -> IvSet {
    let mut mt = Mt19937::new(mt_seed);
    for _ in 0..7 {
        mt.next_u32();
    }

    let mut ivs: IvSet = [0; 6];
    for value in ivs.iter_mut() {
        *value = (mt.next_u32() >> 27) as u8;
    }
    ivs
}

#[cfg(test)]
mod tests {
    use super::{derive_mt_seed, generate_rng_iv_set};

    #[test]
    fn generate_rng_iv_set_repeats_deterministically() {
        let seed = 0x1234_5678_9ABC_DEF0;
        let mt_seed = derive_mt_seed(seed);
        let first = generate_rng_iv_set(mt_seed);
        let second = generate_rng_iv_set(mt_seed);
        assert_eq!(first, second);
    }
}

// ========================================
// WASM-bindgen wrappers for JS interop
// ========================================

/// WASM wrapper for ParentsIVs
#[wasm_bindgen]
pub struct ParentsIVsJs {
    male: IvSet,
    female: IvSet,
}

#[wasm_bindgen]
impl ParentsIVsJs {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ParentsIVsJs {
        ParentsIVsJs {
            male: [0; 6],
            female: [0; 6],
        }
    }

    #[wasm_bindgen(setter = male)]
    pub fn set_male(&mut self, ivs: Vec<u8>) {
        if ivs.len() >= 6 {
            self.male = [ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]];
        }
    }

    #[wasm_bindgen(setter = female)]
    pub fn set_female(&mut self, ivs: Vec<u8>) {
        if ivs.len() >= 6 {
            self.female = [ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]];
        }
    }
}

impl ParentsIVsJs {
    pub fn to_internal(&self) -> ParentsIVs {
        ParentsIVs {
            male: self.male,
            female: self.female,
        }
    }
}

impl Default for ParentsIVsJs {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM wrapper for EggSeedEnumerator
#[wasm_bindgen]
pub struct EggSeedEnumeratorJs {
    inner: EggSeedEnumerator,
}

#[wasm_bindgen]
impl EggSeedEnumeratorJs {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_seed: u64,
        user_offset: u64,
        count: u32,
        conditions: &GenerationConditionsJs,
        parents: &ParentsIVsJs,
        filter: &IndividualFilterJs,
        consider_npc_consumption: bool,
        game_mode: GameMode,
    ) -> EggSeedEnumeratorJs {
        let internal_filter = Some(filter.to_internal());
        let internal_conditions = conditions.to_internal();
        let internal_parents = parents.to_internal();

        EggSeedEnumeratorJs {
            inner: EggSeedEnumerator::new(
                base_seed,
                user_offset,
                count,
                internal_conditions,
                internal_parents,
                internal_filter,
                consider_npc_consumption,
                game_mode,
            ),
        }
    }

    /// Returns the next egg as a JsValue or undefined if exhausted
    pub fn next_egg(&mut self) -> JsValue {
        match self.inner.next_egg() {
            Ok(Some(data)) => {
                // Convert EnumeratedEggData to JS-compatible object
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &"advance".into(), &JsValue::from_f64(data.advance as f64)).ok();
                js_sys::Reflect::set(&obj, &"is_stable".into(), &JsValue::from_bool(data.is_stable)).ok();

                // Create egg object
                let egg_obj = js_sys::Object::new();
                let ivs = js_sys::Array::new();
                for iv in &data.egg.ivs {
                    ivs.push(&JsValue::from(*iv));
                }
                js_sys::Reflect::set(&egg_obj, &"ivs".into(), &ivs).ok();
                js_sys::Reflect::set(&egg_obj, &"nature".into(), &JsValue::from(data.egg.nature as u8)).ok();
                js_sys::Reflect::set(&egg_obj, &"gender".into(), &JsValue::from(match data.egg.gender {
                    crate::egg_iv::Gender::Male => 0u8,
                    crate::egg_iv::Gender::Female => 1u8,
                    crate::egg_iv::Gender::Genderless => 2u8,
                })).ok();
                js_sys::Reflect::set(&egg_obj, &"ability".into(), &JsValue::from(data.egg.ability as u8)).ok();
                js_sys::Reflect::set(&egg_obj, &"shiny".into(), &JsValue::from(data.egg.shiny as u8)).ok();
                js_sys::Reflect::set(&egg_obj, &"pid".into(), &JsValue::from(data.egg.pid)).ok();

                // Hidden power
                let hp_obj = match data.egg.hidden_power {
                    crate::egg_iv::HiddenPowerInfo::Known { r#type, power } => {
                        let hp = js_sys::Object::new();
                        js_sys::Reflect::set(&hp, &"type".into(), &"known".into()).ok();
                        js_sys::Reflect::set(&hp, &"hp_type".into(), &JsValue::from(r#type as u8)).ok();
                        js_sys::Reflect::set(&hp, &"power".into(), &JsValue::from(power)).ok();
                        hp
                    }
                    crate::egg_iv::HiddenPowerInfo::Unknown => {
                        let hp = js_sys::Object::new();
                        js_sys::Reflect::set(&hp, &"type".into(), &"unknown".into()).ok();
                        hp
                    }
                };
                js_sys::Reflect::set(&egg_obj, &"hidden_power".into(), &hp_obj).ok();

                js_sys::Reflect::set(&obj, &"egg".into(), &egg_obj).ok();
                obj.into()
            }
            Ok(None) => JsValue::UNDEFINED,
            Err(_) => JsValue::UNDEFINED,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn remaining(&self) -> u32 {
        self.inner.remaining()
    }
}
