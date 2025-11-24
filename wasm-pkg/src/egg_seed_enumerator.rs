use crate::egg_iv::{
    derive_pending_egg_with_state, matches_filter, resolve_egg_iv, resolve_npc_advance, EggIvError,
    GenerationConditions, IVResolutionConditions, IndividualFilter, IvSet, ResolvedEgg,
};
use crate::mt19937::Mt19937;
use crate::offset_calculator::{calculate_game_offset, GameMode};
use crate::personality_rng::PersonalityRNG;

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
        let total_offset = if overflowed { u64::MAX } else { combined_offset };
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
                let (next_seed, _consumed, stable) = resolve_npc_advance(
                    self.current_seed,
                    NPC_FRAME_THRESHOLD,
                    NPC_FRAME_SLACK,
                );
                (next_seed, stable)
            } else {
                (self.current_seed, false)
            };

            let (pending, final_seed) =
                derive_pending_egg_with_state(seed_after_npc, &self.conditions);
            let resolved = resolve_egg_iv(&pending, &self.iv_sources)?;

            self.current_seed = final_seed;
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
    use super::{generate_rng_iv_set, derive_mt_seed};

    #[test]
    fn generate_rng_iv_set_repeats_deterministically() {
        let seed = 0x1234_5678_9ABC_DEF0;
        let mt_seed = derive_mt_seed(seed);
        let first = generate_rng_iv_set(mt_seed);
        let second = generate_rng_iv_set(mt_seed);
        assert_eq!(first, second);
    }
}
