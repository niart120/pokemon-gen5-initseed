// wasm_bindgen types cannot implement Default trait, so we suppress these warnings
#![allow(clippy::new_without_default)]
// Some WASM API functions require many arguments by design
#![allow(clippy::too_many_arguments)]
// PersonalityRNG::next is intentionally named to match common RNG patterns
#![allow(clippy::should_implement_trait)]

mod datetime_codes;
mod egg_boot_timing_search;
mod egg_iv;
mod egg_seed_enumerator;
mod encounter_calculator;
mod integrated_search;
mod mt19937;
mod offset_calculator;
mod personality_rng;
mod pid_shiny_checker;
mod pokemon_generator;
mod sha1;
mod sha1_simd;
mod utils;

#[cfg(test)]
mod tests;

// Re-export main functionality - 統合検索のみ（内部でsha1/sha1_simdは使用）
pub use datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
pub use egg_boot_timing_search::{EggBootTimingSearchResult, EggBootTimingSearcher};
pub use egg_iv::{
    derive_pending_egg, derive_pending_egg_with_state, hidden_power_from_iv, matches_filter,
    resolve_egg_iv, resolve_npc_advance, AbilitySlot, EggIvError, EverstonePlan, EverstonePlanJs,
    Gender, GenderRatio, GenerationConditions, GenerationConditionsJs, HiddenPowerInfo,
    HiddenPowerType, IVResolutionConditions, IndividualFilter, IndividualFilterJs, InheritanceSlot,
    IvSet, IvValue, Nature, ParentRole, PendingEgg, ResolvedEgg, StatIndex, StatRange, TrainerIds,
    IV_VALUE_UNKNOWN,
};
pub use egg_seed_enumerator::{
    derive_mt_seed, EggSeedEnumerator, EggSeedEnumeratorJs, EnumeratedEggData, ParentsIVs,
    ParentsIVsJs,
};
pub use encounter_calculator::{EncounterCalculator, EncounterType, GameVersion};
pub use integrated_search::{IntegratedSeedSearcher, SearchResult};
pub use offset_calculator::{
    calculate_game_offset, calculate_tid_sid_from_seed, ExtraResult, GameMode, OffsetCalculator,
    TidSidResult,
};
pub use personality_rng::PersonalityRNG;
pub use pid_shiny_checker::{PIDCalculator, ShinyChecker, ShinyType};
pub use pokemon_generator::{
    BWGenerationConfig, EnumeratedPokemonData, PokemonGenerator, RawPokemonData, SeedEnumerator,
};
pub use utils::{ArrayUtils, BitUtils, EndianUtils, NumberUtils, ValidationUtils};
