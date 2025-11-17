mod datetime_codes;
mod encounter_calculator;
mod integrated_search;
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
pub use encounter_calculator::{EncounterCalculator, EncounterType, GameVersion};
pub use integrated_search::{IntegratedSeedSearcher, SearchResult};
pub use offset_calculator::{
    calculate_game_offset, calculate_tid_sid_from_seed, ExtraResult, GameMode, OffsetCalculator,
    TidSidResult,
};
pub use personality_rng::PersonalityRNG;
pub use pid_shiny_checker::{PIDCalculator, ShinyChecker, ShinyType};
pub use pokemon_generator::{
    BWGenerationConfig,
    EnumeratedPokemonData,
    PokemonGenerator,
    RawPokemonData,
    SeedEnumerator,
};
pub use utils::{ArrayUtils, BitUtils, EndianUtils, NumberUtils, ValidationUtils};
