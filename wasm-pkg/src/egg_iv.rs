use crate::personality_rng::PersonalityRNG;
use crate::pid_shiny_checker::{ShinyChecker, ShinyType};
use std::convert::TryInto;
use wasm_bindgen::prelude::*;

/// Unknown IV sentinel value shared between Rust/WASM/TS
pub const IV_VALUE_UNKNOWN: IvValue = 32;
const STAT_COUNT: usize = 6;
const MAX_INHERIT_SLOTS: usize = 3;
const FOUR_FRACTION_FRAMES: [u32; 4] = [32, 64, 96, 128];
const LEFT_DIRECTION_FRAMES: u32 = 20;
const RIGHT_DIRECTION_FRAMES: u32 = 16;
const DIRECTION_MISMATCH_FRAMES: u32 = 20;
const INITIAL_NPC_ADVANCE_COST: u32 = 3;
const FINAL_NPC_ADVANCE_COST: u32 = 2;

/// Individual IV value (0-31) or Unknown (32)
pub type IvValue = u8;
/// Ordered IV set: HP, Atk, Def, SpA, SpD, Spe
pub type IvSet = [IvValue; STAT_COUNT];

/// Errors that can occur while handling egg IVs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EggIvError {
    InvalidIvValue(IvValue),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenPowerType {
    Fighting = 0,
    Flying = 1,
    Poison = 2,
    Ground = 3,
    Rock = 4,
    Bug = 5,
    Ghost = 6,
    Steel = 7,
    Fire = 8,
    Water = 9,
    Grass = 10,
    Electric = 11,
    Psychic = 12,
    Ice = 13,
    Dragon = 14,
    Dark = 15,
}

impl HiddenPowerType {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => HiddenPowerType::Fighting,
            1 => HiddenPowerType::Flying,
            2 => HiddenPowerType::Poison,
            3 => HiddenPowerType::Ground,
            4 => HiddenPowerType::Rock,
            5 => HiddenPowerType::Bug,
            6 => HiddenPowerType::Ghost,
            7 => HiddenPowerType::Steel,
            8 => HiddenPowerType::Fire,
            9 => HiddenPowerType::Water,
            10 => HiddenPowerType::Grass,
            11 => HiddenPowerType::Electric,
            12 => HiddenPowerType::Psychic,
            13 => HiddenPowerType::Ice,
            14 => HiddenPowerType::Dragon,
            _ => HiddenPowerType::Dark,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenPowerInfo {
    Known { r#type: HiddenPowerType, power: u8 },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub struct StatRange {
    pub min: IvValue,
    pub max: IvValue,
}

#[wasm_bindgen]
impl StatRange {
    #[wasm_bindgen(constructor)]
    pub fn new(min: IvValue, max: IvValue) -> Self {
        StatRange { min, max }
    }

    pub fn contains(&self, value: IvValue) -> bool {
        if self.min > self.max {
            return false;
        }
        value >= self.min && value <= self.max
    }
}

impl StatRange {
    pub const fn unrestricted() -> Self {
        StatRange {
            min: 0,
            max: IV_VALUE_UNKNOWN,
        }
    }
}

impl Default for StatRange {
    fn default() -> Self {
        StatRange::unrestricted()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
    Genderless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub struct GenderRatio {
    pub threshold: u8,
    pub genderless: bool,
}

#[wasm_bindgen]
impl GenderRatio {
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: u8, genderless: bool) -> Self {
        GenderRatio { threshold, genderless }
    }

    pub fn resolve(&self, gender_value: u8) -> u8 {
        if self.genderless {
            return 2; // Genderless
        }

        if gender_value < self.threshold {
            1 // Female
        } else {
            0 // Male
        }
    }
}

impl GenderRatio {
    pub fn resolve_enum(&self, gender_value: u8) -> Gender {
        if self.genderless {
            return Gender::Genderless;
        }

        if gender_value < self.threshold {
            Gender::Female
        } else {
            Gender::Male
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbilitySlot {
    One = 0,
    Two = 1,
    Hidden = 2,
}

impl AbilitySlot {
    pub fn from_bit(bit: u8) -> Self {
        match bit {
            0 => AbilitySlot::One,
            1 => AbilitySlot::Two,
            _ => AbilitySlot::One,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nature {
    Hardy = 0,
    Lonely = 1,
    Brave = 2,
    Adamant = 3,
    Naughty = 4,
    Bold = 5,
    Docile = 6,
    Relaxed = 7,
    Impish = 8,
    Lax = 9,
    Timid = 10,
    Hasty = 11,
    Serious = 12,
    Jolly = 13,
    Naive = 14,
    Modest = 15,
    Mild = 16,
    Quiet = 17,
    Bashful = 18,
    Rash = 19,
    Calm = 20,
    Gentle = 21,
    Sassy = 22,
    Careful = 23,
    Quirky = 24,
}

impl Nature {
    pub fn from_roll(roll: u32) -> Self {
        let index = (roll % 25) as u8;
        Nature::from_index(index)
    }

    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Nature::Hardy,
            1 => Nature::Lonely,
            2 => Nature::Brave,
            3 => Nature::Adamant,
            4 => Nature::Naughty,
            5 => Nature::Bold,
            6 => Nature::Docile,
            7 => Nature::Relaxed,
            8 => Nature::Impish,
            9 => Nature::Lax,
            10 => Nature::Timid,
            11 => Nature::Hasty,
            12 => Nature::Serious,
            13 => Nature::Jolly,
            14 => Nature::Naive,
            15 => Nature::Modest,
            16 => Nature::Mild,
            17 => Nature::Quiet,
            18 => Nature::Bashful,
            19 => Nature::Rash,
            20 => Nature::Calm,
            21 => Nature::Gentle,
            22 => Nature::Sassy,
            23 => Nature::Careful,
            _ => Nature::Quirky,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParentRole {
    Male,
    Female,
}

impl ParentRole {
    fn from_bit(bit: u32) -> Self {
        if bit == 0 {
            ParentRole::Male
        } else {
            ParentRole::Female
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatIndex {
    Hp = 0,
    Attack = 1,
    Defense = 2,
    SpecialAttack = 3,
    SpecialDefense = 4,
    Speed = 5,
}

impl StatIndex {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(StatIndex::Hp),
            1 => Some(StatIndex::Attack),
            2 => Some(StatIndex::Defense),
            3 => Some(StatIndex::SpecialAttack),
            4 => Some(StatIndex::SpecialDefense),
            5 => Some(StatIndex::Speed),
            _ => None,
        }
    }

    pub fn as_usize(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InheritanceSlot {
    pub stat: StatIndex,
    pub parent: ParentRole,
}

impl InheritanceSlot {
    pub fn new(stat: StatIndex, parent: ParentRole) -> Self {
        InheritanceSlot { stat, parent }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EverstonePlan {
    None,
    Fixed(Nature),
}

/// WASM wrapper for EverstonePlan
#[wasm_bindgen]
pub struct EverstonePlanJs {
    inner: EverstonePlan,
}

#[wasm_bindgen]
impl EverstonePlanJs {
    #[wasm_bindgen(getter = None)]
    pub fn none() -> EverstonePlanJs {
        EverstonePlanJs { inner: EverstonePlan::None }
    }

    pub fn fixed(nature_index: u8) -> EverstonePlanJs {
        EverstonePlanJs { inner: EverstonePlan::Fixed(Nature::from_index(nature_index)) }
    }
}

impl EverstonePlanJs {
    pub fn unwrap(&self) -> EverstonePlan {
        self.inner
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub struct TrainerIds {
    pub tid: u16,
    pub sid: u16,
    pub tsv: u16,
}

#[wasm_bindgen]
impl TrainerIds {
    #[wasm_bindgen(constructor)]
    pub fn new(tid: u16, sid: u16) -> Self {
        TrainerIds {
            tid,
            sid,
            tsv: tid ^ sid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationConditions {
    pub has_nidoran_flag: bool,
    pub everstone: EverstonePlan,
    pub uses_ditto: bool,
    pub allow_hidden_ability: bool,
    pub female_parent_has_hidden: bool,
    pub reroll_count: u8,
    pub trainer_ids: TrainerIds,
    pub gender_ratio: GenderRatio,
}

/// WASM wrapper for GenerationConditions
#[wasm_bindgen]
pub struct GenerationConditionsJs {
    pub has_nidoran_flag: bool,
    pub uses_ditto: bool,
    pub allow_hidden_ability: bool,
    pub female_parent_has_hidden: bool,
    pub reroll_count: u8,
    everstone: EverstonePlan,
    trainer_ids: TrainerIds,
    gender_ratio: GenderRatio,
}

#[wasm_bindgen]
impl GenerationConditionsJs {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GenerationConditionsJs {
        GenerationConditionsJs {
            has_nidoran_flag: false,
            uses_ditto: false,
            allow_hidden_ability: false,
            female_parent_has_hidden: false,
            reroll_count: 0,
            everstone: EverstonePlan::None,
            trainer_ids: TrainerIds::new(0, 0),
            gender_ratio: GenderRatio::new(127, false),
        }
    }

    pub fn set_everstone(&mut self, plan: &EverstonePlanJs) {
        self.everstone = plan.unwrap();
    }

    pub fn set_trainer_ids(&mut self, ids: &TrainerIds) {
        self.trainer_ids = *ids;
    }

    pub fn set_gender_ratio(&mut self, ratio: &GenderRatio) {
        self.gender_ratio = *ratio;
    }
}

impl GenerationConditionsJs {
    pub fn to_internal(&self) -> GenerationConditions {
        GenerationConditions {
            has_nidoran_flag: self.has_nidoran_flag,
            everstone: self.everstone,
            uses_ditto: self.uses_ditto,
            allow_hidden_ability: self.allow_hidden_ability,
            female_parent_has_hidden: self.female_parent_has_hidden,
            reroll_count: self.reroll_count,
            trainer_ids: self.trainer_ids,
            gender_ratio: self.gender_ratio,
        }
    }
}

impl Default for GenerationConditionsJs {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingEgg {
    pub inherits: [InheritanceSlot; MAX_INHERIT_SLOTS],
    pub nature: Nature,
    pub gender: Gender,
    pub ability: AbilitySlot,
    pub shiny: ShinyType,
    pub pid: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IVResolutionConditions {
    pub male: IvSet,
    pub female: IvSet,
    pub rng: IvSet,
    pub mt_seed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedEgg {
    pub lcg_seed: u64,
    pub mt_seed: u32,
    pub ivs: IvSet,
    pub nature: Nature,
    pub gender: Gender,
    pub ability: AbilitySlot,
    pub shiny: ShinyType,
    pub pid: u32,
    pub hidden_power: HiddenPowerInfo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndividualFilter {
    pub iv_ranges: [StatRange; STAT_COUNT],
    pub nature: Option<Nature>,
    pub gender: Option<Gender>,
    pub ability: Option<AbilitySlot>,
    pub shiny: Option<ShinyType>,
    pub hidden_power_type: Option<HiddenPowerType>,
    pub hidden_power_power: Option<u8>,
}

impl Default for IndividualFilter {
    fn default() -> Self {
        IndividualFilter {
            iv_ranges: [StatRange::default(); STAT_COUNT],
            nature: None,
            gender: None,
            ability: None,
            shiny: None,
            hidden_power_type: None,
            hidden_power_power: None,
        }
    }
}

/// WASM wrapper for IndividualFilter
#[wasm_bindgen]
pub struct IndividualFilterJs {
    inner: IndividualFilter,
}

#[wasm_bindgen]
impl IndividualFilterJs {
    #[wasm_bindgen(constructor)]
    pub fn new() -> IndividualFilterJs {
        IndividualFilterJs { inner: IndividualFilter::default() }
    }

    pub fn set_iv_range(&mut self, stat_index: u8, min: u8, max: u8) {
        if (stat_index as usize) < STAT_COUNT {
            self.inner.iv_ranges[stat_index as usize] = StatRange::new(min, max);
        }
    }

    pub fn set_nature(&mut self, nature_index: u8) {
        self.inner.nature = Some(Nature::from_index(nature_index));
    }

    pub fn set_gender(&mut self, gender: u8) {
        self.inner.gender = Some(match gender {
            0 => Gender::Male,
            1 => Gender::Female,
            _ => Gender::Genderless,
        });
    }

    pub fn set_ability(&mut self, ability: u8) {
        self.inner.ability = Some(match ability {
            0 => AbilitySlot::One,
            1 => AbilitySlot::Two,
            _ => AbilitySlot::Hidden,
        });
    }

    pub fn set_shiny(&mut self, shiny: u8) {
        self.inner.shiny = Some(match shiny {
            0 => ShinyType::Normal,
            1 => ShinyType::Square,
            _ => ShinyType::Star,
        });
    }

    pub fn set_hidden_power_type(&mut self, hp_type: u8) {
        self.inner.hidden_power_type = Some(HiddenPowerType::from_index(hp_type));
    }

    pub fn set_hidden_power_power(&mut self, power: u8) {
        self.inner.hidden_power_power = Some(power);
    }
}

impl IndividualFilterJs {
    pub fn to_internal(&self) -> IndividualFilter {
        self.inner.clone()
    }
}

impl Default for IndividualFilterJs {
    fn default() -> Self {
        Self::new()
    }
}

pub fn derive_pending_egg(seed: u64, conditions: &GenerationConditions) -> PendingEgg {
    let (pending, _) = derive_pending_egg_with_state(seed, conditions);
    pending
}

pub fn derive_pending_egg_with_state(
    seed: u64,
    conditions: &GenerationConditions,
) -> (PendingEgg, u64) {
    debug_assert!(conditions.reroll_count <= 3);

    let mut rng = PersonalityRNG::new(seed);
    let TrainerIds { tid, sid, .. } = conditions.trainer_ids;

    let nature_idx = rng.roll_fraction(25) as u8;
    let nature = match conditions.everstone {
        EverstonePlan::None => Nature::from_index(nature_idx),
        EverstonePlan::Fixed(parent_nature) => {
            let inherit = (rng.next() >> 31) == 0;
            if inherit {
                parent_nature
            } else {
                Nature::from_index(nature_idx)
            }
        }
    };

    let ha_roll = rng.next();
    if conditions.uses_ditto {
        rng.next();
    }

    let mut inherits_vec: Vec<InheritanceSlot> = Vec::with_capacity(MAX_INHERIT_SLOTS);
    while inherits_vec.len() < MAX_INHERIT_SLOTS {
        let stat_roll = rng.roll_fraction(6) as u8;
        let stat = StatIndex::from_u8(stat_roll)
            .unwrap_or_else(|| panic!("stat roll out of range: {stat_roll}"));
        let parent_bit = rng.next() >> 31;
        if inherits_vec.iter().any(|slot| slot.stat == stat) {
            continue;
        }
        inherits_vec.push(InheritanceSlot::new(stat, ParentRole::from_bit(parent_bit)));
    }

    let inherits: [InheritanceSlot; MAX_INHERIT_SLOTS] =
        inherits_vec.try_into().expect("inherits length must be 3");

    let nidoran_roll = if conditions.has_nidoran_flag {
        Some((((rng.next() as u64).wrapping_mul(2)) >> 32) as u8)
    } else {
        None
    };

    let mut chosen_pid: Option<(u32, ShinyType)> = None;
    for attempt in 0..=(conditions.reroll_count as u32) {
        let pid = rng.roll_fraction(0xffffffff);
        let shiny_value = ShinyChecker::get_shiny_value(tid, sid, pid);
        let shiny_type = ShinyChecker::get_shiny_type(shiny_value);
        if shiny_type != ShinyType::Normal || attempt == conditions.reroll_count as u32 {
            chosen_pid = Some((pid, shiny_type));
            break;
        }
    }

    let (pid, shiny_type) = chosen_pid.expect("PID must exist");

    let gender = match nidoran_roll {
        Some(0) => Gender::Female,
        Some(_) => Gender::Male,
        None => conditions.gender_ratio.resolve_enum((pid & 0xFF) as u8),
    };

    let mut ability = AbilitySlot::from_bit(((pid >> 16) & 1) as u8);
    let ha_candidate = conditions.allow_hidden_ability
        && !conditions.uses_ditto
        && conditions.female_parent_has_hidden
        && (((ha_roll as u64).wrapping_mul(5)) >> 32) >= 2;
    if ha_candidate {
        ability = AbilitySlot::Hidden;
    }

    let pending = PendingEgg {
        inherits,
        nature,
        gender,
        ability,
        shiny: shiny_type,
        pid,
    };

    (pending, rng.current_seed())
}

pub fn resolve_egg_iv(
    pending: &PendingEgg,
    iv_sources: &IVResolutionConditions,
    lcg_seed: u64,
) -> Result<ResolvedEgg, EggIvError> {
    validate_iv_set(&iv_sources.male, true)?;
    validate_iv_set(&iv_sources.female, true)?;
    validate_iv_set(&iv_sources.rng, false)?;

    let mut resolved = iv_sources.rng;
    for slot in pending.inherits.iter() {
        let index = slot.stat.as_usize();
        resolved[index] = match slot.parent {
            ParentRole::Male => iv_sources.male[index],
            ParentRole::Female => iv_sources.female[index],
        };
    }

    let hidden_power = hidden_power_from_iv(&resolved);

    Ok(ResolvedEgg {
        lcg_seed,
        mt_seed: iv_sources.mt_seed,
        ivs: resolved,
        nature: pending.nature,
        gender: pending.gender,
        ability: pending.ability,
        shiny: pending.shiny,
        pid: pending.pid,
        hidden_power,
    })
}

pub fn matches_filter(egg: &ResolvedEgg, filter: &IndividualFilter) -> bool {
    for (idx, range) in filter.iv_ranges.iter().enumerate() {
        if !range.contains(egg.ivs[idx]) {
            return false;
        }
    }

    if let Some(expected) = filter.nature {
        if egg.nature != expected {
            return false;
        }
    }

    if let Some(expected) = filter.gender {
        if egg.gender != expected {
            return false;
        }
    }

    if let Some(expected) = filter.ability {
        if egg.ability != expected {
            return false;
        }
    }

    if let Some(expected) = filter.shiny {
        if egg.shiny != expected {
            return false;
        }
    }

    if filter.hidden_power_type.is_some() || filter.hidden_power_power.is_some() {
        match egg.hidden_power {
            HiddenPowerInfo::Unknown => return false,
            HiddenPowerInfo::Known { r#type, power } => {
                if let Some(expected_type) = filter.hidden_power_type {
                    if r#type != expected_type {
                        return false;
                    }
                }
                if let Some(expected_power) = filter.hidden_power_power {
                    if power != expected_power {
                        return false;
                    }
                }
            }
        }
    }

    true
}

pub fn hidden_power_from_iv(iv: &IvSet) -> HiddenPowerInfo {
    if iv.contains(&IV_VALUE_UNKNOWN) {
        return HiddenPowerInfo::Unknown;
    }

    let mut type_bits: u8 = 0;
    let mut power_bits: u8 = 0;

    for (idx, &value) in iv.iter().enumerate() {
        let bit_index = idx as u8;
        type_bits |= (value & 1) << bit_index;
        power_bits |= ((value >> 1) & 1) << bit_index;
    }

    let r#type = HiddenPowerType::from_index((type_bits as u16 * 15 / 63) as u8);
    let power = ((power_bits as u16 * 40 / 63) + 30) as u8;

    HiddenPowerInfo::Known { r#type, power }
}

pub fn resolve_npc_advance(seed: u64, frame_threshold: u8, slack: u8) -> (u64, u32, bool) {
    let mut rng = PersonalityRNG::new(seed);
    let mut consumed = 0u32;

    for _ in 0..INITIAL_NPC_ADVANCE_COST {
        rng.next();
    }
    consumed += INITIAL_NPC_ADVANCE_COST;

    let threshold = frame_threshold as u32;
    let slack = slack as u32;
    let mut elapsed = 0u32;
    let mut overflow: Option<u32> = None;
    let mut first_direction: Option<u32> = None;

    if overflow.is_none() {
        let roll = rng.roll_fraction(4);
        consumed += 1;
        let addition = FOUR_FRACTION_FRAMES[roll as usize];
        overflow = apply_frame_increment(&mut elapsed, threshold, addition);
    }

    if overflow.is_none() {
        let direction = rng.roll_fraction(2);
        consumed += 1;
        first_direction = Some(direction);
        let addition = if direction == 0 {
            LEFT_DIRECTION_FRAMES
        } else {
            RIGHT_DIRECTION_FRAMES
        };
        overflow = apply_frame_increment(&mut elapsed, threshold, addition);
    }

    if overflow.is_none() {
        let roll = rng.roll_fraction(4);
        consumed += 1;
        let addition = FOUR_FRACTION_FRAMES[roll as usize];
        overflow = apply_frame_increment(&mut elapsed, threshold, addition);
    }

    if overflow.is_none() {
        let direction = rng.roll_fraction(2);
        consumed += 1;
        let base_direction = first_direction.unwrap_or(direction);
        let addition = if direction == base_direction {
            0
        } else {
            DIRECTION_MISMATCH_FRAMES
        };
        overflow = apply_frame_increment(&mut elapsed, threshold, addition);
    }

    if overflow.is_none() {
        let roll = rng.roll_fraction(4);
        consumed += 1;
        let addition = FOUR_FRACTION_FRAMES[roll as usize];
        overflow = apply_frame_increment(&mut elapsed, threshold, addition);
    }

    let overflow_value = overflow.unwrap_or(0);
    let is_stable = overflow_value >= slack;

    for _ in 0..FINAL_NPC_ADVANCE_COST {
        rng.next();
    }
    consumed += FINAL_NPC_ADVANCE_COST;

    (rng.current_seed(), consumed, is_stable)
}

fn validate_iv_set(values: &IvSet, allow_unknown: bool) -> Result<(), EggIvError> {
    for &value in values.iter() {
        let limit = if allow_unknown { IV_VALUE_UNKNOWN } else { 31 };
        if value > limit {
            return Err(EggIvError::InvalidIvValue(value));
        }
    }
    Ok(())
}

fn apply_frame_increment(elapsed: &mut u32, threshold: u32, addition: u32) -> Option<u32> {
    *elapsed = elapsed.saturating_add(addition);
    if *elapsed > threshold {
        Some(elapsed.saturating_sub(threshold))
    } else {
        None
    }
}
