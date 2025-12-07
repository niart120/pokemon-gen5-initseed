use crate::egg_iv::{
    EverstonePlan, Gender, GenderRatio, GenerationConditions, IndividualFilter, TrainerIds,
};
use crate::egg_seed_enumerator::{
    derive_mt_seed, generate_rng_iv_set, EggSeedEnumerator, ParentsIVs,
};
use crate::offset_calculator::{calculate_game_offset, GameMode};
use crate::personality_rng::PersonalityRNG;
use crate::{resolve_npc_advance, IV_VALUE_UNKNOWN};

fn sample_parents() -> ParentsIVs {
    ParentsIVs {
        male: [31, 30, 29, IV_VALUE_UNKNOWN, IV_VALUE_UNKNOWN, 10],
        female: [0, 1, 2, 3, 4, 5],
    }
}

fn sample_conditions() -> GenerationConditions {
    GenerationConditions {
        has_nidoran_flag: false,
        everstone: EverstonePlan::None,
        uses_ditto: false,
        allow_hidden_ability: true,
        female_parent_has_hidden: false,
        reroll_count: 0,
        trainer_ids: TrainerIds::new(1234, 5678),
        gender_ratio: GenderRatio {
            threshold: 127,
            genderless: false,
        },
    }
}

#[test]
fn derive_mt_seed_matches_personality_rng() {
    let base_seed = 0x1234_5678_9ABC_DEF0;
    let mut rng = PersonalityRNG::new(base_seed);
    assert_eq!(derive_mt_seed(base_seed), rng.next());
}

#[test]
fn generate_rng_iv_set_matches_expected_sample() {
    let ivs = generate_rng_iv_set(0xA5A5_A5A5);
    assert_eq!(ivs, [26, 13, 14, 10, 20, 20]);
}

#[test]
fn egg_seed_enumerator_streams_results_with_advances() {
    let conditions = sample_conditions();
    let parents = sample_parents();
    let mut enumerator = EggSeedEnumerator::new(
        0x0F0F_0F0F_0F0F_0F0F,
        0,
        3,
        conditions,
        parents,
        None,
        false,
        GameMode::BwContinue,
    );

    let mut advances = Vec::new();
    for _ in 0..3 {
        let data = enumerator
            .next_egg()
            .expect("resolve succeeds")
            .expect("should produce egg");
        advances.push(data.advance);
        assert!(!data.is_stable, "NPC off should always be unstable");
    }

    assert_eq!(advances, vec![0, 1, 2]);
    assert_eq!(enumerator.remaining(), 0);
    assert!(enumerator.next_egg().unwrap().is_none());
}

#[test]
fn egg_seed_enumerator_respects_filter_and_target_count() {
    let conditions = sample_conditions();
    let parents = sample_parents();
    let mut filter = IndividualFilter::default();
    filter.gender = Some(Gender::Genderless); // impossible under current ratio

    let mut enumerator = EggSeedEnumerator::new(
        0xAAAA_BBBB_CCCC_DDDD,
        0,
        2,
        conditions,
        parents,
        Some(filter),
        false,
        GameMode::BwContinue,
    );

    assert!(enumerator.next_egg().unwrap().is_none());
    assert_eq!(enumerator.remaining(), 0);
}

#[test]
fn egg_seed_enumerator_reports_npc_stability() {
    let base_seed = 0x0101_0202_0303_0404;
    let game_mode = GameMode::BwContinue;
    let offset = calculate_game_offset(base_seed, game_mode) as u64;
    let (mul, add) = PersonalityRNG::lcg_affine_for_steps(offset);
    let seed_after_offset = PersonalityRNG::lcg_apply(base_seed, mul, add);
    let (_npc_seed, _frames, expected_stable) = resolve_npc_advance(seed_after_offset, 96, 30);

    let mut enumerator = EggSeedEnumerator::new(
        base_seed,
        0,
        1,
        sample_conditions(),
        sample_parents(),
        None,
        true,
        game_mode,
    );

    let produced = enumerator
        .next_egg()
        .expect("resolve succeeds")
        .expect("should produce egg");
    assert_eq!(produced.is_stable, expected_stable);
}

#[test]
fn egg_seed_enumerator_reports_needle_direction() {
    let base_seed = 0x0F0F_0F0F_0F0F_0F0F;
    let mut enumerator = EggSeedEnumerator::new(
        base_seed,
        0,
        1,
        sample_conditions(),
        sample_parents(),
        None,
        false,
        GameMode::BwContinue,
    );

    let produced = enumerator
        .next_egg()
        .expect("resolve succeeds")
        .expect("should produce egg");

    let expected = PersonalityRNG::calc_report_needle_direction(base_seed);
    assert_eq!(produced.report_needle_direction, expected);
}
