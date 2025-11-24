use crate::pid_shiny_checker::ShinyType;
use crate::{
    derive_pending_egg, hidden_power_from_iv, matches_filter, resolve_egg_iv, resolve_npc_advance,
    AbilitySlot, EverstonePlan, Gender, GenderRatio, GenerationConditions, HiddenPowerInfo,
    HiddenPowerType, IVResolutionConditions, IndividualFilter, InheritanceSlot, IvSet, Nature,
    ParentRole, PendingEgg, PersonalityRNG, ResolvedEgg, StatIndex, StatRange, TrainerIds,
    IV_VALUE_UNKNOWN,
};
use std::collections::HashSet;

fn make_pending_egg() -> PendingEgg {
    PendingEgg {
        inherits: [
            InheritanceSlot::new(StatIndex::Attack, ParentRole::Male),
            InheritanceSlot::new(StatIndex::Defense, ParentRole::Female),
            InheritanceSlot::new(StatIndex::Speed, ParentRole::Male),
        ],
        nature: Nature::Hardy,
        gender: Gender::Male,
        ability: AbilitySlot::One,
        shiny: ShinyType::Normal,
        pid: 0,
    }
}

#[test]
fn resolve_egg_iv_propagates_unknown_values() {
    let pending = make_pending_egg();
    let male: IvSet = [
        IV_VALUE_UNKNOWN,
        IV_VALUE_UNKNOWN,
        10,
        11,
        12,
        IV_VALUE_UNKNOWN,
    ];
    let female: IvSet = [0, 1, IV_VALUE_UNKNOWN, 3, 4, 5];
    let rng: IvSet = [1, 2, 3, 4, 5, 6];

    let resolved = resolve_egg_iv(&pending, &IVResolutionConditions { male, female, rng })
        .expect("resolve succeeds");

    assert_eq!(resolved.ivs[StatIndex::Hp.as_usize()], 1);
    assert_eq!(resolved.ivs[StatIndex::Attack.as_usize()], IV_VALUE_UNKNOWN);
    assert_eq!(
        resolved.ivs[StatIndex::Defense.as_usize()],
        IV_VALUE_UNKNOWN
    );
    assert_eq!(resolved.ivs[StatIndex::Speed.as_usize()], IV_VALUE_UNKNOWN);
}

#[test]
fn matches_filter_handles_unknown_ranges() {
    let pending = make_pending_egg();
    let male: IvSet = [IV_VALUE_UNKNOWN; 6];
    let female: IvSet = [IV_VALUE_UNKNOWN; 6];
    let rng: IvSet = [10, 11, 12, 13, 14, 15];

    let resolved = resolve_egg_iv(&pending, &IVResolutionConditions { male, female, rng })
        .expect("resolve succeeds");

    let mut loose_filter = IndividualFilter::default();
    loose_filter.iv_ranges[StatIndex::Attack.as_usize()] = StatRange {
        min: 0,
        max: IV_VALUE_UNKNOWN,
    };
    assert!(matches_filter(&resolved, &loose_filter));

    let mut strict_filter = IndividualFilter::default();
    strict_filter.iv_ranges[StatIndex::Attack.as_usize()] = StatRange { min: 0, max: 31 };
    assert!(!matches_filter(&resolved, &strict_filter));
}

#[test]
fn hidden_power_computation_and_filters() {
    let maxed: IvSet = [31; 6];
    let info = hidden_power_from_iv(&maxed);
    match info {
        HiddenPowerInfo::Known { r#type, power } => {
            assert_eq!(r#type, HiddenPowerType::Dark);
            assert_eq!(power, 70);
        }
        HiddenPowerInfo::Unknown => panic!("should resolve"),
    }

    let resolved_known = ResolvedEgg {
        ivs: maxed,
        nature: Nature::Hardy,
        gender: Gender::Male,
        ability: AbilitySlot::One,
        shiny: ShinyType::Normal,
        pid: 0,
    };

    let mut filter = IndividualFilter::default();
    filter.hidden_power_type = Some(HiddenPowerType::Dark);
    filter.hidden_power_power = Some(70);
    assert!(matches_filter(&resolved_known, &filter));

    filter.hidden_power_power = Some(60);
    assert!(!matches_filter(&resolved_known, &filter));

    let mut filter_requires_type = IndividualFilter::default();
    filter_requires_type.hidden_power_type = Some(HiddenPowerType::Fire);
    assert!(!matches_filter(&resolved_known, &filter_requires_type));

    let unresolved: IvSet = [IV_VALUE_UNKNOWN, 31, 31, 31, 31, 31];
    assert!(matches!(
        hidden_power_from_iv(&unresolved),
        HiddenPowerInfo::Unknown
    ));

    let resolved_unknown = ResolvedEgg {
        ivs: unresolved,
        nature: Nature::Hardy,
        gender: Gender::Female,
        ability: AbilitySlot::Hidden,
        shiny: ShinyType::Normal,
        pid: 0,
    };

    assert!(!matches_filter(&resolved_unknown, &filter));
}

#[test]
fn derive_pending_egg_respects_generation_flow() {
    let conditions = GenerationConditions {
        has_nidoran_flag: false,
        everstone: EverstonePlan::Fixed(Nature::Modest),
        uses_ditto: false,
        allow_hidden_ability: true,
        female_parent_has_hidden: true,
        reroll_count: 2,
        trainer_ids: TrainerIds::new(123, 456),
        gender_ratio: GenderRatio {
            threshold: 127,
            genderless: false,
        },
    };

    let pending = derive_pending_egg(0x1234567812345678, &conditions);

    assert_eq!(pending.inherits.len(), 3);
    let unique_stats: HashSet<_> = pending
        .inherits
        .iter()
        .map(|slot| slot.stat as u8)
        .collect();
    assert_eq!(unique_stats.len(), 3);
}

#[test]
fn resolve_npc_advance_matches_reference() {
    let cases = [
        (0x1234_5678_1234_5678, 60, 10),
        (0x0FF0_AA55_CCDD_EE22, 180, 5),
        (0x8855_3311_AA44_CC77, 220, 30),
    ];

    for (seed, threshold, slack) in cases {
        let actual = resolve_npc_advance(seed, threshold, slack);
        let reference = resolve_npc_advance_reference(seed, threshold, slack);
        assert_eq!(
            actual, reference,
            "resolve_npc_advance mismatch for seed {seed:016X}"
        );
    }
}

#[test]
fn resolve_npc_advance_handles_immediate_threshold() {
    let seed = 0x1111_2222_3333_4444;
    let (advanced_seed, consumed, is_stable) = resolve_npc_advance(seed, 32, 0);

    assert!(is_stable, "overflow should satisfy zero slack");
    assert_eq!(consumed, 6, "expected minimal consumption (3 + 1 + 2)");

    let mut rng = PersonalityRNG::new(seed);
    for _ in 0..consumed {
        rng.next();
    }
    assert_eq!(
        advanced_seed,
        rng.current_seed(),
        "seed should match consumed advances"
    );
}

fn resolve_npc_advance_reference(seed: u64, frame_threshold: u8, slack: u8) -> (u64, u32, bool) {
    #[derive(Clone, Copy)]
    enum Step {
        FourFraction,
        DirectionPrimary,
        FourFractionRepeat,
        DirectionDifference,
        FourFractionFinal,
    }

    let mut rng = PersonalityRNG::new(seed);
    let mut consumed = 0u32;
    for _ in 0..3 {
        rng.next();
        consumed += 1;
    }

    let threshold = frame_threshold as u32;
    let slack = slack as u32;
    let mut elapsed = 0u32;
    let mut overflow: Option<u32> = None;
    let mut primary_direction: Option<u32> = None;
    let steps = [
        Step::FourFraction,
        Step::DirectionPrimary,
        Step::FourFractionRepeat,
        Step::DirectionDifference,
        Step::FourFractionFinal,
    ];

    for step in steps {
        if overflow.is_some() {
            break;
        }
        match step {
            Step::FourFraction | Step::FourFractionRepeat | Step::FourFractionFinal => {
                let roll = rng.roll_fraction(4);
                consumed += 1;
                let addition = [32u32, 48, 96, 128][roll as usize];
                overflow = update_elapsed(&mut elapsed, threshold, addition);
            }
            Step::DirectionPrimary => {
                let direction = rng.roll_fraction(2);
                consumed += 1;
                primary_direction = Some(direction);
                let addition = if direction == 0 { 20 } else { 16 };
                overflow = update_elapsed(&mut elapsed, threshold, addition);
            }
            Step::DirectionDifference => {
                let direction = rng.roll_fraction(2);
                consumed += 1;
                let base = primary_direction.unwrap_or(direction);
                let addition = if direction == base { 0 } else { 20 };
                overflow = update_elapsed(&mut elapsed, threshold, addition);
            }
        }
    }

    let overflow = overflow.unwrap_or(0);

    for _ in 0..2 {
        rng.next();
        consumed += 1;
    }

    (rng.current_seed(), consumed, overflow >= slack)
}

fn update_elapsed(elapsed: &mut u32, threshold: u32, addition: u32) -> Option<u32> {
    *elapsed = elapsed.saturating_add(addition);
    if *elapsed >= threshold {
        Some(elapsed.saturating_sub(threshold))
    } else {
        None
    }
}
