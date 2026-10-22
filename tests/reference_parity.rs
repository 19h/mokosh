//! Golden vectors emitted by the compiled, pinned C++ reference.
use mokosh::encoders::{SimHashDocumentEncoder, SimHashDocumentEncoderParams};
use mokosh::prelude::*;

fn expected(key: &str) -> Vec<f64> {
    let line = include_str!("reference/htm_core_d869994.txt")
        .lines()
        .find(|line| line.split_whitespace().next() == Some(key))
        .expect("Missing reference vector");
    line.split_whitespace()
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect()
}
fn check_indices(key: &str, actual: &[u32]) {
    let mut reference: Vec<_> = expected(key).into_iter().map(|n| n as u32).collect();
    reference.sort_unstable();
    assert_eq!(actual, reference, "{key}");
}
fn pattern(size: u32, bits: &[u32]) -> Sdr {
    let mut s = Sdr::new(&[size]);
    s.set_sparse(bits).unwrap();
    s
}

#[test]
fn reference_mt19937_words_and_reals() {
    let mut rng = Random::new(42);
    for value in expected("rng.words") {
        assert_eq!(rng.get_uint32(), value as u32);
    }
    let mut rng = Random::new(42);
    for value in expected("rng.reals") {
        assert_eq!(rng.get_real64(), value);
    }
}

#[test]
fn reference_spatial_pooler_global_and_local_learning() {
    for global in [true, false] {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![40],
            column_dimensions: vec![20],
            potential_radius: 40,
            potential_pct: 0.8,
            global_inhibition: global,
            local_area_density: 0.2,
            ..Default::default()
        })
        .unwrap();
        let input = pattern(40, &[1, 3, 8, 14, 20, 27, 32]);
        let mut output = Sdr::new(&[20]);
        for step in 0..5 {
            let prefix = format!("sp.{}.{step}", if global { "global" } else { "local" });
            let overlaps = sp.compute(&input, true, &mut output);
            assert_eq!(
                overlaps.iter().map(|&n| n as f64).collect::<Vec<_>>(),
                expected(&format!("{prefix}.overlaps")),
                "{prefix}"
            );
            check_indices(&format!("{prefix}.active"), &output.get_sparse());
            for (&actual, reference) in sp
                .active_duty_cycles()
                .iter()
                .zip(expected(&format!("{prefix}.duty")))
            {
                assert!(
                    (f64::from(actual) - reference).abs() < 1e-7,
                    "{prefix}: {actual} != {reference}"
                );
            }
        }
    }
}

#[test]
fn reference_scalar_rdse_and_shake256() {
    let scalar = ScalarEncoder::new(ScalarEncoderParams {
        minimum: 0.0,
        maximum: 10.0,
        size: 13,
        active_bits: 3,
        ..Default::default()
    })
    .unwrap();
    for value in [0.0_f32, 0.6, 1.4, 5.0, 9.6, 10.0] {
        check_indices(
            &format!("scalar.{value:.6}"),
            &scalar.encode_to_sdr(value).unwrap().get_sparse(),
        );
    }
    let rdse = RandomDistributedScalarEncoder::new(RdseParams {
        size: 1000,
        active_bits: 40,
        resolution: 1.0,
        seed: 42,
        ..Default::default()
    })
    .unwrap();
    for value in [0.0_f32, 1.0, 10.0, 20.0] {
        check_indices(
            &format!("rdse.{value:.6}"),
            &rdse.encode_to_sdr(value).unwrap().get_sparse(),
        );
    }
    let hash = SimHashDocumentEncoder::new(SimHashDocumentEncoderParams::default()).unwrap();
    check_indices(
        "simhash.hello",
        &hash
            .encode_to_sdr(vec!["hello".to_string()])
            .unwrap()
            .get_sparse(),
    );
}

#[test]
fn reference_temporal_memory_sequence() {
    let mut tm = TemporalMemory::new(TemporalMemoryParams {
        column_dimensions: vec![8],
        cells_per_column: 4,
        activation_threshold: 2,
        min_threshold: 2,
        initial_permanence: 0.5,
        max_new_synapse_count: 3,
        ..Default::default()
    })
    .unwrap();
    let a = pattern(8, &[0, 1, 2]);
    let b = pattern(8, &[4, 5, 6]);
    for _ in 0..8 {
        tm.reset();
        tm.compute(&a, true);
        tm.compute(&b, true);
    }
    tm.reset();
    tm.compute(&a, false);
    check_indices("tm.predicted", tm.predictive_cells());
    check_indices("tm.winners", tm.winner_cells());
    tm.compute(&b, false);
    check_indices("tm.active", tm.active_cells());
    assert_eq!(f64::from(tm.anomaly()), expected("tm.anomaly")[0]);
}

#[test]
fn reference_classifier_with_missing_records() {
    let mut c = SdrClassifier::new(SdrClassifierParams {
        alpha: 0.1,
        steps: vec![0, 1, 3],
    });
    let p = pattern(8, &[0, 1, 2]);
    for i in 0..8 {
        c.learn_at(2 * i, &p, &[(i % 2) as u32]);
    }
    for (step, probabilities) in c.infer(&p) {
        let reference = expected(&format!("classifier.{step}"));
        assert_eq!(probabilities.len(), reference.len());
        for (a, b) in probabilities.into_iter().zip(reference) {
            assert!((f64::from(a) - b).abs() < 1e-6);
        }
    }
}

#[test]
fn reference_multidimensional_local_inhibition_boost_and_update_round() {
    let mut sp = SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![8, 4],
        column_dimensions: vec![4, 8],
        potential_radius: 1,
        potential_pct: 1.0,
        global_inhibition: false,
        local_area_density: 0.25,
        stimulus_threshold: 2,
        boost_strength: 2.0,
        wrap_around: true,
        ..Default::default()
    })
    .unwrap();
    let mut input = Sdr::new(&[8, 4]);
    let mut output = Sdr::new(&[4, 8]);
    for step in 0..60 {
        input
            .set_sparse(&(0..32).filter(|b| (b + step) % 3 == 0).collect::<Vec<_>>())
            .unwrap();
        let overlaps = sp.compute(&input, true, &mut output);
        if [0, 1, 49, 50, 59].contains(&step) {
            let prefix = format!("sp.geometry.{step}");
            assert_eq!(
                overlaps.iter().map(|&n| n as f64).collect::<Vec<_>>(),
                expected(&format!("{prefix}.overlaps")),
                "{prefix}"
            );
            check_indices(&format!("{prefix}.active"), &output.get_sparse());
            assert_eq!(
                sp.inhibition_radius() as f64,
                expected(&format!("{prefix}.radius"))[0],
                "{prefix}"
            );
            for (&actual, reference) in sp
                .boost_factors()
                .iter()
                .zip(expected(&format!("{prefix}.boost")))
            {
                assert!(
                    (f64::from(actual) - reference).abs() < 1e-6,
                    "{prefix}: {actual} != {reference}"
                );
            }
        }
    }
}

#[test]
fn nonwrapping_reference_duplicate_is_not_reproduced() {
    // The reference iterator skips the final center by advancing beyond end,
    // then dereferences the reset iterator, duplicating neighbor 22.
    let reference: Vec<_> = expected("reference_bug.nonwrapping_neighbors")
        .into_iter()
        .map(|n| n as usize)
        .collect();
    assert_eq!(reference, vec![22, 22, 23, 30]);
    let mut actual = Topology::neighborhood(31, &[4, 8], 1, WrappingMode::NoWrap, false);
    actual.sort_unstable();
    assert_eq!(actual, vec![22, 23, 30]);
}

#[test]
fn reference_date_encoding_in_utc() {
    use mokosh::encoders::{DateEncoder, DateEncoderParams};
    let encoder = DateEncoder::new(DateEncoderParams {
        season_width: 3,
        day_of_week_width: 3,
        weekend_width: 3,
        holiday_width: 3,
        time_of_day_width: 3,
        custom_width: 3,
        custom_days: vec!["mon".into(), "fri".into()],
        ..Default::default()
    })
    .unwrap();
    for timestamp in [-1_i64, 0, 1577836800, 1608897600, 1640973600] {
        check_indices(
            &format!("date.{timestamp}"),
            &encoder.encode_to_sdr(timestamp).unwrap().get_sparse(),
        );
    }
}
