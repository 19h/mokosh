use mokosh::encoders::{ScalarEncoder64, ScalarEncoderParams64};
use mokosh::prelude::*;
use mokosh::utils::Metrics;
fn expected(key: &str) -> Vec<f64> {
    include_str!("reference/followup_d869994.txt")
        .lines()
        .find(|line| line.split_whitespace().next() == Some(key))
        .unwrap()
        .split_whitespace()
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect()
}
fn compare(key: &str, values: &[f32]) {
    let expected = expected(key);
    assert_eq!(values.len(), expected.len());
    for (&a, b) in values.iter().zip(expected) {
        assert!((f64::from(a) - b).abs() < 1e-6, "{key}: {a} != {b}");
    }
}
#[test]
fn double_precision_encoders_match_reference_without_f32_narrowing() {
    let e = ScalarEncoder64::new(ScalarEncoderParams64 {
        minimum: 1e9,
        maximum: 1e9 + 1.0,
        size: 1024,
        active_bits: 21,
        ..Default::default()
    })
    .unwrap();
    let r = RandomDistributedScalarEncoder::new(RdseParams {
        size: 1000,
        active_bits: 20,
        resolution: 0.001,
        seed: 42,
        ..Default::default()
    })
    .unwrap();
    for i in 0..5 {
        let s = e.encode_to_sdr(1e9 + i as f64 * 0.001).unwrap();
        assert_eq!(
            s.get_sparse(),
            expected(&format!("scalar64.{i}"))
                .into_iter()
                .map(|x| x as u32)
                .collect::<Vec<_>>()
        );
        let s = r.encode_f64_to_sdr(1e6 + i as f64 * 0.001).unwrap();
        assert_eq!(
            s.get_sparse(),
            expected(&format!("rdse64.{i}"))
                .into_iter()
                .map(|x| x as u32)
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn streaming_metrics_match_reference_and_attach_lifecycle() {
    let mut data = Sdr::new(&[8]);
    let mut metrics = Metrics::new(&[8], 3);
    let attached = Metrics::attach(&data, 3);
    for i in 0..10 {
        data.set_sparse(&(0..8).filter(|b| (b + i) % 3 == 0).collect::<Vec<_>>())
            .unwrap();
        metrics.add_data(&data);
        let s = &metrics.sparsity;
        compare(
            &format!("metrics.sparsity.{i}"),
            &[s.sparsity(), s.min(), s.max(), s.mean(), s.std()],
        );
        compare(
            &format!("metrics.frequency.{i}"),
            metrics.activation_frequency.frequencies(),
        );
        compare(
            &format!("metrics.entropy.{i}"),
            &[metrics.activation_frequency.entropy()],
        );
        if i > 0 {
            let s = &metrics.overlap;
            compare(
                &format!("metrics.overlap.{i}"),
                &[s.overlap(), s.min(), s.max(), s.mean(), s.std()],
            );
        }
        assert_eq!(
            attached.snapshot().activation_frequency.frequencies(),
            metrics.activation_frequency.frequencies()
        );
    }
    drop(attached);
    data.zero();
}
#[cfg(feature = "serde")]
#[test]
fn impossible_rng_state_and_unbounded_replay_are_rejected() {
    use mokosh::serialization::Serializable;
    let rng = Random::new(42);
    let mut state = serde_json::to_value(rng).unwrap();
    state["index"] = serde_json::json!(900);
    assert!(serde_json::from_value::<Random>(state).is_err());
    assert!(Random::from_json(
        r#"{"algorithm":1297362993,"seed":42,"steps":18446744073709551615}"#
    )
    .is_err());
}

#[test]
fn callback_registration_is_removed_on_drop() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let mut sdr = Sdr::new(&[8]);
    let calls = Arc::new(AtomicUsize::new(0));
    let observer = calls.clone();
    let subscription = sdr.subscribe(move |_| {
        observer.fetch_add(1, Ordering::SeqCst);
    });
    sdr.set_sparse(&[1]).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(subscription);
    sdr.zero();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_boost_updates_are_bit_identical() {
    let params = SpatialPoolerParams {
        input_dimensions: vec![16],
        column_dimensions: vec![4096],
        potential_radius: 4,
        boost_strength: 2.0,
        ..Default::default()
    };
    let mut first = SpatialPooler::new(params.clone()).unwrap();
    let mut second = SpatialPooler::new(params).unwrap();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for step in 0..5 {
        let mut input = Sdr::new(&[16]);
        input
            .set_sparse(&(0..16).filter(|i| (i + step) % 3 == 0).collect::<Vec<_>>())
            .unwrap();
        let input_copy = input.clone();
        let (returned, a) = one.install(move || {
            let mut output = Sdr::new(&[4096]);
            first.compute(&input, true, &mut output);
            (first, output)
        });
        first = returned;
        let (returned, b) = four.install(move || {
            let mut output = Sdr::new(&[4096]);
            second.compute(&input_copy, true, &mut output);
            (second, output)
        });
        second = returned;
        assert_eq!(a, b);
        assert_eq!(first.boost_factors(), second.boost_factors());
    }
}
