//! Regression contracts derived from htm.core d86999487036c36c3b242ee46b9ba9b8f566087d.
use mokosh::prelude::*;

fn pattern(size: u32, bits: &[u32]) -> Sdr {
    let mut s = Sdr::new(&[size]);
    s.set_sparse(bits).unwrap();
    s
}

#[test]
fn spatial_pooler_uses_overlaps_and_returns_canonical_sparse_output() {
    let mut sp = SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![40],
        column_dimensions: vec![20],
        potential_radius: 40,
        potential_pct: 0.8,
        local_area_density: 0.2,
        ..Default::default()
    })
    .unwrap();
    let mut output = Sdr::new(&[20]);
    let overlaps = sp.compute(&pattern(40, &[1, 3, 8, 14, 20, 27, 32]), false, &mut output);
    let winners = output.get_sparse();
    assert_eq!(winners.len(), 4);
    assert!(winners.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(
        sp.boosted_overlaps(),
        overlaps.iter().map(|&v| v as f32).collect::<Vec<_>>()
    );
    for &winner in &winners {
        for loser in 0..20 {
            if !winners.contains(&loser) {
                assert!(overlaps[winner as usize] >= overlaps[loser as usize]);
            }
        }
    }
}

#[test]
fn spatial_pooler_duty_cycle_warmup_uses_observation_count() {
    let mut sp = SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![8],
        column_dimensions: vec![8],
        potential_radius: 8,
        potential_pct: 1.0,
        local_area_density: 0.25,
        ..Default::default()
    })
    .unwrap();
    let mut output = Sdr::new(&[8]);
    sp.compute(&pattern(8, &[0, 1, 2, 3, 4, 5, 6, 7]), true, &mut output);
    assert_eq!(sp.active_duty_cycles().iter().sum::<f32>(), 2.0);
}

#[test]
fn wrapping_neighborhood_contains_each_cell_once_even_for_huge_radius() {
    let mut neighbors = Topology::neighborhood(0, &[3, 2], u32::MAX, WrappingMode::Wrap, true);
    neighbors.sort_unstable();
    assert_eq!(neighbors, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn connection_growth_honors_capacity_and_protects_active_candidates() {
    let mut c = Connections::with_cells(1);
    let s = c.create_segment(0, None);
    let protected = c.create_synapse(s, 10, 0.1);
    c.create_synapse(s, 11, 0.9);
    c.grow_synapses(
        s,
        &[10, 12, 13, 14, 15],
        0.5,
        &mut Random::new(42),
        None,
        Some(3),
    );
    assert!(c.num_synapses_on_segment(s) <= 3);
    assert_eq!(c.data_for_synapse(protected).presynaptic_cell, 10);
    assert!(c.synapses_for_segment(s).contains(&protected));
}

#[test]
fn raising_permanences_preserves_relative_weights() {
    let mut c = Connections::new(ConnectionsParams {
        num_cells: 1,
        connected_threshold: 0.5,
        timeseries: false,
    });
    let s = c.create_segment(0, None);
    let a = c.create_synapse(s, 0, 0.1);
    let b = c.create_synapse(s, 1, 0.3);
    let d = c.create_synapse(s, 2, 0.7);
    c.raise_permanences_to_threshold(s, 2);
    for (syn, expected) in [(a, 0.3), (b, 0.5), (d, 0.9)] {
        assert!((c.data_for_synapse(syn).permanence - expected).abs() < 2e-6);
    }
}

#[test]
fn scalar_encoding_matches_reference_rounding_and_disjoint_categories() {
    let e = ScalarEncoder::new(ScalarEncoderParams {
        minimum: 0.0,
        maximum: 10.0,
        size: 13,
        active_bits: 3,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(e.encode_to_sdr(0.6).unwrap().get_sparse(), vec![1, 2, 3]);
    assert_eq!(e.encode_to_sdr(f32::NAN).unwrap().get_sum(), 0);
    let categories = ScalarEncoder::new(ScalarEncoderParams {
        minimum: 0.0,
        maximum: 2.0,
        category: true,
        active_bits: 3,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(categories.size(), 9);
    assert_eq!(
        categories
            .encode_to_sdr(0.0)
            .unwrap()
            .get_overlap(&categories.encode_to_sdr(1.0).unwrap()),
        0
    );
}

#[test]
fn sdr_concatenation_honors_axis() {
    let mut left = Sdr::new(&[2, 2]);
    left.set_sparse(&[0, 3]).unwrap();
    let mut right = Sdr::new(&[2, 1]);
    right.set_sparse(&[0]).unwrap();
    let mut out = Sdr::new(&[2, 3]);
    out.concatenate(&[&left, &right], 1).unwrap();
    assert_eq!(out.get_sparse(), vec![0, 2, 4]);
    assert!(out.concatenate(&[&left, &right], 0).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn rng_round_trip_preserves_mixed_draws() {
    use mokosh::serialization::{Serializable, SerializableFormat};
    let mut rng = Random::new(42);
    rng.get_uint32();
    rng.get_real32();
    rng.get_uint64_range(1, 123);
    rng.get_bool();
    let mut restored: Random = Random::from_bytes(
        &rng.to_bytes(SerializableFormat::Binary).unwrap(),
        SerializableFormat::Binary,
    )
    .unwrap();
    for _ in 0..50 {
        assert_eq!(rng.get_uint32(), restored.get_uint32());
    }
}

#[cfg(feature = "serde")]
#[test]
fn local_pooler_round_trip_preserves_inference() {
    use mokosh::serialization::{Serializable, SerializableFormat};
    let mut sp = SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![20],
        column_dimensions: vec![20],
        global_inhibition: false,
        local_area_density: 0.2,
        ..Default::default()
    })
    .unwrap();
    let mut restored: SpatialPooler = SpatialPooler::from_bytes(
        &sp.to_bytes(SerializableFormat::Binary).unwrap(),
        SerializableFormat::Binary,
    )
    .unwrap();
    let input = pattern(20, &[1, 5, 10, 15]);
    let mut a = Sdr::new(&[20]);
    let mut b = Sdr::new(&[20]);
    sp.compute(&input, false, &mut a);
    restored.compute(&input, false, &mut b);
    assert!(a.get_sum() > 0);
    assert_eq!(a, b);
}

#[test]
fn temporal_memory_predicts_the_next_pattern_and_uses_external_context() {
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
    let cols: Vec<_> = tm.predictive_cells().iter().map(|c| c / 4).collect();
    assert_eq!(cols, vec![4, 5, 6]);
    tm.compute(&b, false);
    assert_eq!(tm.active_cells().len(), 3);
    assert_eq!(tm.anomaly(), 0.0);

    let mut external = TemporalMemory::new(TemporalMemoryParams {
        column_dimensions: vec![2],
        cells_per_column: 2,
        external_predictive_inputs: 3,
        activation_threshold: 2,
        min_threshold: 2,
        initial_permanence: 0.5,
        max_new_synapse_count: 3,
        ..Default::default()
    })
    .unwrap();
    let context = pattern(3, &[0, 1, 2]);
    let column = pattern(2, &[1]);
    external.compute_with_external(&column, true, &context, &context);
    external.reset();
    external.compute_with_external(&column, false, &context, &context);
    assert_eq!(external.active_cells().len(), 1);
    assert_eq!(external.anomaly(), 0.0);
}

#[test]
fn connection_timeseries_suppresses_repeated_updates_without_alternating() {
    let mut c = Connections::new(ConnectionsParams {
        num_cells: 1,
        connected_threshold: 0.5,
        timeseries: true,
    });
    let s = c.create_segment(0, None);
    let syn = c.create_synapse(s, 0, 0.3);
    let input = pattern(1, &[0]);
    for _ in 0..4 {
        c.compute_activity(&[0], true);
        c.adapt_segment(s, &input, 0.1, 0.1, false, 0);
    }
    assert!((c.data_for_synapse(syn).permanence - 0.4).abs() < 1e-6);
    c.compute_activity(&[0], true);
    c.adapt_segment(s, &input, 0.2, 0.1, false, 0);
    assert!((c.data_for_synapse(syn).permanence - 0.6).abs() < 1e-6);
}

#[test]
fn rdse_negative_values_do_not_collapse_to_zero() {
    let e = RandomDistributedScalarEncoder::new(RdseParams {
        size: 1000,
        active_bits: 40,
        resolution: 1.0,
        seed: 42,
        ..Default::default()
    })
    .unwrap();
    assert_ne!(
        e.encode_to_sdr(-10.0).unwrap(),
        e.encode_to_sdr(-20.0).unwrap()
    );
    assert!(
        e.encode_to_sdr(-1.0)
            .unwrap()
            .get_overlap(&e.encode_to_sdr(0.0).unwrap())
            >= 35
    );
}

#[cfg(feature = "serde")]
#[test]
fn invalid_sdr_payloads_return_errors() {
    use mokosh::serialization::Serializable;
    for json in [
        r#"{"dimensions":[],"sparse":[]}"#,
        r#"{"dimensions":[8],"sparse":[8]}"#,
        r#"{"dimensions":[8],"sparse":[3,1]}"#,
        r#"{"dimensions":[4294967295,2],"sparse":[]}"#,
    ] {
        assert!(Sdr::from_json(json).is_err());
    }
}

#[test]
fn classifier_respects_record_gaps_and_untrained_horizons() {
    let p = pattern(8, &[0, 2, 4]);
    let mut c = SdrClassifier::new(SdrClassifierParams {
        alpha: 0.1,
        steps: vec![0, 1, 3],
    });
    assert!(c.infer(&p).values().all(Vec::is_empty));
    c.learn_at(0, &p, &[0]);
    c.learn_at(2, &p, &[1]);
    let prediction = c.infer(&p);
    assert_eq!(prediction[&0].len(), 2);
    assert!(prediction[&1].is_empty());
    assert!(prediction[&3].is_empty());
    c.learn_at(3, &p, &[0, 1]);
    assert_eq!(c.infer(&p)[&1], vec![0.5, 0.5]);
    c.reset();
    assert_eq!(c.infer_single(&pattern(8, &[]), 0), Some(0));
}

#[test]
fn datetime_negative_epoch_and_leap_centuries_round_trip() {
    use mokosh::encoders::DateTime;
    let d = DateTime::from_timestamp(-1);
    assert_eq!(
        (d.year, d.month, d.day, d.hour, d.minute, d.second),
        (1969, 12, 31, 23, 59, 59)
    );
    for ts in [
        -2203977601,
        -86401,
        -86400,
        -1,
        0,
        951782400,
        4107542400,
        253402300799,
    ] {
        assert_eq!(DateTime::from_timestamp(ts).to_timestamp(), ts);
    }
}

#[test]
fn anomaly_modes_use_distinct_likelihood_and_finite_log_scale() {
    use mokosh::algorithms::AnomalyLikelihood;
    let mut likelihood = AnomalyLikelihood::with_window(2, 3, 20, 2, 1);
    for _ in 0..5 {
        assert_eq!(likelihood.anomaly_probability(0.1), 0.5);
    }
    assert!((likelihood.anomaly_probability(0.1) - 0.5).abs() < 1e-6);
    assert!(likelihood.anomaly_probability(0.9) > 0.99);
    for p in [0.0, 0.5, 0.99, 1.0] {
        let log = AnomalyLikelihood::compute_log_likelihood(p);
        assert!(log.is_finite() && (0.0..=1.0).contains(&log));
    }
    let mut tm = TemporalMemory::new(TemporalMemoryParams {
        column_dimensions: vec![4],
        anomaly_mode: AnomalyMode::Likelihood,
        ..Default::default()
    })
    .unwrap();
    tm.compute(&pattern(4, &[0]), false);
    assert_eq!(tm.anomaly(), 0.5);
}

#[cfg(feature = "serde")]
#[test]
fn serialization_propagates_buffer_flush_failure() {
    use mokosh::serialization::{Serializable, SerializableFormat};
    struct FailingWriter;
    impl std::io::Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "write failed",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert!(pattern(4, &[1])
        .save(FailingWriter, SerializableFormat::Binary)
        .is_err());
}

#[test]
#[should_panic(expected = "assertion")]
fn simd_safe_api_rejects_mismatched_lengths_in_release_too() {
    mokosh::utils::simd::exp_batch(&[1.0; 16], &mut [0.0; 1]);
}

#[test]
fn reference_connection_threshold_includes_epsilon_boundary() {
    let mut c = Connections::with_cells(1);
    let s = c.create_segment(0, None);
    c.create_synapse(s, 10, 0.5 - 0.5e-6);
    assert_eq!(c.compute_activity(&[10], false)[s as usize], 1);
}

#[test]
fn periodic_scalar_rounds_across_the_wrap_boundary() {
    let e = ScalarEncoder::new(ScalarEncoderParams {
        minimum: 0.0,
        maximum: 10.0,
        size: 10,
        active_bits: 3,
        periodic: true,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(e.encode_to_sdr(9.9).unwrap(), e.encode_to_sdr(0.0).unwrap());
    assert_eq!(
        e.encode_to_sdr(-0.1).unwrap(),
        e.encode_to_sdr(0.0).unwrap()
    );
}

#[cfg(feature = "serde")]
#[test]
fn restored_models_continue_learning_identically() {
    use mokosh::serialization::{Serializable, SerializableFormat};
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
    for _ in 0..3 {
        tm.compute(&pattern(8, &[0, 1, 2]), true);
        tm.compute(&pattern(8, &[4, 5, 6]), true);
    }
    for format in [SerializableFormat::Binary, SerializableFormat::Json] {
        let mut original = tm.clone();
        let mut restored: TemporalMemory =
            TemporalMemory::from_bytes(&tm.to_bytes(format).unwrap(), format).unwrap();
        for bits in [vec![0, 1, 2], vec![4, 5, 6], vec![1, 3, 7], vec![0, 1, 2]] {
            let input = pattern(8, &bits);
            original.compute(&input, true);
            restored.compute(&input, true);
            assert_eq!(original.active_cells(), restored.active_cells());
            assert_eq!(original.predictive_cells(), restored.predictive_cells());
            assert_eq!(original.winner_cells(), restored.winner_cells());
            assert_eq!(original.anomaly(), restored.anomaly());
            assert_eq!(original.connections(), restored.connections());
        }
    }
}

#[test]
fn invalid_parameters_fail_before_model_allocation() {
    for dimensions in [vec![0], vec![u32::MAX, 2]] {
        assert!(TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: dimensions.clone(),
            ..Default::default()
        })
        .is_err());
        assert!(SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: dimensions.clone(),
            column_dimensions: dimensions,
            ..Default::default()
        })
        .is_err());
    }
    assert!(TemporalMemory::new(TemporalMemoryParams {
        max_segments_per_cell: 0,
        ..Default::default()
    })
    .is_err());
    assert!(TemporalMemory::new(TemporalMemoryParams {
        initial_permanence: f32::NAN,
        ..Default::default()
    })
    .is_err());
    assert!(SpatialPooler::new(SpatialPoolerParams {
        duty_cycle_period: 0,
        ..Default::default()
    })
    .is_err());
    assert!(SpatialPooler::new(SpatialPoolerParams {
        potential_pct: f32::NAN,
        ..Default::default()
    })
    .is_err());
    assert!(SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![100000],
        potential_radius: 100000,
        potential_pct: 1.0,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn connection_competition_uniformly_shifts_permanences() {
    for (values, min, max, expected) in [
        ([0.1, 0.2, 0.6], 2, 2, [0.4, 0.5, 0.9]),
        ([0.6, 0.7, 0.8], 1, 2, [0.4, 0.5, 0.6]),
    ] {
        let mut c = Connections::with_cells(1);
        let s = c.create_segment(0, None);
        let synapses: Vec<_> = values
            .into_iter()
            .enumerate()
            .map(|(i, p)| c.create_synapse(s, i as u32, p))
            .collect();
        c.synapse_competition(s, min, max);
        for (syn, value) in synapses.into_iter().zip(expected) {
            assert!((c.data_for_synapse(syn).permanence - value).abs() < 1e-6);
        }
    }
}

#[test]
fn segment_eviction_uses_strength_instead_of_synapse_count() {
    let mut c = Connections::with_cells(1);
    let weak = c.create_segment(0, None);
    for i in 0..3 {
        c.create_synapse(weak, i, 0.1);
    }
    let strong = c.create_segment(0, None);
    c.create_synapse(strong, 10, 0.9);
    c.create_segment(0, Some(2));
    assert!(c.segments_for_cell(0).contains(&strong));
    assert_eq!(c.num_synapses(), 1);
}

#[test]
fn duplicate_coordinates_are_canonical_across_all_representations() {
    let mut s = Sdr::new(&[3, 3]);
    s.set_coordinates(&vec![vec![2, 0, 2], vec![1, 0, 1]])
        .unwrap();
    assert_eq!(s.get_sparse(), vec![0, 7]);
    assert_eq!(s.get_coordinates(), vec![vec![0, 2], vec![0, 1]]);
    assert_eq!(
        s.get_dense().iter().map(|&n| n as usize).sum::<usize>(),
        s.get_sum()
    );
}

#[test]
fn duplicate_destruction_does_not_corrupt_free_lists() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut c = Connections::with_cells(1);
    let segment = c.create_segment(0, None);
    let synapse = c.create_synapse(segment, 10, 0.5);
    c.destroy_synapse(synapse);
    assert!(catch_unwind(AssertUnwindSafe(|| c.destroy_synapse(synapse))).is_err());
    assert_eq!(c.num_synapses(), 0);
    c.create_synapse(segment, 11, 0.5);
    c.destroy_segment(segment);
    assert!(catch_unwind(AssertUnwindSafe(|| c.destroy_segment(segment))).is_err());
    assert_eq!(c.num_segments(), 0);
    assert_eq!(c.num_synapses(), 0);
}
