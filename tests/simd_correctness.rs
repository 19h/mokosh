//! Comprehensive correctness tests for SIMD optimization candidates.
//!
//! These tests establish correctness invariants that must hold before and after
//! any SIMD optimization. They use property-based testing to exhaustively verify
//! edge cases and mathematical properties.
//!
//! Run with: `cargo test --test simd_correctness`
//! Run with verbose output: `cargo test --test simd_correctness -- --nocapture`

use mokosh::prelude::*;
use mokosh::utils::Random;
use proptest::prelude::*;
use std::collections::HashSet;

// =============================================================================
// SDR OPERATION CORRECTNESS TESTS
// =============================================================================

mod sdr_operations {
    use super::*;

    /// Helper to create an SDR with specific sparse indices
    fn make_sdr(size: u32, indices: &[u32]) -> Sdr {
        let mut sdr = Sdr::new(&[size]);
        if !indices.is_empty() {
            let mut sorted = indices.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            // Filter out-of-bounds
            let valid: Vec<u32> = sorted.into_iter().filter(|&i| i < size).collect();
            if !valid.is_empty() {
                sdr.set_sparse(&valid).unwrap();
            }
        }
        sdr
    }

    // -------------------------------------------------------------------------
    // Overlap Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_overlap_empty_sdrs() {
        let sdr1 = Sdr::new(&[100]);
        let sdr2 = Sdr::new(&[100]);
        assert_eq!(sdr1.get_overlap(&sdr2), 0);
    }

    #[test]
    fn test_overlap_one_empty() {
        let sdr1 = make_sdr(100, &[1, 5, 10]);
        let sdr2 = Sdr::new(&[100]);
        assert_eq!(sdr1.get_overlap(&sdr2), 0);
        assert_eq!(sdr2.get_overlap(&sdr1), 0);
    }

    #[test]
    fn test_overlap_identical() {
        let sdr1 = make_sdr(100, &[1, 5, 10, 20, 50]);
        let sdr2 = make_sdr(100, &[1, 5, 10, 20, 50]);
        assert_eq!(sdr1.get_overlap(&sdr2), 5);
    }

    #[test]
    fn test_overlap_disjoint() {
        let sdr1 = make_sdr(100, &[0, 1, 2]);
        let sdr2 = make_sdr(100, &[50, 51, 52]);
        assert_eq!(sdr1.get_overlap(&sdr2), 0);
    }

    #[test]
    fn test_overlap_partial() {
        let sdr1 = make_sdr(100, &[1, 2, 3, 4, 5]);
        let sdr2 = make_sdr(100, &[3, 4, 5, 6, 7]);
        assert_eq!(sdr1.get_overlap(&sdr2), 3);
    }

    #[test]
    fn test_overlap_commutative() {
        let sdr1 = make_sdr(100, &[1, 5, 10, 20]);
        let sdr2 = make_sdr(100, &[5, 10, 30, 40]);
        assert_eq!(sdr1.get_overlap(&sdr2), sdr2.get_overlap(&sdr1));
    }

    #[test]
    fn test_overlap_subset() {
        let sdr1 = make_sdr(100, &[1, 2, 3, 4, 5]);
        let sdr2 = make_sdr(100, &[2, 3, 4]);
        assert_eq!(sdr1.get_overlap(&sdr2), 3);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_overlap_equals_set_intersection(
            indices1 in proptest::collection::vec(0u32..1000, 0..100),
            indices2 in proptest::collection::vec(0u32..1000, 0..100)
        ) {
            let sdr1 = make_sdr(1000, &indices1);
            let sdr2 = make_sdr(1000, &indices2);

            let overlap = sdr1.get_overlap(&sdr2);

            // Reference implementation using HashSet
            let set1: HashSet<u32> = sdr1.get_sparse().into_iter().collect();
            let set2: HashSet<u32> = sdr2.get_sparse().into_iter().collect();
            let expected = set1.intersection(&set2).count();

            prop_assert_eq!(overlap, expected);
        }

        #[test]
        fn prop_overlap_is_commutative(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);

            prop_assert_eq!(sdr1.get_overlap(&sdr2), sdr2.get_overlap(&sdr1));
        }

        #[test]
        fn prop_overlap_bounded_by_smaller_sdr(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);

            let overlap = sdr1.get_overlap(&sdr2);
            let min_size = sdr1.get_sum().min(sdr2.get_sum());

            prop_assert!(overlap <= min_size);
        }

        #[test]
        fn prop_self_overlap_equals_sum(
            indices in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr = make_sdr(500, &indices);
            prop_assert_eq!(sdr.get_overlap(&sdr), sdr.get_sum());
        }
    }

    // -------------------------------------------------------------------------
    // Intersection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_intersection_empty() {
        let sdr1 = make_sdr(100, &[1, 2, 3]);
        let sdr2 = Sdr::new(&[100]);
        let mut result = Sdr::new(&[100]);

        result.intersection(&sdr1, &sdr2).unwrap();
        assert_eq!(result.get_sum(), 0);
    }

    #[test]
    fn test_intersection_disjoint() {
        let sdr1 = make_sdr(100, &[0, 1, 2]);
        let sdr2 = make_sdr(100, &[50, 51, 52]);
        let mut result = Sdr::new(&[100]);

        result.intersection(&sdr1, &sdr2).unwrap();
        assert_eq!(result.get_sum(), 0);
    }

    #[test]
    fn test_intersection_partial() {
        let sdr1 = make_sdr(100, &[1, 2, 3, 4, 5]);
        let sdr2 = make_sdr(100, &[3, 4, 5, 6, 7]);
        let mut result = Sdr::new(&[100]);

        result.intersection(&sdr1, &sdr2).unwrap();
        assert_eq!(result.get_sparse(), vec![3, 4, 5]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn prop_intersection_equals_set_intersection(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);
            let mut result = Sdr::new(&[500]);

            result.intersection(&sdr1, &sdr2).unwrap();

            // Reference implementation
            let set1: HashSet<u32> = sdr1.get_sparse().into_iter().collect();
            let set2: HashSet<u32> = sdr2.get_sparse().into_iter().collect();
            let mut expected: Vec<u32> = set1.intersection(&set2).copied().collect();
            expected.sort_unstable();

            prop_assert_eq!(result.get_sparse(), expected);
        }

        #[test]
        fn prop_intersection_is_commutative(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);
            let mut result1 = Sdr::new(&[500]);
            let mut result2 = Sdr::new(&[500]);

            result1.intersection(&sdr1, &sdr2).unwrap();
            result2.intersection(&sdr2, &sdr1).unwrap();

            prop_assert_eq!(result1.get_sparse(), result2.get_sparse());
        }

        #[test]
        fn prop_intersection_size_equals_overlap(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);
            let mut result = Sdr::new(&[500]);

            result.intersection(&sdr1, &sdr2).unwrap();

            prop_assert_eq!(result.get_sum(), sdr1.get_overlap(&sdr2));
        }
    }

    // -------------------------------------------------------------------------
    // Union Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_union_empty() {
        let sdr1 = make_sdr(100, &[1, 2, 3]);
        let sdr2 = Sdr::new(&[100]);
        let mut result = Sdr::new(&[100]);

        result.set_union(&sdr1, &sdr2).unwrap();
        assert_eq!(result.get_sparse(), vec![1, 2, 3]);
    }

    #[test]
    fn test_union_disjoint() {
        let sdr1 = make_sdr(100, &[0, 1, 2]);
        let sdr2 = make_sdr(100, &[50, 51, 52]);
        let mut result = Sdr::new(&[100]);

        result.set_union(&sdr1, &sdr2).unwrap();
        assert_eq!(result.get_sparse(), vec![0, 1, 2, 50, 51, 52]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn prop_union_equals_set_union(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);
            let mut result = Sdr::new(&[500]);

            result.set_union(&sdr1, &sdr2).unwrap();

            // Reference implementation
            let set1: HashSet<u32> = sdr1.get_sparse().into_iter().collect();
            let set2: HashSet<u32> = sdr2.get_sparse().into_iter().collect();
            let mut expected: Vec<u32> = set1.union(&set2).copied().collect();
            expected.sort_unstable();

            prop_assert_eq!(result.get_sparse(), expected);
        }

        #[test]
        fn prop_union_is_commutative(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);
            let mut result1 = Sdr::new(&[500]);
            let mut result2 = Sdr::new(&[500]);

            result1.set_union(&sdr1, &sdr2).unwrap();
            result2.set_union(&sdr2, &sdr1).unwrap();

            prop_assert_eq!(result1.get_sparse(), result2.get_sparse());
        }

        #[test]
        fn prop_union_size_formula(
            indices1 in proptest::collection::vec(0u32..500, 0..50),
            indices2 in proptest::collection::vec(0u32..500, 0..50)
        ) {
            let sdr1 = make_sdr(500, &indices1);
            let sdr2 = make_sdr(500, &indices2);
            let mut result = Sdr::new(&[500]);

            result.set_union(&sdr1, &sdr2).unwrap();

            // |A ∪ B| = |A| + |B| - |A ∩ B|
            let expected = sdr1.get_sum() + sdr2.get_sum() - sdr1.get_overlap(&sdr2);
            prop_assert_eq!(result.get_sum(), expected);
        }
    }

    // -------------------------------------------------------------------------
    // Dense/Sparse Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dense_sparse_roundtrip() {
        let mut sdr = Sdr::new(&[100]);
        let original_sparse = vec![1u32, 5, 10, 20, 50, 99];
        sdr.set_sparse(&original_sparse).unwrap();

        let dense = sdr.get_dense();
        let mut sdr2 = Sdr::new(&[100]);
        sdr2.set_dense(&dense).unwrap();

        assert_eq!(sdr2.get_sparse(), original_sparse);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn prop_dense_sparse_roundtrip(
            indices in proptest::collection::vec(0u32..200, 0..30)
        ) {
            let original = make_sdr(200, &indices);
            let sparse_before = original.get_sparse();

            let dense = original.get_dense();
            let mut restored = Sdr::new(&[200]);
            restored.set_dense(&dense).unwrap();

            prop_assert_eq!(restored.get_sparse(), sparse_before);
        }

        #[test]
        fn prop_dense_sum_equals_sparse_len(
            indices in proptest::collection::vec(0u32..200, 0..30)
        ) {
            let sdr = make_sdr(200, &indices);
            let dense = sdr.get_dense();
            let sparse = sdr.get_sparse();

            let dense_sum: usize = dense.iter().map(|&b| b as usize).sum();
            prop_assert_eq!(dense_sum, sparse.len());
        }
    }
}

// =============================================================================
// PERMANENCE OPERATION CORRECTNESS TESTS
// =============================================================================

mod permanence_operations {
    use super::*;

    /// Reference implementation for permanence clamping.
    fn clamp_reference(value: f32, min: f32, max: f32) -> f32 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }

    #[test]
    fn test_clamp_basic() {
        assert_eq!(clamp_reference(0.5, 0.0, 1.0), 0.5);
        assert_eq!(clamp_reference(-0.1, 0.0, 1.0), 0.0);
        assert_eq!(clamp_reference(1.5, 0.0, 1.0), 1.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_clamp_bounds(value in -1.0f32..2.0f32) {
            let result = clamp_reference(value, 0.0, 1.0);
            prop_assert!(result >= 0.0);
            prop_assert!(result <= 1.0);
        }

        #[test]
        fn prop_clamp_preserves_valid(value in 0.0f32..1.0f32) {
            let result = clamp_reference(value, 0.0, 1.0);
            prop_assert!((result - value).abs() < 1e-6);
        }

        #[test]
        fn prop_permanence_update_bounds(
            initial in 0.0f32..1.0f32,
            delta in -0.2f32..0.2f32
        ) {
            let result = clamp_reference(initial + delta, 0.0, 1.0);
            prop_assert!(result >= 0.0);
            prop_assert!(result <= 1.0);
        }
    }

    /// Test batch permanence updates (simulate SIMD operation).
    #[test]
    fn test_batch_permanence_update() {
        let permanences = vec![0.1, 0.5, 0.9, 0.0, 1.0];
        let deltas = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        let results: Vec<f32> = permanences
            .iter()
            .zip(deltas.iter())
            .map(|(&p, &d)| clamp_reference(p + d, 0.0, 1.0))
            .collect();

        assert_eq!(results, vec![0.2, 0.6, 1.0, 0.1, 1.0]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_batch_permanence_all_bounded(
            permanences in proptest::collection::vec(0.0f32..1.0f32, 1..100),
            delta in -0.2f32..0.2f32
        ) {
            let results: Vec<f32> = permanences
                .iter()
                .map(|&p| clamp_reference(p + delta, 0.0, 1.0))
                .collect();

            for &r in &results {
                prop_assert!(r >= 0.0);
                prop_assert!(r <= 1.0);
            }
        }
    }
}

// =============================================================================
// BOOST FACTOR CORRECTNESS TESTS
// =============================================================================

mod boost_operations {
    use super::*;

    /// Reference implementation for boost factor computation.
    fn compute_boost_reference(duty_cycle: f32, target: f32, strength: f32) -> f32 {
        (strength * (target - duty_cycle)).exp()
    }

    #[test]
    fn test_boost_at_target() {
        let boost = compute_boost_reference(0.02, 0.02, 3.0);
        assert!((boost - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_boost_below_target() {
        let boost = compute_boost_reference(0.01, 0.02, 3.0);
        assert!(boost > 1.0);
    }

    #[test]
    fn test_boost_above_target() {
        let boost = compute_boost_reference(0.04, 0.02, 3.0);
        assert!(boost < 1.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_boost_positive(
            duty in 0.0f32..0.1f32,
            target in 0.01f32..0.05f32,
            strength in 0.0f32..10.0f32
        ) {
            let boost = compute_boost_reference(duty, target, strength);
            prop_assert!(boost > 0.0);
            prop_assert!(boost.is_finite());
        }

        #[test]
        fn prop_boost_monotonic_in_duty(
            duty1 in 0.0f32..0.05f32,
            duty2 in 0.05f32..0.1f32,
            target in 0.02f32..0.03f32,
            strength in 1.0f32..5.0f32
        ) {
            // Lower duty cycle should give higher boost
            let boost1 = compute_boost_reference(duty1, target, strength);
            let boost2 = compute_boost_reference(duty2, target, strength);

            // duty1 < duty2 => boost1 > boost2
            prop_assert!(boost1 >= boost2);
        }

        #[test]
        fn prop_boost_at_target_equals_one(
            target in 0.01f32..0.1f32,
            strength in 0.0f32..10.0f32
        ) {
            let boost = compute_boost_reference(target, target, strength);
            prop_assert!((boost - 1.0).abs() < 1e-5);
        }
    }
}

// =============================================================================
// DUTY CYCLE CORRECTNESS TESTS
// =============================================================================

mod duty_cycle_operations {
    use super::*;

    /// Reference implementation for duty cycle update (exponential moving average).
    fn update_duty_cycle_reference(old: f32, new_value: f32, period: f32) -> f32 {
        ((period - 1.0) * old + new_value) / period
    }

    #[test]
    fn test_duty_cycle_convergence() {
        let mut duty = 0.0;
        let period = 100.0;

        // Always active - should converge to 1/period contribution per step
        for _ in 0..1000 {
            duty = update_duty_cycle_reference(duty, 1.0, period);
        }

        // Should be close to 1.0 after many iterations
        assert!((duty - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_duty_cycle_decay() {
        let mut duty = 1.0;
        let period = 100.0;

        // Never active - should decay towards 0
        for _ in 0..1000 {
            duty = update_duty_cycle_reference(duty, 0.0, period);
        }

        assert!(duty < 0.01);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_duty_cycle_bounded(
            old in 0.0f32..1.0f32,
            new_value in 0.0f32..1.0f32,
            period in 10.0f32..1000.0f32
        ) {
            let result = update_duty_cycle_reference(old, new_value, period);
            prop_assert!(result >= 0.0);
            prop_assert!(result <= 1.0);
        }

        #[test]
        fn prop_duty_cycle_smoothing(
            old in 0.0f32..1.0f32,
            new_value in 0.0f32..1.0f32,
            period in 10.0f32..1000.0f32
        ) {
            let result = update_duty_cycle_reference(old, new_value, period);

            // Result should be between old and new (weighted average)
            let min = old.min(new_value);
            let max = old.max(new_value);
            prop_assert!(result >= min - 1e-6);
            prop_assert!(result <= max + 1e-6);
        }
    }
}

// =============================================================================
// CONNECTIONS CORRECTNESS TESTS
// =============================================================================

mod connections_operations {
    use super::*;

    #[test]
    fn test_connections_activity_empty() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let activity = conn.compute_activity(&[], false);
        assert!(activity.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_connections_activity_basic() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        conn.create_synapse(seg, 50, 0.6); // Connected
        conn.create_synapse(seg, 51, 0.4); // Not connected
        conn.create_synapse(seg, 52, 0.6); // Connected

        // Activate cells 50 and 51
        let activity = conn.compute_activity(&[50, 51], false);

        // Only cell 50 has connected synapse
        assert_eq!(activity[seg as usize], 1);
    }

    #[test]
    fn test_adapt_segment_increases_active() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        let syn = conn.create_synapse(seg, 50, 0.5);

        let mut input = Sdr::new(&[100]);
        input.set_sparse(&[50]).unwrap(); // Cell 50 is active

        let old_perm = conn.data_for_synapse(syn).permanence;
        conn.adapt_segment(seg, &input, 0.1, 0.1, false, 0);
        let new_perm = conn.data_for_synapse(syn).permanence;

        assert!(new_perm > old_perm);
    }

    #[test]
    fn test_adapt_segment_decreases_inactive() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        let syn = conn.create_synapse(seg, 50, 0.5);

        let mut input = Sdr::new(&[100]);
        input.set_sparse(&[60]).unwrap(); // Cell 50 is NOT active

        let old_perm = conn.data_for_synapse(syn).permanence;
        conn.adapt_segment(seg, &input, 0.1, 0.1, false, 0);
        let new_perm = conn.data_for_synapse(syn).permanence;

        assert!(new_perm < old_perm);
    }

    #[test]
    fn test_permanence_stays_bounded() {
        let mut conn = Connections::new(ConnectionsParams {
            num_cells: 100,
            connected_threshold: 0.5,
            timeseries: false,
        });

        let seg = conn.create_segment(10, None);
        let syn = conn.create_synapse(seg, 50, 0.95);

        let mut input = Sdr::new(&[100]);
        input.set_sparse(&[50]).unwrap();

        // Try to push permanence above 1.0
        for _ in 0..20 {
            conn.adapt_segment(seg, &input, 0.1, 0.1, false, 0);
        }

        let perm = conn.data_for_synapse(syn).permanence;
        assert!(perm <= 1.0);
    }
}

// =============================================================================
// SPATIAL POOLER CORRECTNESS TESTS
// =============================================================================

mod spatial_pooler_operations {
    use super::*;

    #[test]
    fn test_sp_deterministic() {
        // Same seed should produce same results
        let params = SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![200],
            seed: 42,
            ..Default::default()
        };

        let mut sp1 = SpatialPooler::new(params.clone()).unwrap();
        let mut sp2 = SpatialPooler::new(params).unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output1 = Sdr::new(&[200]);
        let mut output2 = Sdr::new(&[200]);

        input.set_sparse(&[1, 5, 10, 20, 30]).unwrap();

        sp1.compute(&input, false, &mut output1);
        sp2.compute(&input, false, &mut output2);

        assert_eq!(output1.get_sparse(), output2.get_sparse());
    }

    #[test]
    fn test_sp_sparsity() {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![1000],
            global_inhibition: true,
            local_area_density: 0.02,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output = Sdr::new(&[1000]);

        let mut rng = Random::new(42);
        input.randomize(0.1, &mut rng);

        sp.compute(&input, false, &mut output);

        // Should be approximately 2% active
        let sparsity = output.get_sparsity();
        assert!(sparsity > 0.01);
        assert!(sparsity < 0.05);
    }

    #[test]
    fn test_sp_stability() {
        let mut sp = SpatialPooler::new(SpatialPoolerParams {
            input_dimensions: vec![100],
            column_dimensions: vec![200],
            boost_strength: 0.0,
            ..Default::default()
        })
        .unwrap();

        let mut input = Sdr::new(&[100]);
        let mut output1 = Sdr::new(&[200]);
        let mut output2 = Sdr::new(&[200]);

        input.set_sparse(&[1, 5, 10, 20, 30]).unwrap();

        // Same input should produce same output (no learning)
        sp.compute(&input, false, &mut output1);
        sp.compute(&input, false, &mut output2);

        assert_eq!(output1.get_sparse(), output2.get_sparse());
    }
}

// =============================================================================
// TEMPORAL MEMORY CORRECTNESS TESTS
// =============================================================================

mod temporal_memory_operations {
    use super::*;

    #[test]
    fn test_tm_bursting() {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![50],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        let mut active_columns = Sdr::new(&[50]);
        active_columns.set_sparse(&[0, 1, 2]).unwrap();

        // First presentation - should burst
        tm.compute(&active_columns, true);

        // All cells in active columns should be active
        let active_cells = tm.active_cells();
        assert_eq!(active_cells.len(), 3 * 4); // 3 columns * 4 cells
    }

    #[test]
    fn test_tm_reset() {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![50],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        let mut active_columns = Sdr::new(&[50]);
        active_columns.set_sparse(&[0, 1, 2]).unwrap();

        tm.compute(&active_columns, true);
        assert!(!tm.active_cells().is_empty());

        tm.reset();
        assert!(tm.active_cells().is_empty());
        assert!(tm.winner_cells().is_empty());
        assert!(tm.predictive_cells().is_empty());
    }

    #[test]
    fn test_tm_cell_column_mapping() {
        let tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![10],
            cells_per_column: 4,
            ..Default::default()
        })
        .unwrap();

        // Verify internal consistency
        assert_eq!(tm.num_columns(), 10);
        assert_eq!(tm.num_cells(), 40);
        assert_eq!(tm.cells_per_column(), 4);
    }
}

// =============================================================================
// SORTED VECTOR MERGE CORRECTNESS (SIMD TARGET)
// =============================================================================

mod sorted_vector_operations {
    use super::*;

    /// Reference implementation of sorted vector merge intersection.
    fn merge_intersection_reference(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    /// Reference implementation of sorted vector merge union.
    fn merge_union_reference(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(a.len() + b.len());
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    result.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }

        result.extend(&a[i..]);
        result.extend(&b[j..]);
        result
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn prop_merge_intersection_sorted_output(
            mut a in proptest::collection::vec(0u32..1000, 0..100),
            mut b in proptest::collection::vec(0u32..1000, 0..100)
        ) {
            a.sort_unstable();
            a.dedup();
            b.sort_unstable();
            b.dedup();

            let result = merge_intersection_reference(&a, &b);

            // Verify sorted and no duplicates
            for window in result.windows(2) {
                prop_assert!(window[0] < window[1]);
            }
        }

        #[test]
        fn prop_merge_union_sorted_output(
            mut a in proptest::collection::vec(0u32..1000, 0..100),
            mut b in proptest::collection::vec(0u32..1000, 0..100)
        ) {
            a.sort_unstable();
            a.dedup();
            b.sort_unstable();
            b.dedup();

            let result = merge_union_reference(&a, &b);

            // Verify sorted and no duplicates
            for window in result.windows(2) {
                prop_assert!(window[0] < window[1]);
            }
        }

        #[test]
        fn prop_merge_intersection_matches_hashset(
            mut a in proptest::collection::vec(0u32..500, 0..50),
            mut b in proptest::collection::vec(0u32..500, 0..50)
        ) {
            a.sort_unstable();
            a.dedup();
            b.sort_unstable();
            b.dedup();

            let merge_result = merge_intersection_reference(&a, &b);

            let set_a: HashSet<u32> = a.iter().copied().collect();
            let set_b: HashSet<u32> = b.iter().copied().collect();
            let mut set_result: Vec<u32> = set_a.intersection(&set_b).copied().collect();
            set_result.sort_unstable();

            prop_assert_eq!(merge_result, set_result);
        }

        #[test]
        fn prop_merge_union_matches_hashset(
            mut a in proptest::collection::vec(0u32..500, 0..50),
            mut b in proptest::collection::vec(0u32..500, 0..50)
        ) {
            a.sort_unstable();
            a.dedup();
            b.sort_unstable();
            b.dedup();

            let merge_result = merge_union_reference(&a, &b);

            let set_a: HashSet<u32> = a.iter().copied().collect();
            let set_b: HashSet<u32> = b.iter().copied().collect();
            let mut set_result: Vec<u32> = set_a.union(&set_b).copied().collect();
            set_result.sort_unstable();

            prop_assert_eq!(merge_result, set_result);
        }
    }
}
