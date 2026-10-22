use mokosh::prelude::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn connection_mutations_preserve_ownership_and_activity(ops in prop::collection::vec((0u8..6,any::<u8>(),any::<u8>(),0u8..101),1..150)) {
        let mut graph=Connections::with_cells(4);
        let mut rng=Random::new(173);
        for (op,x,y,p) in ops {
            let segments:Vec<_>=(0..4).flat_map(|c|graph.segments_for_cell(c).to_vec()).collect();
            if op==0 || segments.is_empty() {graph.create_segment(u32::from(x)%4,Some(3));}
            else {
                let segment=segments[x as usize%segments.len()];
                let synapses=graph.synapses_for_segment(segment).to_vec();
                match op {
                    1=>{graph.create_synapse(segment,u32::from(y)%16,p as f32/100.0);},
                    2 if !synapses.is_empty()=>graph.update_synapse_permanence(synapses[y as usize%synapses.len()],p as f32/100.0),
                    3 if !synapses.is_empty()=>graph.destroy_synapse(synapses[y as usize%synapses.len()]),
                    4=>graph.destroy_segment(segment),
                    5=>graph.grow_synapses(segment,&[0,2,4,6,8,10,12,14],0.4,&mut rng,Some(3),Some(10)),
                    _=>{},
                }
            }
            prop_assert!(graph.validate().is_ok());
            let active:Vec<_>=(0..16).filter(|c|(*c+u32::from(x))%3==0).collect();
            let (connected,potential)=graph.compute_activity_full(&active,false);
            for segment in (0..4).flat_map(|c|graph.segments_for_cell(c).to_vec()) {
                let data:Vec<_>=graph.synapses_for_segment(segment).iter().map(|&s|graph.data_for_synapse(s)).collect();
                let expected_potential=data.iter().filter(|s|active.contains(&s.presynaptic_cell)).count();
                let expected_connected=data.iter().filter(|s|active.contains(&s.presynaptic_cell)&&s.permanence>=graph.connected_threshold()).count();
                prop_assert_eq!(potential[segment as usize] as usize,expected_potential);
                prop_assert_eq!(connected[segment as usize] as usize,expected_connected);
            }
        }
    }
}

#[test]
fn exhaustive_small_topologies_match_coordinate_definition() {
    for a in 1..=5u32 {
        for b in 1..=5u32 {
            for center in 0..a * b {
                for radius in [0, 1, 2, 5, 20, u32::MAX] {
                    for wrap in [false, true] {
                        let x = center / b;
                        let y = center % b;
                        let mut expected = Vec::new();
                        for xx in 0..a {
                            for yy in 0..b {
                                let dx = x.abs_diff(xx);
                                let dy = y.abs_diff(yy);
                                let dx = if wrap { dx.min(a - dx) } else { dx };
                                let dy = if wrap { dy.min(b - dy) } else { dy };
                                if dx <= radius && dy <= radius {
                                    expected.push((xx * b + yy) as usize);
                                }
                            }
                        }
                        let mut actual = Topology::neighborhood(
                            center as usize,
                            &[a, b],
                            radius,
                            if wrap {
                                WrappingMode::Wrap
                            } else {
                                WrappingMode::NoWrap
                            },
                            true,
                        );
                        actual.sort_unstable();
                        assert_eq!(
                            actual, expected,
                            "shape={a},{b} center={center} radius={radius} wrap={wrap}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn corrupt_models_are_rejected_by_deserialize() {
    let mut graph = Connections::with_cells(4);
    let segment = graph.create_segment(1, None);
    graph.create_synapse(segment, 2, 0.6);
    let graph = serde_json::to_value(graph).unwrap();
    for (key, bad) in [
        ("connected_threshold", serde_json::json!(3)),
        ("destroyed_segments", serde_json::json!([0, 0])),
        (
            "potential_synapses_for_presynaptic_cell",
            serde_json::json!({}),
        ),
        ("cells", serde_json::json!([])),
    ] {
        let mut bad_graph = graph.clone();
        bad_graph[key] = bad;
        assert!(
            serde_json::from_value::<Connections>(bad_graph).is_err(),
            "{key}"
        );
    }
    let sp = SpatialPooler::new(SpatialPoolerParams {
        input_dimensions: vec![8],
        column_dimensions: vec![8],
        global_inhibition: false,
        ..Default::default()
    })
    .unwrap();
    let mut bad = serde_json::to_value(sp).unwrap();
    bad["neighbor_map"] = serde_json::json!({"neighbors":{}});
    assert!(serde_json::from_value::<SpatialPooler>(bad).is_err());
    let tm = TemporalMemory::new(TemporalMemoryParams {
        column_dimensions: vec![8],
        ..Default::default()
    })
    .unwrap();
    let mut bad = serde_json::to_value(tm).unwrap();
    bad["num_cells"] = serde_json::json!(1);
    assert!(serde_json::from_value::<TemporalMemory>(bad).is_err());
}

#[test]
fn derived_encoder_resolution_cannot_underflow_to_zero() {
    use mokosh::encoders::{
        RandomDistributedScalarEncoder, RdseParams, ScalarEncoder, ScalarEncoderParams,
    };
    assert!(ScalarEncoder::new(ScalarEncoderParams {
        minimum: 0.0,
        maximum: f32::from_bits(1),
        size: 100,
        active_bits: 3,
        ..Default::default()
    })
    .is_err());
    assert!(RandomDistributedScalarEncoder::new(RdseParams {
        size: 1000,
        active_bits: 40,
        radius: f32::from_bits(1),
        seed: 42,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn boosting_rejects_unrepresentable_exponential_range() {
    assert!(SpatialPooler::new(SpatialPoolerParams {
        boost_strength: f32::MAX,
        ..Default::default()
    })
    .is_err());
}
