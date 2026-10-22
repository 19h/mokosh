use mokosh::prelude::*;

#[test]
fn reference_sequence_campaign() {
    let fixture = std::env::var("MOKOSH_CAMPAIGN_FIXTURE")
        .map(|path| std::fs::read_to_string(path).expect("Read live C++ fixture"))
        .unwrap_or_else(|_| include_str!("reference/tm_campaign_d869994.txt").to_string());
    let mut lines = fixture.lines();
    for seed in 1..=12 {
        let mut tm = TemporalMemory::new(TemporalMemoryParams {
            column_dimensions: vec![16],
            cells_per_column: 4,
            activation_threshold: 3,
            initial_permanence: 0.4,
            connected_permanence: 0.5,
            min_threshold: 2,
            max_new_synapse_count: 3,
            permanence_increment: 0.1,
            permanence_decrement: 0.05,
            predicted_segment_decrement: 0.02,
            seed,
            max_segments_per_cell: 4,
            max_synapses_per_segment: 8,
            ..Default::default()
        })
        .unwrap();
        let mut state = 0x12345678u32 + seed as u32;
        for step in 0..300 {
            if step % 37 == 0 {
                tm.reset();
            }
            let mut cols = Vec::new();
            for _ in 0..6 {
                state = 1664525u32.wrapping_mul(state).wrapping_add(1013904223);
                cols.push((state >> 16) % 16);
            }
            cols.sort_unstable();
            cols.dedup();
            let mut input = Sdr::new(&[16]);
            input.set_sparse(&cols).unwrap();
            tm.compute(&input, step % 3 != 0);
            let mut actual = vec![seed as i64, step];
            for cells in [tm.active_cells(), tm.winner_cells(), tm.predictive_cells()] {
                actual.push(-1);
                actual.extend(cells.iter().map(|&c| i64::from(c)));
            }
            let mut hash = 2166136261u32;
            let mut full_graph = Vec::new();
            let mut mix = |value: u32| {
                full_graph.push(value);
                hash = (hash ^ value).wrapping_mul(16777619);
            };
            let connections = tm.connections();
            for cell in 0..64 {
                let mut segments = connections.segments_for_cell(cell).to_vec();
                segments.sort_unstable();
                mix(cell);
                mix(segments.len() as u32);
                for segment in segments {
                    mix(segment);
                    let mut synapses: Vec<_> = connections
                        .synapses_for_segment(segment)
                        .iter()
                        .map(|&syn| {
                            let data = connections.data_for_synapse(syn);
                            (data.presynaptic_cell, data.permanence.to_bits())
                        })
                        .collect();
                    synapses.sort_unstable();
                    mix(synapses.len() as u32);
                    for (cell, bits) in synapses {
                        mix(cell);
                        mix(bits);
                    }
                }
            }
            actual.extend([-2, i64::from(hash)]);
            if std::env::var_os("MOKOSH_FULL_GRAPH").is_some() {
                actual.push(-3);
                actual.extend(full_graph.into_iter().map(i64::from));
            }
            let expected: Vec<i64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            assert_eq!(actual, expected, "seed={seed}, step={step}");
        }
    }
    assert!(lines.next().is_none());
}
