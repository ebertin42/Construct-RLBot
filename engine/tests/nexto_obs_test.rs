use construct_engine::obs_nexto::{build_nexto_obs, n_entities, NEXTO_KV, NEXTO_Q};
use construct_engine::test_support::{build_state, FixtureCar};

fn arr3(v: &serde_json::Value) -> [f32; 3] {
    let a = v.as_array().unwrap();
    [
        a[0].as_f64().unwrap() as f32,
        a[1].as_f64().unwrap() as f32,
        a[2].as_f64().unwrap() as f32,
    ]
}

fn car_from(v: &serde_json::Value) -> FixtureCar {
    FixtureCar {
        pos: arr3(&v["pos"]),
        vel: arr3(&v["lin_vel"]),
        ang_vel: arr3(&v["ang_vel"]),
        forward: arr3(&v["forward"]),
        up: arr3(&v["up"]),
        boost: v["boost"].as_f64().unwrap() as f32,
        on_ground: v["on_ground"].as_f64().unwrap() != 0.0,
        has_flip: v["has_flip"].as_f64().unwrap() != 0.0,
        is_demoed: v["demo"].as_f64().unwrap() != 0.0,
        team_orange: v["team"].as_f64().unwrap() == 1.0,
    }
}

/// Canonical pad positions, matching the engine's arena order. Only used so
/// build_state can place pad activity by position.
fn canonical_pads() -> Vec<[f32; 3]> {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v["pad_positions"].as_array().unwrap().iter().map(arr3).collect()
}

#[test]
fn nexto_obs_matches_python_fixture() {
    let raw = std::fs::read_to_string("tests/fixtures/nexto_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let pad_positions = canonical_pads();

    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let cars: Vec<FixtureCar> =
            case["cars"].as_array().unwrap().iter().map(car_from).collect();
        let ball = &case["ball"];
        // fixture pads are in rlgym BOOST_LOCATIONS order; the engine stores
        // canonical order, and the builder permutes. Map back here so the
        // engine state carries the same physical pad activity.
        let rl_pads: Vec<bool> = case["pads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() != 0.0)
            .collect();
        let pad_active = canon_from_rlgym(&rl_pads, &pad_positions);

        let mut prev = [0.0f32; 8];
        let pv = case["prev_action"].as_array().unwrap();
        for k in 0..8 {
            prev[k] = pv[k].as_f64().unwrap() as f32;
        }

        let gs = build_state(
            &cars,
            arr3(&ball["pos"]),
            arr3(&ball["lin_vel"]),
            arr3(&ball["ang_vel"]),
            &pad_positions,
            &pad_active,
        );

        // fixture self_idx indexes the car-id-sorted player order, which
        // build_state produces as [blue, orange]
        let self_idx = case["self_idx"].as_u64().unwrap() as usize;
        let n_ent = n_entities(cars.len());
        let mut q = vec![0.0f32; NEXTO_Q];
        let mut kv = vec![0.0f32; n_ent * NEXTO_KV];
        let mut mask = vec![false; n_ent];
        build_nexto_obs(&gs, self_idx, &prev, &mut q, &mut kv, &mut mask);

        let eq = case["q"].as_array().unwrap();
        for k in 0..NEXTO_Q {
            let e = eq[k].as_f64().unwrap() as f32;
            assert!(
                (q[k] - e).abs() < 2e-4,
                "case {ci} q[{k}]: rust {} py {}", q[k], e
            );
        }
        let ekv = case["kv"].as_array().unwrap();
        for r in 0..n_ent {
            let row = ekv[r].as_array().unwrap();
            for k in 0..NEXTO_KV {
                let e = row[k].as_f64().unwrap() as f32;
                assert!(
                    (kv[r * NEXTO_KV + k] - e).abs() < 2e-4,
                    "case {ci} kv[{r}][{k}]: rust {} py {}",
                    kv[r * NEXTO_KV + k], e
                );
            }
        }
        let em = case["mask"].as_array().unwrap();
        for r in 0..n_ent {
            assert_eq!(mask[r], em[r].as_f64().unwrap() != 0.0, "case {ci} mask[{r}]");
        }
    }
}

/// rlgym-ordered pad flags -> canonical-ordered, by nearest xy.
fn canon_from_rlgym(rl: &[bool], canon_pos: &[[f32; 3]]) -> Vec<bool> {
    const BOOST_LOCATIONS: [[f32; 2]; 34] = [
        [0.0, -4240.0], [-1792.0, -4184.0], [1792.0, -4184.0], [-3072.0, -4096.0],
        [3072.0, -4096.0], [-940.0, -3308.0], [940.0, -3308.0], [0.0, -2816.0],
        [-3584.0, -2484.0], [3584.0, -2484.0], [-1788.0, -2300.0], [1788.0, -2300.0],
        [-2048.0, -1036.0], [0.0, -1024.0], [2048.0, -1036.0], [-3584.0, 0.0],
        [-1024.0, 0.0], [1024.0, 0.0], [3584.0, 0.0], [-2048.0, 1036.0],
        [0.0, 1024.0], [2048.0, 1036.0], [-1788.0, 2300.0], [1788.0, 2300.0],
        [-3584.0, 2484.0], [3584.0, 2484.0], [0.0, 2816.0], [-940.0, 3310.0],
        [940.0, 3308.0], [-3072.0, 4096.0], [3072.0, 4096.0], [-1792.0, 4184.0],
        [1792.0, 4184.0], [0.0, 4240.0],
    ];
    let mut out = vec![false; canon_pos.len()];
    for (j, loc) in BOOST_LOCATIONS.iter().enumerate() {
        let mut best = 0usize;
        let mut bd = f32::MAX;
        for (i, c) in canon_pos.iter().enumerate() {
            let d = (c[0] - loc[0]).powi(2) + (c[1] - loc[1]).powi(2);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        out[best] = rl[j];
    }
    out
}
