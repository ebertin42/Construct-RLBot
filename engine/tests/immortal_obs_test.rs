use construct_engine::obs_advanced::{
    build_advanced_obs, build_advanced_obs_one_other, nearest_opponent_to_ball, ADV_OBS_SIZE,
};
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
        vel: arr3(&v["vel"]),
        ang_vel: arr3(&v["ang_vel"]),
        forward: arr3(&v["forward"]),
        up: arr3(&v["up"]),
        boost: v["boost"].as_f64().unwrap() as f32,
        on_ground: v["on_ground"].as_bool().unwrap(),
        has_flip: v["has_flip"].as_bool().unwrap(),
        is_demoed: v["is_demoed"].as_bool().unwrap(),
        team_orange: v["team"].as_i64().unwrap() == 1,
    }
}

#[test]
fn advanced_obs_matches_python_fixture() {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let pad_positions: Vec<[f32; 3]> =
        v["pad_positions"].as_array().unwrap().iter().map(arr3).collect();

    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let st = &case["state"];
        let cars: Vec<FixtureCar> = st["cars"].as_array().unwrap().iter().map(car_from).collect();
        let ball = &st["ball"];
        let pad_active: Vec<bool> =
            st["pad_active"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() != 0).collect();
        let prev_v = st["prev_action"].as_array().unwrap();
        let mut prev = [0.0f32; 8];
        for k in 0..8 {
            prev[k] = prev_v[k].as_f64().unwrap() as f32;
        }

        let gs = build_state(
            &cars,
            arr3(&ball["pos"]), arr3(&ball["vel"]), arr3(&ball["ang_vel"]),
            &pad_positions, &pad_active,
        );
        let self_idx = case["self_idx"].as_u64().unwrap() as usize;
        let mut out = vec![0.0f32; ADV_OBS_SIZE];
        build_advanced_obs(&gs, self_idx, &prev, &mut out);

        let expect = case["obs"].as_array().unwrap();
        for k in 0..ADV_OBS_SIZE {
            let e = expect[k].as_f64().unwrap() as f32;
            assert!(
                (out[k] - e).abs() < 1e-5,
                "case {ci} idx {k}: rust {} py {}",
                out[k], e
            );
        }
    }
}

/// The truncated builder must be BYTE-IDENTICAL to the untruncated one at 1v1.
///
/// This is what keeps every existing 1v1 element/immortal bench row comparable
/// across the 2026-07-26 change that routed those two bots through the
/// truncated obs: at 1v1 there is exactly one other car and it is the enemy,
/// so "show the nearest opponent" and "show all others" are the same list.
/// Unlike the necto reset fix, this change does not invalidate prior
/// measurements -- and this test is the reason we can say so. It runs over the
/// same real fixture states as the Python-parity test above.
#[test]
fn truncated_obs_is_byte_identical_to_full_obs_at_1v1() {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let pad_positions: Vec<[f32; 3]> =
        v["pad_positions"].as_array().unwrap().iter().map(arr3).collect();

    let mut checked = 0usize;
    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let st = &case["state"];
        let cars: Vec<FixtureCar> = st["cars"].as_array().unwrap().iter().map(car_from).collect();
        // Only 1v1 states can be compared: above that the untruncated builder
        // overruns its 107-float buffer by construction (that overrun is the
        // whole reason the truncated builder exists).
        if cars.len() != 2 {
            continue;
        }
        let ball = &st["ball"];
        let pad_active: Vec<bool> =
            st["pad_active"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() != 0).collect();
        let prev_v = st["prev_action"].as_array().unwrap();
        let mut prev = [0.0f32; 8];
        for k in 0..8 {
            prev[k] = prev_v[k].as_f64().unwrap() as f32;
        }
        let gs = build_state(
            &cars,
            arr3(&ball["pos"]), arr3(&ball["vel"]), arr3(&ball["ang_vel"]),
            &pad_positions, &pad_active,
        );
        let self_idx = case["self_idx"].as_u64().unwrap() as usize;

        let mut full = vec![0.0f32; ADV_OBS_SIZE];
        build_advanced_obs(&gs, self_idx, &prev, &mut full);

        // The selector must find the single opponent, and it must BE an opponent.
        let oi = nearest_opponent_to_ball(&gs, self_idx)
            .unwrap_or_else(|| panic!("case {ci}: no opponent found in a 1v1 state"));
        assert_ne!(oi, self_idx, "case {ci}: selector returned self");
        assert_ne!(
            gs.cars[oi].team, gs.cars[self_idx].team,
            "case {ci}: selector returned a teammate -- that would write an ally \
             into the enemy slot"
        );

        let mut trunc = vec![0.0f32; ADV_OBS_SIZE];
        build_advanced_obs_one_other(&gs, self_idx, oi, &prev, &mut trunc);

        assert_eq!(full, trunc, "case {ci}: truncated obs differs from full obs at 1v1");
        checked += 1;
    }
    assert!(checked > 0, "fixture contained no 1v1 cases to compare");
}
