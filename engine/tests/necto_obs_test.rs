//! Golden test for Necto's observation, against the REAL NectoObsBuilder.
//!
//! Necto's timers are stateful, so each fixture case is a SEQUENCE of frames
//! replayed through one `NectoTimers` -- a per-frame comparison would not catch a
//! timer that advances at the wrong rate.
use construct_engine::obs_necto::{
    build_necto_obs, n_entities, NectoTimers, NECTO_KV, NECTO_Q,
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
        ang_vel: arr3(&v["ang"]),
        forward: arr3(&v["fw"]),
        up: arr3(&v["up"]),
        boost: v["boost"].as_f64().unwrap() as f32,
        on_ground: v["on_ground"].as_bool().unwrap(),
        has_flip: v["has_flip"].as_bool().unwrap(),
        is_demoed: false,
        team_orange: v["team"].as_i64().unwrap() == 1,
    }
}

fn canonical_pads() -> Vec<[f32; 3]> {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v["pad_positions"].as_array().unwrap().iter().map(arr3).collect()
}

/// rlgym-ordered pad flags -> canonical order, by nearest xy.
fn canon_from_rlgym(rl: &[bool], canon_pos: &[[f32; 3]]) -> Vec<bool> {
    let rlgym: Vec<[f32; 2]> = {
        let raw = std::fs::read_to_string("tests/fixtures/necto_rlgym_pads.json")
            .unwrap_or_default();
        if raw.is_empty() {
            // fall back to the canonical order if the helper fixture is absent
            return rl.to_vec();
        }
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        v.as_array().unwrap().iter()
            .map(|p| {
                let a = p.as_array().unwrap();
                [a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32]
            })
            .collect()
    };
    let mut out = vec![false; canon_pos.len()];
    for (j, loc) in rlgym.iter().enumerate() {
        let mut best = 0usize;
        let mut bd = f32::MAX;
        for (i, c) in canon_pos.iter().enumerate() {
            let d = (c[0] - loc[0]).powi(2) + (c[1] - loc[1]).powi(2);
            if d < bd { bd = d; best = i; }
        }
        out[best] = rl[j];
    }
    out
}

#[test]
fn necto_obs_matches_python_fixture() {
    let raw = std::fs::read_to_string("tests/fixtures/necto_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let pad_positions = canonical_pads();

    for (si, seq) in v["sequences"].as_array().unwrap().iter().enumerate() {
        // one timer instance per sequence, exactly as the reference keeps one
        // builder per match
        let mut timers = NectoTimers::new(2);
        for (fi, frame) in seq.as_array().unwrap().iter().enumerate() {
            let cars: Vec<FixtureCar> =
                frame["cars"].as_array().unwrap().iter().map(car_from).collect();
            let ball = &frame["ball"];
            let rl_pads: Vec<bool> = frame["pads"].as_array().unwrap().iter()
                .map(|x| x.as_f64().unwrap() != 0.0).collect();
            let pad_active = canon_from_rlgym(&rl_pads, &pad_positions);
            let mut prev = [0.0f32; 8];
            let pv = frame["prev_action"].as_array().unwrap();
            for k in 0..8 { prev[k] = pv[k].as_f64().unwrap() as f32; }

            let gs = build_state(
                &cars,
                arr3(&ball["pos"]), arr3(&ball["vel"]), arr3(&ball["ang"]),
                &pad_positions, &pad_active,
            );
            let self_idx = frame["self_idx"].as_u64().unwrap() as usize;
            let n_ent = n_entities(cars.len());
            let mut q = vec![0.0f32; NECTO_Q];
            let mut kv = vec![0.0f32; n_ent * NECTO_KV];
            let mut mask = vec![false; n_ent];
            build_necto_obs(&gs, self_idx, &prev, &mut timers, &mut q, &mut kv, &mut mask);

            let eq = frame["q"].as_array().unwrap();
            for k in 0..NECTO_Q {
                let e = eq[k].as_f64().unwrap() as f32;
                assert!((q[k] - e).abs() < 2e-4,
                        "seq {si} frame {fi} q[{k}]: rust {} py {}", q[k], e);
            }
            let ekv = frame["kv"].as_array().unwrap();
            for r in 0..n_ent {
                let row = ekv[r].as_array().unwrap();
                for k in 0..NECTO_KV {
                    let e = row[k].as_f64().unwrap() as f32;
                    assert!((kv[r * NECTO_KV + k] - e).abs() < 2e-4,
                            "seq {si} frame {fi} kv[{r}][{k}]: rust {} py {}",
                            kv[r * NECTO_KV + k], e);
                }
            }
        }
    }
}
