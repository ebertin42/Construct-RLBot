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

            let mut gs = build_state(
                &cars,
                arr3(&ball["pos"]), arr3(&ball["vel"]), arr3(&ball["ang"]),
                &pad_positions, &pad_active,
            );
            // `build_state` returns a fresh arena every call, so tick_count is 0
            // on every frame. Since 2026-07-26 `build_necto_obs` advances its
            // timers only when tick_count CHANGES (one advance per frame, not
            // one per car -- see NectoTimers), so a sequence replayed at a
            // constant tick_count would freeze the timers and this golden test
            // would compare frame 0's clock against the reference's frame N.
            // Stamp the frame index in tick_skip units, which is what a real
            // rollout produces.
            gs.tick_count = (fi as u64 + 1) * 8;
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

/// E3 regression: Necto's stateful timers must advance once per FRAME, not once
/// per CAR.
///
/// The reference builder is called once per frame for a whole team; ours is
/// called once per car, so before this fix a 3-car frame advanced the arena's
/// boost/demo clocks three times and each car saw a different world. Measured
/// on the broken build, same frame, three cars: pad col21
/// [1.0, 0.99333, 0.98667], demo col21 [0.30, 0.29333, 0.28667] -- differences
/// of exactly TICK_SKIP/1200 and TICK_SKIP/1200, i.e. one extra advance each.
///
/// Invisible at 1v1, which is why it only became reachable when the per-kind
/// foreign guard made 2v2/3v3 necto arenas expressible at all.
#[test]
fn necto_timers_advance_once_per_frame_not_per_car() {
    let pad_positions = canonical_pads();
    let pad_active = vec![true; pad_positions.len()];
    let mk = |orange: bool| FixtureCar {
        pos: [0.0, if orange { 1000.0 } else { -1000.0 }, 17.0],
        vel: [0.0, 0.0, 0.0],
        ang_vel: [0.0, 0.0, 0.0],
        forward: [0.0, 1.0, 0.0],
        up: [0.0, 0.0, 1.0],
        boost: 0.34,
        on_ground: true,
        has_flip: true,
        is_demoed: false,
        team_orange: orange,
    };
    // 3v3: six cars in ONE frame, all reading the same arena.
    let cars: Vec<FixtureCar> = (0..6).map(|i| mk(i >= 3)).collect();
    let mut gs = build_state(
        &cars, [0.0, 0.0, 93.15], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        &pad_positions, &pad_active,
    );
    gs.tick_count = 4242;

    let n_ent = n_entities(cars.len());
    let mut timers = NectoTimers::new(cars.len());
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for ci in 0..cars.len() {
        let mut q = vec![0.0f32; NECTO_Q];
        let mut kv = vec![0.0f32; n_ent * NECTO_KV];
        let mut mask = vec![false; n_ent];
        build_necto_obs(&gs, ci, &[0.0; 8], &mut timers, &mut q, &mut kv, &mut mask);
        // col 21 of every PAD row and every PLAYER row -- the two stateful clocks
        let timer_col = 21;
        let mut got = Vec::new();
        for r in 0..n_ent {
            got.push(kv[r * NECTO_KV + timer_col].abs());
        }
        rows.push(got);
    }
    // Every car of the frame must read an identical timer column (up to the
    // orange sign inversion, which `abs()` above removes -- invert_at only
    // touches cols 5..20, so col 21 is actually unsigned, but abs() makes the
    // test robust to that detail changing).
    for (ci, got) in rows.iter().enumerate().skip(1) {
        for (r, (a, b)) in rows[0].iter().zip(got.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7,
                    "car {ci} read a DIFFERENT timer than car 0 at entity {r}: {b} vs {a} \
                     -- the per-arena timers advanced per car instead of per frame");
        }
    }
    // Sanity: the clocks are actually running, so the assertion above is not
    // vacuously comparing a column of zeros.
    assert!(rows[0].iter().any(|&x| x > 0.0), "no timer fired; the test proves nothing");

    // ...and a NEW frame must advance exactly once, for everyone.
    gs.tick_count = 4250;
    let mut q = vec![0.0f32; NECTO_Q];
    let mut kv = vec![0.0f32; n_ent * NECTO_KV];
    let mut mask = vec![false; n_ent];
    build_necto_obs(&gs, 0, &[0.0; 8], &mut timers, &mut q, &mut kv, &mut mask);
    let after: Vec<f32> = (0..n_ent).map(|r| kv[r * NECTO_KV + 21].abs()).collect();
    assert!(after != rows[0], "a new tick_count must advance the timers");
}
