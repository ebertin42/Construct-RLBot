// ADVERSARIAL PROBE (do not ship): price the farm of a HANG-TIME-gated takeoff
// bonus. Candidate gate, evaluated per car per decision on (prev, cur) only:
//
//   !cur.is_on_ground
//   && prev.air_time_since_jump <  T_HANG
//   && cur.air_time_since_jump  >= T_HANG
//
// air_time_since_jump accumulates only while (has_jumped && !is_jumping), is
// zeroed on every grounded tick (Car.cpp:692-701), and is_jumping can only be
// re-armed from the ground (Car.cpp:570), so within one air phase the quantity
// is strictly monotone -> the crossing fires AT MOST ONCE PER AIR PHASE, and
// only for a phase that a real jump impulse from a surface started.
//
// What this measures, per tape: air phases, the max air_time_since_jump each
// phase reached (which gives the crossing count for ANY T in one run), apex,
// and boost spent. Rate x 5 min x r = reward/match.
// Run: cargo run --example hang_gate_probe   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::Vec3;
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32; // 15 decisions/s
const SECS: f32 = 200.0;
const N: u32 = (SECS * HZ) as u32;
const TS: [f32; 8] = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.4];

struct Run {
    label: String,
    phase_max: Vec<f32>, // max air_time_since_jump per air phase
    phase_apex: Vec<f32>,
    phase_boost: Vec<f32>,
    crossings_raw: usize, // prev<T<=cur counted directly at T=1.2, sanity check
    air_decisions: u32,
}

fn run(label: &str, start: Vec3, fwd_yaw: f32, tape: &dyn Fn(u32, &rocketsim_rs::sim::CarState, bool) -> CarControls) -> Run {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = start;
    cs.vel = Vec3::new(0.0, 0.0, 0.0);
    cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    cs.rot_mat = rocketsim_rs::math::RotMat {
        forward: Vec3::new(fwd_yaw.cos(), fwd_yaw.sin(), 0.0),
        right: Vec3::new(-fwd_yaw.sin(), fwd_yaw.cos(), 0.0),
        up: Vec3::new(0.0, 0.0, 1.0),
    };
    cs.boost = 100.0;
    arena.pin_mut().set_car(id, cs).unwrap();
    // ball parked far away and low: this probe is about the car only
    let mut b = arena.pin_mut().get_ball();
    b.pos = Vec3::new(0.0, 4000.0, 93.15);
    b.vel = Vec3::new(0.0, 0.0, 0.0);
    arena.pin_mut().set_ball(b);

    let mut r = Run {
        label: label.to_string(),
        phase_max: vec![],
        phase_apex: vec![],
        phase_boost: vec![],
        crossings_raw: 0,
        air_decisions: 0,
    };
    let mut prev = arena.pin_mut().get_car(id);
    let mut last_jump = false;
    let (mut cur_max, mut cur_apex, mut boost_at_takeoff, mut in_air) = (0.0f32, 0.0f32, 100.0f32, false);
    for i in 0..N {
        let st = arena.pin_mut().get_car(id);
        let c = tape(i, &st, last_jump);
        last_jump = c.jump;
        arena.pin_mut().set_car_controls(id, c).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_car(id);
        if !cur.is_on_ground {
            r.air_decisions += 1;
        }
        if prev.air_time_since_jump < 1.2 && cur.air_time_since_jump >= 1.2 && !cur.is_on_ground {
            r.crossings_raw += 1;
        }
        // air-phase bookkeeping: a phase is a maximal !is_on_ground run
        if !cur.is_on_ground {
            if !in_air {
                in_air = true;
                cur_max = 0.0;
                cur_apex = 0.0;
                boost_at_takeoff = prev.boost;
            }
            cur_max = cur_max.max(cur.air_time_since_jump);
            cur_apex = cur_apex.max(cur.pos.z);
        } else if in_air {
            in_air = false;
            r.phase_max.push(cur_max);
            r.phase_apex.push(cur_apex);
            r.phase_boost.push((boost_at_takeoff - prev.boost).max(0.0));
        }
        prev = cur;
    }
    r
}

fn main() {
    sim_init::ensure_init(None);
    let mut runs: Vec<Run> = vec![];

    // A: greedy hop farm -- re-press jump the decision after landing, hold h.
    for h in 1..=6u32 {
        let label = format!("greedy hop, hold {h} dec");
        // state machine encoded in the closure via the live CarState: press jump
        // whenever on the ground, and keep pressing while jump_time is short.
        let tape = move |_i: u32, st: &rocketsim_rs::sim::CarState, lj: bool| CarControls {
            // RISING EDGE discipline: hold only while the jump is still doing
            // work, and re-press only from the ground after a release.
            jump: if lj {
                st.is_jumping && st.jump_time < h as f32 / HZ - 1e-4
            } else {
                st.is_on_ground
            },
            ..Default::default()
        };
        runs.push(run(&label, Vec3::new(0.0, -2000.0, 17.0), 1.5708, &tape));
    }

    // B: greedy hop + double jump (release, then a second press in the air)
    runs.push(run(
        "greedy hop + double jump",
        Vec3::new(0.0, -2000.0, 17.0),
        1.5708,
        &|_i, st, lj| CarControls {
            jump: if lj {
                st.is_jumping && st.jump_time < 0.1999
            } else {
                st.is_on_ground
                    || (!st.is_on_ground && !st.is_jumping && st.has_jumped && !st.has_double_jumped)
            },
            ..Default::default()
        },
    ));

    // C: greedy powered takeoff -- jump, pitch back, boost up. Tank-limited (no
    // pad pickups in this tape), so boost/phase is the number that sets the
    // sustainable rate once pads are available.
    runs.push(run(
        "greedy jump+pitch+boost (aerial)",
        Vec3::new(0.0, -2000.0, 17.0),
        1.5708,
        &|_i, st, lj| CarControls {
            jump: if lj { st.is_jumping && st.jump_time < 0.1999 } else { st.is_on_ground },
            pitch: if !st.is_on_ground && st.rot_mat.forward.z < 0.85 { 1.0 } else { 0.0 },
            boost: !st.is_on_ground && st.rot_mat.forward.z > 0.5,
            ..Default::default()
        },
    ));

    // D: drive into the corner, NO jump input at all (false-fire check: the
    // ground->air proxy fires 7-9/min here).
    runs.push(run(
        "corner drive, zero jump input",
        Vec3::new(2000.0, -3000.0, 17.0),
        -0.7854,
        &|_i, _st, _lj| CarControls { throttle: 1.0, boost: true, ..Default::default() },
    ));

    // E: wall-climb + jump loop -- buy altitude for free on the wall, jump off
    // it, and let the FALL supply the hang time.
    runs.push(run(
        "wall climb then jump off",
        Vec3::new(3000.0, 0.0, 17.0),
        0.0,
        &|_i, st, lj| CarControls {
            throttle: 1.0,
            boost: st.is_on_ground,
            jump: if lj { false } else { st.is_on_ground && st.pos.z > 600.0 },
            ..Default::default()
        },
    ));

    // F: controls -- sit still, and drive flat with no jump.
    runs.push(run("sit still", Vec3::new(0.0, -2000.0, 17.0), 1.5708, &|_i, _st, _lj| {
        CarControls::default()
    }));
    runs.push(run("drive flat, no jump", Vec3::new(0.0, -3000.0, 17.0), 1.5708, &|_i, _st, _lj| CarControls {
        throttle: 1.0,
        ..Default::default()
    }));
    // E2: wall climb, jump off LOW (faster cycle than the z>600 variant?)
    runs.push(run("wall climb, jump off at z>300", Vec3::new(3000.0, 0.0, 17.0), 0.0, &|_i, st, lj| CarControls {
        throttle: 1.0,
        boost: st.is_on_ground,
        jump: if lj { false } else { st.is_on_ground && st.pos.z > 300.0 },
        ..Default::default()
    }));
    // G: hop while cruising forward (the "jumping bean off-ball" shape) -- does
    // forward speed change the qualifying rate?
    runs.push(run("cruise + greedy hop hold 3", Vec3::new(0.0, -4000.0, 17.0), 1.5708, &|_i, st, lj| CarControls {
        throttle: 1.0,
        jump: if lj { st.is_jumping && st.jump_time < 0.1999 } else { st.is_on_ground },
        ..Default::default()
    }));

    println!("{SECS} s per tape, tick_skip {SKIP} ({HZ} decisions/s)\n");
    println!(
        "{:<34} {:>6} {:>7} {:>7} {:>7} {:>7}  qualifying crossings/min at T =",
        "tape", "phases", "air%", "apex", "bst/ph", "xraw"
    );
    print!("{:<34} {:>6} {:>7} {:>7} {:>7} {:>7}  ", "", "", "", "", "", "");
    for t in TS {
        print!("{t:>6.1}");
    }
    println!();
    for r in &runs {
        let n = r.phase_max.len().max(1) as f32;
        let apex = r.phase_apex.iter().cloned().fold(0.0f32, f32::max);
        let bst: f32 = r.phase_boost.iter().sum::<f32>() / n;
        print!(
            "{:<34} {:>6} {:>6.1}% {:>7.1} {:>7.1} {:>7}  ",
            r.label,
            r.phase_max.len(),
            100.0 * r.air_decisions as f32 / N as f32,
            apex,
            bst,
            r.crossings_raw
        );
        for t in TS {
            let q = r.phase_max.iter().filter(|m| **m >= t).count();
            print!("{:>6.1}", 60.0 * q as f32 / SECS);
        }
        println!();
    }

    // per-phase detail for the two tapes that decide the design
    for r in &runs {
        if r.label.starts_with("greedy hop, hold 3") || r.label.starts_with("wall climb, jump off at z>300") || r.label.starts_with("greedy jump+pitch") {
            let k = r.phase_max.len().min(6);
            println!(
                "\n{}: first {k} phases (max air_time_since_jump / apex / boost)",
                r.label
            );
            for j in 0..k {
                println!(
                    "  phase {j}: since_jump {:.3} apex {:.1} boost {:.1}",
                    r.phase_max[j], r.phase_apex[j], r.phase_boost[j]
                );
            }
        }
    }
}
