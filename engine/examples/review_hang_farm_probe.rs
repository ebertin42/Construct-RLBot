// REVIEW PROBE (scratch, not part of the diff under review).
// Prices the FULL ball-ignoring farm the shipped hang gate would open, with the
// live coefficients (vel_to_ball 0.05, air_setup 0.3 @ gamma 0.9954, ramp
// z_lo 100 / z_hi 1400), plus two cycles the author's hang_gate_probe tape set
// does not contain:
//
//  1. RATE. hold-3 measures 28.2 qualifying/min, which is the number the
//     exposure arithmetic and the `per_min <= 35.0` test bound are priced off.
//     A "rescue double jump" fired only when a phase would otherwise fall short
//     of T raises the qualifying fraction toward 1.0.
//
//  2. LAND INVERTED. `is_on_ground` is `numWheelsInContact >= 3` with no surface
//     test, and Car.cpp:550 only clears `has_jumped` when `isOnGround`, so a car
//     resting on its ROOF on the floor keeps accumulating
//     `air_time_since_jump`. Tape G puts a motionless upside-down car on the
//     floor and asks whether the gate pays it.
use construct_engine::sim_init;
use rocketsim_rs::math::{RotMat, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32;
const T: f32 = 1.2;
const BALL_Z: f32 = 900.0;
const GAMMA: f32 = 0.9954;
const W_SETUP: f32 = 0.3;
const W_V2B: f32 = 0.05;

fn phi(on_ground: bool, car: Vec3, ball: Vec3) -> f32 {
    if on_ground {
        return 0.0;
    }
    let d = [ball.x - car.x, ball.y - car.y, ball.z - car.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let h = ((ball.z - 100.0) / (1400.0 - 100.0)).clamp(0.0, 1.0);
    h * (1.0 - dist / 2500.0).clamp(0.0, 1.0)
}

fn v2b(p: Vec3, v: Vec3, ball: Vec3) -> f32 {
    let d = [ball.x - p.x, ball.y - p.y, ball.z - p.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let proj = (v.x * d[0] + v.y * d[1] + v.z * d[2]) / dist;
    W_V2B * (proj / 2300.0).clamp(0.0, 1.0)
}

fn run(
    label: &str,
    secs: f32,
    invert_start: bool,
    tape: &dyn Fn(u32, &rocketsim_rs::sim::CarState, bool) -> CarControls,
) {
    let n = (secs * HZ) as u32;
    let mut arena = Arena::default_standard();
    let _ = arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = Vec3::new(0.0, -2000.0, if invert_start { 30.0 } else { 17.0 });
    cs.vel = Vec3::new(0.0, 0.0, 0.0);
    cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    cs.rot_mat = if invert_start {
        // upside down: up = -z
        RotMat {
            forward: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, -1.0),
        }
    } else {
        RotMat {
            forward: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(-1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0),
        }
    };
    cs.boost = 100.0;
    if invert_start {
        // the state a car that jumped and flipped onto its roof carries
        cs.has_jumped = true;
        cs.is_jumping = false;
    }
    arena.pin_mut().set_car(id, cs).unwrap();

    let mut prev = arena.pin_mut().get_car(id);
    let (mut phases, mut crossings, mut air_decs, mut in_air) = (0u32, 0u32, 0u32, false);
    let (mut v2b_sum, mut setup_sum, mut max_clock, mut still_air) = (0.0f32, 0.0f32, 0.0f32, 0u32);
    let mut boost0 = 100.0f32;
    for i in 0..n {
        // ball pinned directly overhead: the gate-wide-open geometry
        let st = arena.pin_mut().get_car(id);
        let mut b = arena.pin_mut().get_ball();
        b.pos = Vec3::new(st.pos.x, st.pos.y, BALL_Z);
        b.vel = Vec3::new(0.0, 0.0, 0.0);
        b.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(b);
        let ball_now = arena.pin_mut().get_ball().pos;
        let prev_phi = phi(prev.is_on_ground, prev.pos, ball_now);

        let c = tape(i, &st, prev.last_controls.jump);
        arena.pin_mut().set_car_controls(id, c).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_car(id);
        let ball_after = arena.pin_mut().get_ball().pos;

        if !cur.is_on_ground {
            air_decs += 1;
            if !in_air {
                in_air = true;
                phases += 1;
            }
            let sp = (cur.vel.x * cur.vel.x + cur.vel.y * cur.vel.y + cur.vel.z * cur.vel.z).sqrt();
            if cur.pos.z < 60.0 && sp < 50.0 {
                still_air += 1;
            }
        } else {
            in_air = false;
        }
        if !cur.is_on_ground && prev.air_time_since_jump < T && cur.air_time_since_jump >= T {
            crossings += 1;
        }
        max_clock = max_clock.max(cur.air_time_since_jump);
        v2b_sum += v2b(cur.pos, cur.vel, ball_after);
        setup_sum += W_SETUP * (GAMMA * phi(cur.is_on_ground, cur.pos, ball_after) - prev_phi);
        prev = cur;
    }
    let mins = secs / 60.0;
    let xmin = crossings as f32 / mins;
    // own money at the doc's own honest band for r, per 300 s match
    let (lo, hi) = (0.0134f32, 0.0282f32);
    let per_match = 5.0;
    let own_lo = xmin * per_match * lo;
    let own_hi = xmin * per_match * hi;
    let v2b_m = v2b_sum / mins * per_match;
    let setup_m = setup_sum / mins * per_match;
    println!(
        "{label:<40} ph {:>5.1}/min  X {:>5.1}/min  air {:>5.1}%  clk_max {:>5.2}  \
         still&air {:>4.1}%  bst {:>5.1}  ||  air_hang {:>5.2}-{:>5.2}  v2b {:>5.2}  \
         setup {:>+5.2}  TOTAL/match {:>5.2}-{:>5.2}",
        phases as f32 / mins,
        xmin,
        100.0 * air_decs as f32 / n as f32,
        max_clock,
        100.0 * still_air as f32 / n as f32,
        boost0 - prev.boost,
        own_lo,
        own_hi,
        v2b_m,
        setup_m,
        own_lo + v2b_m + setup_m,
        own_hi + v2b_m + setup_m,
    );
    boost0 = 0.0;
    let _ = boost0;
}

fn main() {
    sim_init::ensure_init(None);
    println!("gate = !is_on_ground && prev.clock < {T} && cur.clock >= {T}");
    println!("ball pinned overhead at z {BALL_Z}; 300 s match; r band 0.0134-0.0282/event\n");

    run("A hold-3 hop (author's tape)", 200.0, false, &|_i, st, lj| CarControls {
        jump: if lj { st.is_jumping && st.jump_time < 0.1999 } else { st.is_on_ground },
        ..Default::default()
    });
    run("B min hop (max rate, unqualifying)", 200.0, false, &|_i, st, lj| CarControls {
        jump: !lj && st.is_on_ground,
        ..Default::default()
    });
    run("D hold-3 + rescue double jump", 200.0, false, &|_i, st, lj| CarControls {
        jump: if lj {
            st.is_jumping && st.jump_time < 0.1999
        } else if st.is_on_ground {
            true
        } else {
            !st.has_double_jumped
                && !st.has_flipped
                && st.air_time_since_jump > 0.55
                && st.air_time_since_jump < 1.2
                && st.vel.z < 0.0
                && st.pos.z < 90.0
        },
        ..Default::default()
    });
    run("D2 hold-3 + rescue + pitch-down boost", 200.0, false, &|_i, st, lj| CarControls {
        jump: if lj {
            st.is_jumping && st.jump_time < 0.1999
        } else if st.is_on_ground {
            true
        } else {
            !st.has_double_jumped
                && !st.has_flipped
                && st.air_time_since_jump > 0.55
                && st.air_time_since_jump < 1.2
                && st.vel.z < 0.0
                && st.pos.z < 90.0
        },
        pitch: if !st.is_on_ground && st.air_time_since_jump >= T { -1.0 } else { 0.0 },
        boost: !st.is_on_ground && st.air_time_since_jump >= T && st.rot_mat.forward.z < -0.4,
        ..Default::default()
    });
    // G: MOTIONLESS car resting on its ROOF on the floor, has_jumped already set,
    // zero control input for the whole run.
    run("G inverted at rest on the floor, no input", 30.0, true, &|_i, _st, _lj| {
        CarControls::default()
    });
}
