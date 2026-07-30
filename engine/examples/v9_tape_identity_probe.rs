// THROWAWAY REVIEW PROBE (2026-07-30). Question: does the air-hang append change
// the reward stream that configs/reward_v9_aerial.toml -- the tape both live
// 2.15B-step runs are on -- produces?
//
// It only uses symbols that exist BOTH before and after the change, so the same
// file compiles against HEAD's reward.rs and against the working-tree one, and
// the two outputs can be diffed byte for byte.
//
// Run (from engine/, ASLR off because the engine is not bit-reproducible under it):
//   setarch -R cargo run --example v9_tape_identity_probe
use construct_engine::reward::{compute_terms, RewardConfig, N_TERMS};
use construct_engine::sim_init::ensure_init;
use rocketsim_rs::math::{RotMat, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const T: f32 = 1.2; // AIR_HANG_T, hardcoded so this compiles pre-change too

struct R(u64);
impl R {
    fn nx(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn uni(&mut self) -> f32 {
        (self.nx() % 20001) as f32 / 10000.0 - 1.0
    }
    fn chance(&mut self, n: u32) -> bool {
        self.nx() % n == 0
    }
}

/// FNV-1a over the exact bits of every reward, so a single differing f32 in
/// thousands moves the digest.
struct H(u64);
impl H {
    fn new() -> Self {
        H(0xcbf29ce484222325)
    }
    fn eat(&mut self, x: f32) {
        for b in x.to_bits().to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn main() {
    ensure_init(None);
    let cfg = RewardConfig::load("../configs/reward_v9_aerial.toml").expect("v9 tape");
    cfg.validate().expect("v9 tape validates");
    println!("# N_TERMS at compile time = {N_TERMS}");

    // ---------------------------------------------------------------- part A --
    // SYNTHETIC state pairs. No physics, so this half is deterministic
    // regardless of ASLR: the post-jump hang clock is swept straight across the
    // threshold under every flag combination, which is exactly the input the new
    // branch keys on.
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(7));
    let base = arena.pin_mut().get_game_state();

    let mut ha = H::new();
    let mut n_a = 0u32;
    let mut cross_a = 0u32;
    for step in 0..41u32 {
        let t0 = step as f32 * 0.05;
        for &t1 in &[t0, t0 + 0.02, t0 + 0.0667, t0 + 0.15] {
            for &ground in &[false, true] {
                for &jumped in &[false, true] {
                    for &touched in &[false, true] {
                        let mut prev = base.clone();
                        prev.tick_count = 1000;
                        prev.ball.pos = Vec3::new(0.0, 0.0, 800.0);
                        prev.ball.vel = Vec3::new(0.0, 100.0, -50.0);
                        for c in prev.cars.iter_mut() {
                            c.state.pos = Vec3::new(0.0, -300.0, 700.0);
                            c.state.vel = Vec3::new(0.0, 900.0, 120.0);
                            c.state.boost = 47.0;
                            c.state.is_on_ground = ground;
                            c.state.has_jumped = jumped;
                            c.state.is_jumping = false;
                            c.state.air_time = t0 + 0.3;
                            c.state.air_time_since_jump = t0;
                            c.state.ball_hit_info.is_valid = true;
                            c.state.ball_hit_info.tick_count_when_hit = 900;
                        }
                        let mut cur = prev.clone();
                        cur.tick_count = 1008;
                        cur.ball.vel = Vec3::new(0.0, 1400.0, 40.0);
                        for c in cur.cars.iter_mut() {
                            c.state.pos = Vec3::new(0.0, -240.0, 760.0);
                            c.state.air_time = t1 + 0.3;
                            c.state.air_time_since_jump = t1;
                            c.state.boost = 61.0; // a pad, to keep boost_pickup live
                            if touched {
                                c.state.ball_hit_info.tick_count_when_hit = 1004;
                            }
                        }
                        for i in 0..cur.cars.len() {
                            for scored in [None, Some(Team::Blue)] {
                                let mut terms = [0.0f64; N_TERMS];
                                let r =
                                    compute_terms(&prev, &cur, i, scored, &cfg, &mut terms);
                                ha.eat(r);
                                n_a += 1;
                            }
                            if !ground && jumped && t0 < T && t1 >= T {
                                cross_a += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!("A synthetic  rows={n_a} hang_crossings_in_input={cross_a} digest={:016x}", ha.0);
    assert!(cross_a > 0, "the synthetic sweep must actually cross the threshold");

    // ---------------------------------------------------------------- part B --
    // REAL rollout. Two cars, a deterministic pseudo-random control tape with a
    // hop/hold-heavy jump pattern, ball parked in reach so touches, boost pads,
    // flips and hang crossings all land inside one stream.
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(3));
    let ids: Vec<u32> = arena
        .pin_mut()
        .get_game_state()
        .cars
        .iter()
        .map(|c| c.id)
        .collect();
    for (k, id) in ids.iter().enumerate() {
        let mut cs = arena.pin_mut().get_car(*id);
        cs.pos = Vec3::new(-500.0 + 1000.0 * k as f32, -1500.0, 17.0);
        cs.vel = Vec3::new(0.0, 300.0, 0.0);
        cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        cs.rot_mat = RotMat {
            forward: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(-1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0),
        };
        cs.boost = 40.0;
        arena.pin_mut().set_car(*id, cs).unwrap();
    }
    let mut b = arena.pin_mut().get_ball();
    b.pos = Vec3::new(0.0, 400.0, 300.0);
    b.vel = Vec3::new(0.0, -200.0, 0.0);
    arena.pin_mut().set_ball(b);

    let mut rng = R(0xC0FFEE);
    let mut hb = H::new();
    let mut prev = arena.pin_mut().get_game_state();
    let (mut n_b, mut cross_b, mut air_b, mut hold) = (0u32, 0u32, 0u32, [0u32; 2]);
    let mut sum = 0.0f64;
    for _ in 0..1200u32 {
        for (k, id) in ids.iter().enumerate() {
            // Jump pattern: start a hop at random, then hold it for 1-4
            // decisions. That is the family that straddles AIR_HANG_T.
            let st = arena.pin_mut().get_car(*id);
            if hold[k] > 0 {
                hold[k] -= 1;
            } else if st.is_on_ground && rng.chance(4) {
                hold[k] = 1 + rng.nx() % 4;
            }
            let c = CarControls {
                throttle: rng.uni(),
                steer: rng.uni(),
                pitch: rng.uni(),
                yaw: rng.uni(),
                roll: rng.uni(),
                jump: hold[k] > 0,
                boost: rng.chance(3),
                handbrake: rng.chance(9),
            };
            arena.pin_mut().set_car_controls(*id, c).unwrap();
        }
        arena.pin_mut().step(8);
        let cur = arena.pin_mut().get_game_state();
        for i in 0..cur.cars.len() {
            let mut terms = [0.0f64; N_TERMS];
            let r = compute_terms(&prev, &cur, i, None, &cfg, &mut terms);
            hb.eat(r);
            sum += r as f64;
            n_b += 1;
            let (p, c) = (&prev.cars[i].state, &cur.cars[i].state);
            if !c.is_on_ground {
                air_b += 1;
            }
            if !c.is_on_ground && p.air_time_since_jump < T && c.air_time_since_jump >= T {
                cross_b += 1;
            }
        }
        prev = cur;
    }
    println!(
        "B rollout    rows={n_b} airborne={air_b} hang_crossings_in_input={cross_b} \
         sum={:.9} digest={:016x}",
        sum, hb.0
    );
    assert!(cross_b > 0, "the rollout must actually cross the threshold");
    println!("COMBINED digest={:016x}", ha.0 ^ hb.0);
}
