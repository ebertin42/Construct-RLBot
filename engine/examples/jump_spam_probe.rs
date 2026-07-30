// ADVERSARIAL PROBE (do not ship). Question: if a reward term pays a flat bonus on
// every ground->air transition (prev.is_on_ground && !cur.is_on_ground, sampled once
// per tick_skip=8 decision, exactly like reward.rs sees state), what is the MAXIMUM
// transitions/minute a car can produce, and can wheel-contact chatter beat the jump
// cycle?
//
// Run: cargo run --example jump_spam_probe   (from engine/, debug, no wheel touched)
use construct_engine::sim_init;
use rocketsim_rs::math::{Angle, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32; // 15 decisions/s
const MATCH_DECISIONS: u32 = 4500; // 300 s

#[derive(Default)]
struct Res {
    transitions: u32,
    decisions: u32,
    air_decisions: u32,
    flips: u32,
    jump_edges: u32,
    v2b_sum: f32, // sum of (proj/2300).clamp(0,1) -- multiply by cfg.vel_to_ball
    speed_end: f32,
}

impl Res {
    fn report(&self, label: &str) {
        let per_min = self.transitions as f32 * HZ * 60.0 / self.decisions as f32;
        let cycle_ticks = if self.transitions > 0 {
            (self.decisions * SKIP) as f32 / self.transitions as f32
        } else {
            f32::INFINITY
        };
        println!(
            "{label:<38} trans {:>5}  = {per_min:7.1}/min  cycle {cycle_ticks:6.1} ticks \
             ({:5.3} s)  air {:4.1}%  flips {:>4}  edges {:>4}  \
             pay/min @4e-3 {:6.3} @8e-3 {:6.3}  pay/match(300s) {:6.2} / {:6.2}  \
             v2b_unit/dec {:.4}  end speed {:6.0}",
            self.transitions,
            cycle_ticks / 120.0,
            100.0 * self.air_decisions as f32 / self.decisions as f32,
            self.flips,
            self.jump_edges,
            per_min * 0.004,
            per_min * 0.008,
            per_min * 0.004 * 5.0,
            per_min * 0.008 * 5.0,
            self.v2b_sum / self.decisions as f32,
            self.speed_end,
        );
    }
}

/// mode: 0 = greedy hop (jump iff on ground at last sample and jump was low last
///          decision -- the fastest legal rising-edge cycle)
///       1 = greedy hop + dodge-cancel (flip once per air phase to kill v_z)
///       2 = blind alternate jump 1,0,1,0,...
///       3 = jump held forever (control: one edge)
///       4 = no jump at all (pure driving / chatter test)
fn run(
    label: &str,
    mode: u32,
    n: u32,
    place: impl FnOnce(&mut cxx::UniquePtr<Arena>, u32),
    base: CarControls,
) -> Res {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    place(&mut arena, id);

    let mut res = Res::default();
    let mut prev = arena.pin_mut().get_game_state();
    let mut last_jump = false;
    let mut flipped_this_air = false;
    for i in 0..n {
        let c = &prev.cars[0].state;
        let on_ground = c.is_on_ground;
        let mut ctl = base;
        match mode {
            0 => ctl.jump = on_ground && !last_jump,
            1 => {
                if on_ground {
                    ctl.jump = !last_jump;
                    flipped_this_air = false;
                } else if !last_jump && !flipped_this_air && !c.has_flipped {
                    ctl.jump = true;
                    ctl.pitch = 1.0; // dodge input: kills upward v_z 0.15 s later
                    flipped_this_air = true;
                }
            }
            2 => ctl.jump = i % 2 == 0,
            3 => ctl.jump = true,
            _ => ctl.jump = false,
        }
        if ctl.jump && !last_jump {
            res.jump_edges += 1;
        }
        last_jump = ctl.jump;
        arena.pin_mut().set_car_controls(id, ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        if was.is_on_ground && !now.is_on_ground {
            res.transitions += 1;
        }
        if now.has_flipped && !was.has_flipped {
            res.flips += 1;
        }
        if !now.is_on_ground {
            res.air_decisions += 1;
        }
        // vel_to_ball unit (reward.rs:623-628), coefficient-free
        let d = [
            cur.ball.pos.x - now.pos.x,
            cur.ball.pos.y - now.pos.y,
            cur.ball.pos.z - now.pos.z,
        ];
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        let proj = (now.vel.x * d[0] + now.vel.y * d[1] + now.vel.z * d[2]) / dist;
        res.v2b_sum += (proj / 2300.0).clamp(0.0, 1.0);
        res.decisions += 1;
        prev = cur;
    }
    let st = arena.pin_mut().get_car(id);
    res.speed_end = (st.vel.x * st.vel.x + st.vel.y * st.vel.y + st.vel.z * st.vel.z).sqrt();
    res.report(label);
    res
}

fn park(x: f32, y: f32, yaw: f32, speed: f32, boost: f32) -> impl Fn(&mut cxx::UniquePtr<Arena>, u32) {
    move |arena: &mut cxx::UniquePtr<Arena>, id: u32| {
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(x, y, 17.0);
        cs.vel = Vec3::new(speed * yaw.cos(), speed * yaw.sin(), 0.0);
        cs.boost = boost;
        cs.rot_mat = Angle { yaw, pitch: 0.0, roll: 0.0 }.to_rotmat();
        arena.pin_mut().set_car(id, cs).unwrap();
        // park the ball far away and high so it never interferes with the car
        let mut ball = arena.pin_mut().get_ball();
        ball.pos = Vec3::new(0.0, 4000.0, 1000.0);
        ball.vel = Vec3::new(0.0, 0.0, 0.0);
        ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(ball);
    }
}

fn main() {
    sim_init::ensure_init(None);
    println!("=== 1. max ground->air transitions per minute (300 s each) ===");
    run("greedy hop, standing start", 0, MATCH_DECISIONS, park(0.0, -3000.0, 1.5708, 0.0, 100.0), CarControls::default());
    run(
        "greedy hop, throttle held",
        0,
        MATCH_DECISIONS,
        park(0.0, -4500.0, 1.5708, 0.0, 100.0),
        CarControls { throttle: 1.0, ..Default::default() },
    );
    run(
        "greedy hop @ 1410 cruise",
        0,
        MATCH_DECISIONS,
        park(-2000.0, -1000.0, 0.0, 1410.0, 100.0),
        CarControls { throttle: 1.0, ..Default::default() },
    );
    run("greedy hop + dodge cancel", 1, MATCH_DECISIONS, park(0.0, -3000.0, 1.5708, 0.0, 100.0), CarControls::default());
    run("blind alternate jump 1,0,1,0", 2, MATCH_DECISIONS, park(0.0, -3000.0, 1.5708, 0.0, 100.0), CarControls::default());
    run("jump held forever (control)", 3, MATCH_DECISIONS, park(0.0, -3000.0, 1.5708, 0.0, 100.0), CarControls::default());

    println!("\n=== 2. wheel-contact chatter: NO jump input at all ===");
    run(
        "powerslide circle @1410 + hb",
        4,
        MATCH_DECISIONS,
        park(-1500.0, 0.0, 0.0, 1410.0, 100.0),
        CarControls { throttle: 1.0, steer: 1.0, handbrake: true, ..Default::default() },
    );
    run(
        "hard turn @1410, no hb",
        4,
        MATCH_DECISIONS,
        park(-1500.0, 0.0, 0.0, 1410.0, 100.0),
        CarControls { throttle: 1.0, steer: 1.0, ..Default::default() },
    );
    run(
        "boost circle into corner",
        4,
        MATCH_DECISIONS,
        park(2000.0, 2000.0, 0.7854, 1410.0, 100.0),
        CarControls { throttle: 1.0, steer: 0.35, boost: true, ..Default::default() },
    );
    run(
        "drive straight into corner wall",
        4,
        MATCH_DECISIONS,
        park(2000.0, 2000.0, 0.7854, 1410.0, 100.0),
        CarControls { throttle: 1.0, boost: true, ..Default::default() },
    );

    println!("\n=== 3. landing bounce: does one hop yield >1 transition? ===");
    for (label, drop_z) in [("drop from z=200", 200.0f32), ("drop from z=600", 600.0), ("drop from z=1500", 1500.0)] {
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -3000.0, drop_z);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_car(id, cs).unwrap();
        let mut prev = arena.pin_mut().get_game_state();
        let mut trans = 0;
        for _ in 0..120 {
            arena.pin_mut().set_car_controls(id, CarControls::default()).unwrap();
            arena.pin_mut().step(SKIP);
            let cur = arena.pin_mut().get_game_state();
            if prev.cars[0].state.is_on_ground && !cur.cars[0].state.is_on_ground {
                trans += 1;
            }
            prev = cur;
        }
        println!("{label:<38} post-landing ground->air transitions in 8 s: {trans}");
    }

    println!("\n=== 4. hop into a RESTING ball: ball dwell above z gates ===");
    {
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -300.0, 17.0);
        cs.vel = Vec3::new(0.0, 1200.0, 0.0);
        cs.boost = 100.0;
        cs.rot_mat = Angle { yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, roll: 0.0 }.to_rotmat();
        arena.pin_mut().set_car(id, cs).unwrap();
        let mut ball = arena.pin_mut().get_ball();
        ball.pos = Vec3::new(0.0, 0.0, 93.15);
        ball.vel = Vec3::new(0.0, 0.0, 0.0);
        ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(ball);
        let (mut above150, mut above300, mut above500, mut maxz) = (0u32, 0u32, 0u32, 0.0f32);
        let n = 150u32; // 10 s
        for i in 0..n {
            // jump into the ball on the way in, then nothing
            arena
                .pin_mut()
                .set_car_controls(id, CarControls { throttle: 1.0, jump: i < 3, ..Default::default() })
                .unwrap();
            arena.pin_mut().step(SKIP);
            let bz = arena.pin_mut().get_ball().pos.z;
            maxz = maxz.max(bz);
            if bz > 150.0 {
                above150 += 1;
            }
            if bz > 300.0 {
                above300 += 1;
            }
            if bz > 500.0 {
                above500 += 1;
            }
        }
        println!(
            "hop-tap a resting ball: ball max z {maxz:.0}  |  dwell of 10 s above 150/300/500 = \
             {:.2}/{:.2}/{:.2} s ({:.0}%/{:.0}%/{:.0}% duty)",
            above150 as f32 / HZ,
            above300 as f32 / HZ,
            above500 as f32 / HZ,
            100.0 * above150 as f32 / n as f32,
            100.0 * above300 as f32 / n as f32,
            100.0 * above500 as f32 / n as f32,
        );
    }
}
