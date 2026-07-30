// ADVERSARIAL PROBE part 2 (do not ship).
//   (a) what does hop-spam COST in vel_to_ball / arrival / steering?
//   (b) does a tilted surface, a dodge-cancel, or a forced handbrake shorten the
//       ground->air cycle below the ~119 ticks part 1 measured?
//   (c) hop-spam UNDER a gate-passing ball: does the rate hold, and where does the
//       ball go (own-goal risk)?
// Run: cargo run --example jump_spam_probe2   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::{Angle, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32;

fn v2b_unit(car_p: Vec3, car_v: Vec3, ball_p: Vec3) -> f32 {
    let d = [ball_p.x - car_p.x, ball_p.y - car_p.y, ball_p.z - car_p.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let proj = (car_v.x * d[0] + car_v.y * d[1] + car_v.z * d[2]) / dist;
    (proj / 2300.0).clamp(0.0, 1.0)
}

/// hop: 0 = never jump, 1 = greedy rising-edge hop spam
/// Returns (v2b_sum, decisions_to_first_touch, transitions, final dist)
fn approach(label: &str, hop: u32, base: CarControls, ball: Vec3, car_xy: (f32, f32), yaw: f32, n: u32) {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    {
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(car_xy.0, car_xy.1, 17.0);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        cs.boost = 100.0;
        cs.rot_mat = Angle { yaw, pitch: 0.0, roll: 0.0 }.to_rotmat();
        arena.pin_mut().set_car(id, cs).unwrap();
        let mut b = arena.pin_mut().get_ball();
        b.pos = ball;
        b.vel = Vec3::new(0.0, 0.0, 0.0);
        b.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(b);
    }
    let mut prev = arena.pin_mut().get_game_state();
    let (mut v2b, mut trans, mut first_touch) = (0.0f32, 0u32, -1i32);
    let mut last_jump = false;
    for i in 0..n {
        let mut ctl = base;
        if hop == 1 {
            ctl.jump = prev.cars[0].state.is_on_ground && !last_jump;
        }
        last_jump = ctl.jump;
        arena.pin_mut().set_car_controls(id, ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        if was.is_on_ground && !now.is_on_ground {
            trans += 1;
        }
        v2b += v2b_unit(now.pos, now.vel, cur.ball.pos);
        let (h, hb) = (&now.ball_hit_info, &was.ball_hit_info);
        if first_touch < 0
            && h.is_valid
            && h.tick_count_when_hit != hb.tick_count_when_hit
            && h.tick_count_when_hit > prev.tick_count
        {
            first_touch = i as i32;
        }
        prev = cur;
    }
    let st = arena.pin_mut().get_car(id);
    let b = arena.pin_mut().get_ball();
    let dist = ((b.pos.x - st.pos.x).powi(2) + (b.pos.y - st.pos.y).powi(2) + (b.pos.z - st.pos.z).powi(2)).sqrt();
    println!(
        "{label:<44} v2b_unit sum {v2b:7.3} (x0.05 = {:6.3} reward over {n} dec)  \
         first touch @dec {first_touch:>4} ({:5.2}s)  trans {trans:>3} (pays {:5.3} @8e-3)  \
         end dist {dist:6.0}  end speed {:6.0}",
        0.05 * v2b,
        if first_touch < 0 { f32::NAN } else { first_touch as f32 / HZ },
        trans as f32 * 0.008,
        (st.vel.x * st.vel.x + st.vel.y * st.vel.y + st.vel.z * st.vel.z).sqrt(),
    );
}

/// Greedy-hop cycle time under a variant control tape / start surface.
fn cycle(label: &str, n: u32, place: impl FnOnce(&mut cxx::UniquePtr<Arena>, u32), base: CarControls, dodge: Option<(f32, f32)>) {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    place(&mut arena, id);
    let mut prev = arena.pin_mut().get_game_state();
    let (mut trans, mut air, mut flips) = (0u32, 0u32, 0u32);
    let mut last_jump = false;
    let mut dodged = false;
    for _ in 0..n {
        let c = &prev.cars[0].state;
        let mut ctl = base;
        if c.is_on_ground {
            ctl.jump = !last_jump;
            dodged = false;
        } else if let Some((pitch, roll)) = dodge {
            if !last_jump && !dodged && !c.has_flipped {
                ctl.jump = true;
                ctl.pitch = pitch;
                ctl.roll = roll;
                dodged = true;
            }
        }
        last_jump = ctl.jump;
        arena.pin_mut().set_car_controls(id, ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        if was.is_on_ground && !now.is_on_ground {
            trans += 1;
        }
        if now.has_flipped && !was.has_flipped {
            flips += 1;
        }
        if !now.is_on_ground {
            air += 1;
        }
        prev = cur;
    }
    let per_min = trans as f32 * HZ * 60.0 / n as f32;
    println!(
        "{label:<44} trans {trans:>4} = {per_min:7.1}/min  cycle {:6.1} ticks  air {:4.1}%  \
         flips {flips:>4}  pay/match @4e-3 {:5.2} @8e-3 {:5.2}",
        if trans > 0 { (n * SKIP) as f32 / trans as f32 } else { f32::INFINITY },
        100.0 * air as f32 / n as f32,
        per_min * 0.004 * 5.0,
        per_min * 0.008 * 5.0,
    );
}

fn park(x: f32, y: f32, yaw: f32, speed: f32) -> impl Fn(&mut cxx::UniquePtr<Arena>, u32) {
    move |arena: &mut cxx::UniquePtr<Arena>, id: u32| {
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(x, y, 17.0);
        cs.vel = Vec3::new(speed * yaw.cos(), speed * yaw.sin(), 0.0);
        cs.boost = 100.0;
        cs.rot_mat = Angle { yaw, pitch: 0.0, roll: 0.0 }.to_rotmat();
        arena.pin_mut().set_car(id, cs).unwrap();
        let mut ball = arena.pin_mut().get_ball();
        ball.pos = Vec3::new(0.0, 4500.0, 1000.0);
        ball.vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(ball);
    }
}

fn main() {
    sim_init::ensure_init(None);

    println!("=== A. cost of hop-spam: straight approach to a resting ball 3000uu away ===");
    let fwd = CarControls { throttle: 1.0, ..Default::default() };
    let fwdb = CarControls { throttle: 1.0, boost: true, ..Default::default() };
    approach("drive only (throttle)", 0, fwd, Vec3::new(0.0, 0.0, 93.15), (0.0, -3000.0), 1.5708, 90);
    approach("drive + hop spam (throttle)", 1, fwd, Vec3::new(0.0, 0.0, 93.15), (0.0, -3000.0), 1.5708, 90);
    approach("drive only (throttle+boost)", 0, fwdb, Vec3::new(0.0, 0.0, 93.15), (0.0, -3000.0), 1.5708, 90);
    approach("drive + hop spam (throttle+boost)", 1, fwdb, Vec3::new(0.0, 0.0, 93.15), (0.0, -3000.0), 1.5708, 90);

    println!("\n=== B. cost of hop-spam: ball 2500uu away at 90 deg (needs steering) ===");
    let turn = CarControls { throttle: 1.0, steer: 1.0, ..Default::default() };
    approach("drive+steer only", 0, turn, Vec3::new(2500.0, -3000.0, 93.15), (0.0, -3000.0), 1.5708, 120);
    approach("drive+steer + hop spam", 1, turn, Vec3::new(2500.0, -3000.0, 93.15), (0.0, -3000.0), 1.5708, 120);

    println!("\n=== C. can anything shorten the ~119-tick hop cycle? (300 s each) ===");
    cycle("flat floor, plain greedy hop", 4500, park(0.0, -3000.0, 1.5708, 0.0), CarControls::default(), None);
    cycle(
        "flat floor, hop + handbrake (18/20 jump rows)",
        4500,
        park(0.0, -3000.0, 1.5708, 0.0),
        CarControls { handbrake: true, ..Default::default() },
        None,
    );
    cycle(
        "hop while cruising (throttle held)",
        4500,
        park(0.0, -4500.0, 1.5708, 0.0),
        CarControls { throttle: 1.0, ..Default::default() },
        None,
    );
    cycle("hop + BACK flip cancel (pitch=-1)", 4500, park(0.0, -3000.0, 1.5708, 0.0), CarControls::default(), Some((-1.0, 0.0)));
    cycle("hop + SIDE dodge cancel (roll=+1)", 4500, park(0.0, -3000.0, 1.5708, 0.0), CarControls::default(), Some((0.0, 1.0)));
    // tilted surface: sit the car on the floor->wall curve in the corner
    cycle(
        "hop on the corner curve (tilted up-dir)",
        4500,
        |arena: &mut cxx::UniquePtr<Arena>, id: u32| {
            let mut cs = arena.pin_mut().get_car(id);
            // just inside the +x/+y corner, nose along the diagonal, tilted into the curve
            cs.pos = Vec3::new(3400.0, 4200.0, 60.0);
            cs.vel = Vec3::new(0.0, 0.0, 0.0);
            cs.boost = 100.0;
            cs.rot_mat = Angle { yaw: 0.7854, pitch: 0.0, roll: -0.6 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
            let mut ball = arena.pin_mut().get_ball();
            ball.pos = Vec3::new(0.0, -4500.0, 1000.0);
            ball.vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(ball);
        },
        CarControls::default(),
        None,
    );
    cycle(
        "hop on the goal-wall ramp seam",
        4500,
        |arena: &mut cxx::UniquePtr<Arena>, id: u32| {
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(0.0, 5050.0, 17.0);
            cs.vel = Vec3::new(0.0, 0.0, 0.0);
            cs.boost = 100.0;
            cs.rot_mat = Angle { yaw: 1.5708, pitch: 0.0, roll: 0.0 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
            let mut ball = arena.pin_mut().get_ball();
            ball.pos = Vec3::new(0.0, -4500.0, 1000.0);
            ball.vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(ball);
        },
        CarControls::default(),
        None,
    );

    println!("\n=== D. hop-spam UNDER a gate-passing ball (ball pinned at z, dist<300) ===");
    for bz in [300.0f32, 500.0] {
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -1000.0, 17.0);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_car(id, cs).unwrap();
        let mut prev = arena.pin_mut().get_game_state();
        let (mut trans, mut touches, mut last_jump) = (0u32, 0u32, false);
        let mut dv_y_sum = 0.0f32;
        let n = 1500u32; // 100 s
        for _ in 0..n {
            // hold the ball at (car.x, car.y+0, bz): the gate's dream state, pinned
            let c = arena.pin_mut().get_car(id);
            let mut b = arena.pin_mut().get_ball();
            b.pos = Vec3::new(c.pos.x, c.pos.y, bz);
            b.vel = Vec3::new(0.0, 0.0, 0.0);
            b.ang_vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(b);
            let jump = c.is_on_ground && !last_jump;
            last_jump = jump;
            arena.pin_mut().set_car_controls(id, CarControls { jump, ..Default::default() }).unwrap();
            arena.pin_mut().step(SKIP);
            let cur = arena.pin_mut().get_game_state();
            let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
            if was.is_on_ground && !now.is_on_ground {
                trans += 1;
            }
            let (h, hb) = (&now.ball_hit_info, &was.ball_hit_info);
            if h.is_valid && h.tick_count_when_hit != hb.tick_count_when_hit && h.tick_count_when_hit > prev.tick_count {
                touches += 1;
                dv_y_sum += cur.ball.vel.y - prev.ball.vel.y;
            }
            prev = cur;
        }
        let per_min = trans as f32 * HZ * 60.0 / n as f32;
        println!(
            "ball pinned at z={bz:<5}  trans {trans:>4} = {per_min:6.1}/min  \
             pay/match @8e-3 {:5.2}  |  touches {touches} (flat touch 0.5 would pay {:6.1}/match)",
            per_min * 0.008 * 5.0,
            touches as f32 * 0.5 * 300.0 / (n as f32 / HZ),
        );
    }
}
