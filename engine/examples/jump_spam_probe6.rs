// ADVERSARIAL PROBE part 6 (do not ship): the state the proposed gate ("ball
// reachable / above") POINTS AT is "stand under a high ball and hop". What do the
// terms ALREADY LIVE pay for exactly that behaviour?
//   vel_to_ball = 0.05 : a hop under a ball ABOVE you has +v_z pointing AT the ball,
//                        and the descent is clamped to 0 rather than charged.
//   air_setup   = 0.3   : potential shaping, so a hop cycle should be a small NET
//                        COST (gamma drag), computed here with the shipped ramp
//                        (z_lo 100, z_hi 1400, dist/2500, gamma 0.9954).
// Run: cargo run --example jump_spam_probe6   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::Vec3;
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32;
const GAMMA: f32 = 0.9954;
const W_SETUP: f32 = 0.3;
const W_V2B: f32 = 0.05;

fn phi(on_ground: bool, car: Vec3, ball: Vec3) -> f32 {
    if on_ground {
        return 0.0;
    }
    let d = [ball.x - car.x, ball.y - car.y, ball.z - car.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let h = ((ball.z - 100.0) / (1400.0 - 100.0)).clamp(0.0, 1.0); // air_setup_z_lo/hi
    h * (1.0 - dist / 2500.0).clamp(0.0, 1.0)
}

fn v2b(car_p: Vec3, car_v: Vec3, ball: Vec3) -> f32 {
    let d = [ball.x - car_p.x, ball.y - car_p.y, ball.z - car_p.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let proj = (car_v.x * d[0] + car_v.y * d[1] + car_v.z * d[2]) / dist;
    W_V2B * (proj / 2300.0).clamp(0.0, 1.0)
}

/// hop: true = greedy hop spam, false = sit still (the control)
fn run(bz: f32, offset: f32, hop: bool, n: u32) {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = Vec3::new(0.0, -1000.0, 17.0);
    cs.vel = Vec3::new(0.0, 0.0, 0.0);
    arena.pin_mut().set_car(id, cs).unwrap();

    let mut prev = arena.pin_mut().get_game_state();
    let (mut trans, mut v2b_sum, mut setup_sum, mut touches) = (0u32, 0.0f32, 0.0f32, 0u32);
    let mut last_jump = false;
    for _ in 0..n {
        let c = arena.pin_mut().get_car(id);
        // pin the ball `offset` uu in front of the car at height bz: gate wide open
        let mut b = arena.pin_mut().get_ball();
        b.pos = Vec3::new(c.pos.x, c.pos.y + offset, bz);
        b.vel = Vec3::new(0.0, 0.0, 0.0);
        b.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(b);
        // recompute prev's PHI against the ball as it is NOW (the pin moves it), so
        // the telescoping is measured the way the shaping would actually see it
        let prev_phi = phi(
            prev.cars[0].state.is_on_ground,
            prev.cars[0].state.pos,
            arena.pin_mut().get_ball().pos,
        );
        let jump = hop && c.is_on_ground && !last_jump;
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
        }
        v2b_sum += v2b(now.pos, now.vel, cur.ball.pos);
        setup_sum += W_SETUP * (GAMMA * phi(now.is_on_ground, now.pos, cur.ball.pos) - prev_phi);
        prev = cur;
    }
    let mins = n as f32 / HZ / 60.0;
    println!(
        "  ball z {bz:5.0} offset {offset:5.0}  {:<9} trans {trans:>4} ({:5.1}/min)  \
         r_vel_to_ball {:6.3}/min  r_air_setup {:+7.3}/min  new term @8e-3 {:6.3}/min  \
         TOTAL {:+6.3}/min = {:+6.2}/match  touches {touches}",
        if hop { "HOP SPAM" } else { "sit still" },
        trans as f32 / mins,
        v2b_sum / mins,
        setup_sum / mins,
        trans as f32 * 0.008 / mins,
        (v2b_sum + setup_sum + trans as f32 * 0.008) / mins,
        5.0 * (v2b_sum + setup_sum + trans as f32 * 0.008) / mins,
    );
}

fn main() {
    sim_init::ensure_init(None);
    println!("=== what the LIVE terms pay a hop-spammer parked under a pinned ball (100 s each) ===");
    for bz in [300.0f32, 500.0, 900.0] {
        for offset in [0.0f32, 400.0] {
            run(bz, offset, false, 1500);
            run(bz, offset, true, 1500);
        }
    }
}
