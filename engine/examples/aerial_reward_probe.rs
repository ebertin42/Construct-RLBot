// INVESTIGATION PROBE (do not ship). Measures the facts the jump/aerial reward
// design rests on:
//
//   1. Does SUSTAINED car-ball contact re-fire reward.rs's `touched` predicate on
//      every tick_skip step? (i.e. is the flat `touch` weight frequency-farmable
//      by pinning/carrying the ball?)  RocketSim writes
//      `ballHitInfo.tickCountWhenHit = tickCount` inside the per-contact callback
//      with NO cooldown (Ball.cpp:256) while the extra-impulse path right below it
//      DOES have one -- so the source says yes; this measures it, and measures the
//      per-step |d ball_vel| in the same regime (does an impulse-scaled term starve
//      the farm that a flat term feeds?).
//   2. Height and air_time reached by a hop vs a real jump+boost aerial, to
//      calibrate an "is this actually an aerial" gate.
//
// Run: cargo run --release --example aerial_reward_probe   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::{Angle, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8; // schema/v1.toml tick_skip
const HZ: f32 = 120.0 / SKIP as f32;

fn mag(a: &[f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Steps `n` times with fixed controls, counting how often reward.rs's `touched`
/// predicate fires and summing |d ball_vel| on those steps.
fn contact_rate(
    label: &str,
    place: impl FnOnce(&mut cxx::UniquePtr<Arena>, u32),
    ctl: CarControls,
    n: u32,
) {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    place(&mut arena, id);

    let mut prev = arena.pin_mut().get_game_state();
    let (mut fires, mut dv_sum, mut dv_max) = (0u32, 0.0f32, 0.0f32);
    let mut near = 0u32; // steps with the ball within contact range of the car
    for _ in 0..n {
        arena.pin_mut().set_car_controls(id, ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let h = &cur.cars[0].state.ball_hit_info;
        let hb = &prev.cars[0].state.ball_hit_info;
        let touched = h.is_valid
            && h.tick_count_when_hit != hb.tick_count_when_hit
            && h.tick_count_when_hit > prev.tick_count
            && h.tick_count_when_hit <= cur.tick_count;
        if touched {
            fires += 1;
            let dv = [
                cur.ball.vel.x - prev.ball.vel.x,
                cur.ball.vel.y - prev.ball.vel.y,
                cur.ball.vel.z - prev.ball.vel.z,
            ];
            let m = mag(&dv);
            dv_sum += m;
            dv_max = dv_max.max(m);
        }
        let sep = mag(&[
            cur.ball.pos.x - cur.cars[0].state.pos.x,
            cur.ball.pos.y - cur.cars[0].state.pos.y,
            cur.ball.pos.z - cur.cars[0].state.pos.z,
        ]);
        if sep < 185.0 {
            near += 1;
        }
        prev = cur;
    }
    let per_s = fires as f32 * HZ / n as f32;
    println!(
        "[1] {label:<26} touch fired {fires:>3}/{n} steps = {per_s:5.2}/s  |  flat touch=0.5 pays \
         {:5.2}/s  |  mean |dv| on firing steps {:6.1} uu/s (impact {:.3}), max {:6.1}  |  \
         impulse-scaled w=1.0 pays {:5.3}/s  |  in contact range on {near}/{n} steps",
        0.5 * per_s,
        dv_sum / fires.max(1) as f32,
        dv_sum / fires.max(1) as f32 / 2300.0,
        dv_max,
        (dv_sum / 2300.0) * HZ / n as f32,
    );
}

fn main() {
    sim_init::ensure_init(None);

    // (a) ball pinned against the back wall, car driving into it: the ball cannot
    //     escape, so contact is sustained for hundreds of ticks.
    contact_rate(
        "pinned on side wall",
        |arena, id| {
            // ball resting against the +x side wall (wall at 4096, radius 91.25),
            // car nose-to-ball behind it: the ball is trapped, contact is sustained.
            let mut ball = arena.pin_mut().get_ball();
            ball.pos = Vec3::new(4096.0 - 91.25, 0.0, 93.15);
            ball.vel = Vec3::new(0.0, 0.0, 0.0);
            ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(ball);
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(4096.0 - 91.25 - 91.25 - 60.0, 0.0, 17.0);
            cs.vel = Vec3::new(200.0, 0.0, 0.0);
            cs.boost = 100.0;
            cs.rot_mat = Angle { yaw: 0.0, pitch: 0.0, roll: 0.0 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
        },
        CarControls { throttle: 1.0, boost: true, ..Default::default() },
        120,
    );

    // (b) hood carry: ball resting on the roof, car rolling forward at the same
    //     speed. This is the "dribble" a jump-capable policy discovers first.
    contact_rate(
        "hood carry (ground)",
        |arena, id| {
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(0.0, -2000.0, 17.0);
            cs.vel = Vec3::new(0.0, 900.0, 0.0);
            cs.rot_mat = Angle { yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, roll: 0.0 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
            let mut ball = arena.pin_mut().get_ball();
            ball.pos = Vec3::new(0.0, -2000.0, 17.0 + 55.0 + 93.15);
            ball.vel = Vec3::new(0.0, 900.0, 0.0);
            ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(ball);
        },
        CarControls { throttle: 0.62, ..Default::default() },
        120,
    );

    // (c) reference: a single hard strike on a resting ball.
    contact_rate(
        "single 2000 uu/s strike",
        |arena, id| {
            let mut ball = arena.pin_mut().get_ball();
            ball.pos = Vec3::new(0.0, 0.0, 93.15);
            ball.vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(ball);
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(0.0, -400.0, 17.0);
            cs.vel = Vec3::new(0.0, 2000.0, 0.0);
            cs.rot_mat = Angle { yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, roll: 0.0 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
        },
        CarControls { throttle: 1.0, boost: true, ..Default::default() },
        60,
    );

    // (d) FORCED continuous contact: the ball is re-placed in overlap with the car
    //     every step, so a contact manifold exists on every single tick. This is the
    //     decisive test of whether reward.rs's `touched` predicate is
    //     frequency-farmable by holding the ball (a hood/air dribble), as opposed to
    //     the scenarios above where the ball simply escapes.
    {
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -2000.0, 17.0);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_car(id, cs).unwrap();

        let mut prev = arena.pin_mut().get_game_state();
        let (mut fires, mut dv_sum) = (0u32, 0.0f32);
        let n = 120u32;
        for _ in 0..n {
            let c = arena.pin_mut().get_car(id);
            let mut ball = arena.pin_mut().get_ball();
            ball.pos = Vec3::new(c.pos.x, c.pos.y, c.pos.z + 130.0); // overlapping the roof
            ball.vel = Vec3::new(c.vel.x, c.vel.y, 0.0);
            ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(ball);
            arena.pin_mut().set_car_controls(id, CarControls { throttle: 1.0, ..Default::default() }).unwrap();
            arena.pin_mut().step(SKIP);
            let cur = arena.pin_mut().get_game_state();
            let h = &cur.cars[0].state.ball_hit_info;
            let hb = &prev.cars[0].state.ball_hit_info;
            let touched = h.is_valid
                && h.tick_count_when_hit != hb.tick_count_when_hit
                && h.tick_count_when_hit > prev.tick_count
                && h.tick_count_when_hit <= cur.tick_count;
            if touched {
                fires += 1;
                dv_sum += mag(&[
                    cur.ball.vel.x - prev.ball.vel.x,
                    cur.ball.vel.y - prev.ball.vel.y,
                    cur.ball.vel.z - prev.ball.vel.z,
                ]);
            }
            prev = cur;
        }
        println!(
            "[1] FORCED overlap every step  touch fired {fires:>3}/{n} steps = {:5.2}/s  |  \
             flat touch=0.5 pays {:5.2}/s  |  impulse-scaled w=1.0 pays {:5.3}/s",
            fires as f32 * HZ / n as f32,
            0.5 * fires as f32 * HZ / n as f32,
            (dv_sum / 2300.0) * HZ / n as f32,
        );
    }

    // ---- 2. hop vs aerial: apex height, air_time, boost spent ----------------
    // Sequences are hand-written control tapes at tick_skip 8 (one entry = 1 step).
    let tapes: [(&str, &dyn Fn(u32) -> CarControls); 4] = [
        ("single hop", &|i: u32| CarControls { jump: i < 3, ..Default::default() }),
        (
            "double jump (straight up)",
            &|i: u32| CarControls { jump: i < 3 || i == 8, ..Default::default() },
        ),
        (
            "hop + boost (no rotate)",
            &|i: u32| CarControls { jump: i < 3, boost: i >= 3, ..Default::default() },
        ),
        (
            "real aerial (jump,pitch,boost)",
            &|i: u32| CarControls {
                jump: i < 3,
                // rotate nose up for ~0.4 s, then boost along it
                pitch: if (3..9).contains(&i) { 1.0 } else { 0.0 },
                boost: i >= 5,
                ..Default::default()
            },
        ),
    ];
    for (label, tape) in tapes {
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -2000.0, 17.0);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        cs.boost = 100.0;
        arena.pin_mut().set_car(id, cs).unwrap();

        let (mut apex, mut air_at_apex, mut t400, mut t1000, mut boost_at_apex) =
            (0.0f32, 0.0f32, -1.0f32, -1.0f32, 100.0f32);
        for i in 0..120 {
            arena.pin_mut().set_car_controls(id, tape(i)).unwrap();
            arena.pin_mut().step(SKIP);
            let st = arena.pin_mut().get_car(id);
            if st.pos.z > apex {
                apex = st.pos.z;
                air_at_apex = st.air_time;
                boost_at_apex = st.boost;
            }
            if t400 < 0.0 && st.pos.z > 400.0 {
                t400 = st.air_time;
            }
            if t1000 < 0.0 && st.pos.z > 1000.0 {
                t1000 = st.air_time;
            }
        }
        println!(
            "[2] {label:<31} apex z {apex:7.1}  air_time@apex {air_at_apex:.2}s  \
             boost left {boost_at_apex:5.1}  air_time@z=400 {t400:5.2}s  @z=1000 {t1000:5.2}s",
        );
    }

    // ---- 3. wall drive: is_on_ground while riding the wall at height ---------
    {
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        // parked flat ON the +x side wall (x = 4096), nose up the wall
        cs.pos = Vec3::new(4096.0 - 17.0, 0.0, 900.0);
        cs.vel = Vec3::new(0.0, 0.0, 300.0);
        cs.rot_mat = Angle { yaw: 0.0, pitch: std::f32::consts::FRAC_PI_2, roll: -std::f32::consts::FRAC_PI_2 }
            .to_rotmat();
        arena.pin_mut().set_car(id, cs).unwrap();
        for _ in 0..6 {
            arena
                .pin_mut()
                .set_car_controls(id, CarControls { throttle: 1.0, boost: true, ..Default::default() })
                .unwrap();
            arena.pin_mut().step(SKIP);
        }
        let st = arena.pin_mut().get_car(id);
        println!(
            "[3] wall drive: z {:.0}  is_on_ground {}  air_time {:.2}  \
             world_contact {}  (a wall-riding car must NOT read as airborne)",
            st.pos.z, st.is_on_ground, st.air_time, st.world_contact.has_contact
        );
    }
}
