// ADVERSARIAL PROBE part 3 (do not ship): tilted-launch attempts. A jump impulse
// along the car's up-dir gives hang time ~ 2*v_z/g, so a TILTED stance should
// shorten the ground->air cycle below the flat-floor 114-119 ticks. Does any
// reachable stance actually do it?
// Run: cargo run --example jump_spam_probe3   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::{Angle, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32;

fn spam(label: &str, n: u32, base: CarControls, place: impl FnOnce(&mut cxx::UniquePtr<Arena>, u32)) {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    place(&mut arena, id);
    let mut prev = arena.pin_mut().get_game_state();
    let (mut trans, mut air, mut ground, mut maxz) = (0u32, 0u32, 0u32, 0.0f32);
    let mut last_jump = false;
    for _ in 0..n {
        let c = &prev.cars[0].state;
        let mut ctl = base;
        ctl.jump = c.is_on_ground && !last_jump;
        last_jump = ctl.jump;
        arena.pin_mut().set_car_controls(id, ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        if was.is_on_ground && !now.is_on_ground {
            trans += 1;
        }
        if now.is_on_ground {
            ground += 1;
        } else {
            air += 1;
        }
        maxz = maxz.max(now.pos.z);
        prev = cur;
    }
    let per_min = trans as f32 * HZ * 60.0 / n as f32;
    println!(
        "{label:<40} trans {trans:>4} = {per_min:6.1}/min  cycle {:6.1} ticks  \
         ground {:4.1}% air {:4.1}%  max z {maxz:6.0}  pay/match @8e-3 {:5.2}",
        if trans > 0 { (n * SKIP) as f32 / trans as f32 } else { f32::INFINITY },
        100.0 * ground as f32 / n as f32,
        100.0 * air as f32 / n as f32,
        per_min * 0.008 * 5.0,
    );
}

fn far_ball(arena: &mut cxx::UniquePtr<Arena>) {
    let mut b = arena.pin_mut().get_ball();
    b.pos = Vec3::new(0.0, 0.0, 1500.0);
    b.vel = Vec3::new(0.0, 0.0, 0.0);
    b.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    arena.pin_mut().set_ball(b);
}

fn main() {
    sim_init::ensure_init(None);
    let n = 1500; // 100 s each

    // 1. drive INTO the side wall while hop-spamming: the car rides up the
    //    floor->wall curve, so its up-dir tilts and the jump impulse loses its
    //    vertical component -> shorter hang time?
    spam(
        "hop + drive into side wall",
        n,
        CarControls { throttle: 1.0, ..Default::default() },
        |arena, id| {
            far_ball(arena);
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(3600.0, 0.0, 17.0);
            cs.vel = Vec3::new(1000.0, 0.0, 0.0);
            cs.boost = 100.0;
            cs.rot_mat = Angle { yaw: 0.0, pitch: 0.0, roll: 0.0 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
        },
    );
    spam(
        "hop + BOOST into side wall",
        n,
        CarControls { throttle: 1.0, boost: true, ..Default::default() },
        |arena, id| {
            far_ball(arena);
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(3600.0, 0.0, 17.0);
            cs.vel = Vec3::new(1000.0, 0.0, 0.0);
            cs.boost = 100.0;
            cs.rot_mat = Angle { yaw: 0.0, pitch: 0.0, roll: 0.0 }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
        },
    );
    // 2. start ON the wall at height, tilted 90deg: every jump leaves the surface
    //    and the car falls all the way to the floor -- the worst case, for contrast.
    spam("hop while riding the side wall", n, CarControls { throttle: 1.0, ..Default::default() }, |arena, id| {
        far_ball(arena);
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(4096.0 - 17.0, 0.0, 900.0);
        cs.vel = Vec3::new(0.0, 0.0, 600.0);
        cs.boost = 100.0;
        cs.rot_mat = Angle { yaw: 0.0, pitch: std::f32::consts::FRAC_PI_2, roll: -std::f32::consts::FRAC_PI_2 }.to_rotmat();
        arena.pin_mut().set_car(id, cs).unwrap();
    });
    // 3. sit ON the floor->wall curve (tilted stance, throttle into the curve)
    for (label, x, z, roll) in [
        ("hop on curve, roll -0.3", 3900.0f32, 40.0f32, -0.3f32),
        ("hop on curve, roll -0.6", 3960.0, 80.0, -0.6),
        ("hop on curve, roll -0.9", 4010.0, 140.0, -0.9),
    ] {
        spam(label, n, CarControls { throttle: 0.6, ..Default::default() }, move |arena, id| {
            far_ball(arena);
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(x, 0.0, z);
            cs.vel = Vec3::new(0.0, 300.0, 0.0);
            cs.boost = 100.0;
            cs.rot_mat = Angle { yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, roll }.to_rotmat();
            arena.pin_mut().set_car(id, cs).unwrap();
        });
    }
    // 4. control: flat floor, same duration
    spam("flat floor control", n, CarControls::default(), |arena, id| {
        far_ball(arena);
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -3000.0, 17.0);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        cs.boost = 100.0;
        arena.pin_mut().set_car(id, cs).unwrap();
    });
}
