// ADVERSARIAL PROBE (do not ship). Question: which NON-JUMP events produce a
// `prev.is_on_ground && !cur.is_on_ground` transition when sampled once per
// tick_skip=8 decision -- exactly the granularity reward::compute_terms sees?
//
// Cases: demo respawn (grounded victim, airborne victim), driving off the side
// wall, kickoff/random-reset spawns, and whether the ACTION-level jump edge
// (cur.last_controls.jump && !prev.last_controls.jump) equals the physics edge
// (has_jumped rising) for a real ground jump.
//
// Run: cargo run --example ground_air_edge_probe   (debug, no wheel touched)
use construct_engine::curriculum::{random_reset, RandomStateBounds};
use construct_engine::sampler::Pcg32;
use construct_engine::sim_init;
use rocketsim_rs::math::{Angle, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;

fn park_ball_away(arena: &mut cxx::UniquePtr<Arena>) {
    let mut ball = arena.pin_mut().get_ball();
    ball.pos = Vec3::new(0.0, 4800.0, 1800.0);
    ball.vel = Vec3::new(0.0, 0.0, 0.0);
    ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    arena.pin_mut().set_ball(ball);
}

// ---------------------------------------------------------------- demo respawn
fn demo_case(label: &str, airborne_victim: bool) {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(7));
    park_ball_away(&mut arena);
    let id = arena.pin_mut().get_game_state().cars[0].id;

    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = Vec3::new(0.0, -2000.0, if airborne_victim { 700.0 } else { 17.0 });
    cs.vel = Vec3::new(0.0, 0.0, 0.0);
    cs.rot_mat = Angle { yaw: 1.5708, pitch: 0.0, roll: 0.0 }.to_rotmat();
    arena.pin_mut().set_car(id, cs).unwrap();

    // settle a few decisions so is_on_ground reflects real wheel contact
    let mut prev = arena.pin_mut().get_game_state();
    for _ in 0..4 {
        arena.pin_mut().set_car_controls(id, CarControls::default()).unwrap();
        arena.pin_mut().step(SKIP);
        prev = arena.pin_mut().get_game_state();
    }
    println!(
        "{label}: at demo time  on_ground={} z={:.1}",
        prev.cars[0].state.is_on_ground, prev.cars[0].state.pos.z
    );
    arena.pin_mut().demolish_car(id).unwrap();

    let mut fires = 0;
    for d in 0..80 {
        arena.pin_mut().set_car_controls(id, CarControls::default()).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        let fire = was.is_on_ground && !now.is_on_ground;
        if fire {
            fires += 1;
            println!(
                "  decision {d:>3}: GROUND->AIR FIRES  prev(demoed={} on_ground={} z={:.1})  \
                 cur(demoed={} on_ground={} z={:.1})  jump_edge_action={}  has_jumped_edge={}",
                was.is_demoed, was.is_on_ground, was.pos.z,
                now.is_demoed, now.is_on_ground, now.pos.z,
                now.last_controls.jump && !was.last_controls.jump,
                now.has_jumped && !was.has_jumped,
            );
        }
        prev = cur;
    }
    println!("{label}: total false ground->air fires over the demo+respawn = {fires}\n");
}

// ------------------------------------------------------------------ wall exit
fn wall_case() {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(9));
    park_ball_away(&mut arena);
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    // exact setup from reward.rs::wall_riding_is_not_airborne (which asserts
    // is_on_ground == true here after 6 boosted decisions)
    cs.pos = Vec3::new(4096.0 - 17.0, 0.0, 1065.0);
    cs.rot_mat = Angle { yaw: 0.0, pitch: 0.0, roll: -std::f32::consts::FRAC_PI_2 }.to_rotmat();
    cs.vel = Vec3::new(0.0, 1200.0, 0.0);
    cs.boost = 100.0;
    arena.pin_mut().set_car(id, cs).unwrap();

    // SETTLE: 6 decisions of boost along the wall, exactly like reward.rs's
    // `wall_riding_is_not_airborne`, so is_on_ground reflects real wheel contact
    // on the wall rather than the value `set_car` wrote.
    for _ in 0..6 {
        arena
            .pin_mut()
            .set_car_controls(id, CarControls { throttle: 1.0, boost: true, ..Default::default() })
            .unwrap();
        arena.pin_mut().step(SKIP);
    }
    let mut prev = arena.pin_mut().get_game_state();
    println!(
        "  wall settled: z={:.0} on_ground={} has_jumped={}",
        prev.cars[0].state.pos.z, prev.cars[0].state.is_on_ground, prev.cars[0].state.has_jumped
    );
    let mut fires = 0;
    let mut jump_edges = 0;
    // NO jump input, ever. Drive along the wall, then steer off it.
    for d in 0..90 {
        let ctl = CarControls {
            throttle: 1.0,
            boost: d < 10,
            steer: if d < 10 { 0.0 } else { -1.0 },
            ..Default::default()
        };
        arena.pin_mut().set_car_controls(id, ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        if was.is_on_ground && !now.is_on_ground {
            fires += 1;
            println!(
                "  wall decision {d:>3}: GROUND->AIR FIRES  prev z={:.0} on_ground={}  \
                 cur z={:.0} on_ground={}  has_jumped={} jump_edge_action={}",
                was.pos.z, was.is_on_ground, now.pos.z, now.is_on_ground,
                now.has_jumped, now.last_controls.jump && !was.last_controls.jump
            );
        }
        if now.last_controls.jump && !was.last_controls.jump {
            jump_edges += 1;
        }
        prev = cur;
    }
    println!("wall drive-off: {fires} ground->air fires with {jump_edges} jump presses\n");
}

// --------------------------------------------------------------- corner / ramp
fn corner_case() {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(21));
    park_ball_away(&mut arena);
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = Vec3::new(2200.0, 2200.0, 17.0);
    cs.rot_mat = Angle { yaw: 0.7854, pitch: 0.0, roll: 0.0 }.to_rotmat();
    cs.vel = Vec3::new(1410.0 * 0.7071, 1410.0 * 0.7071, 0.0);
    cs.boost = 100.0;
    arena.pin_mut().set_car(id, cs).unwrap();
    // settle one decision so the latched is_on_ground is physics-derived
    arena
        .pin_mut()
        .set_car_controls(id, CarControls { throttle: 1.0, boost: true, ..Default::default() })
        .unwrap();
    arena.pin_mut().step(SKIP);
    let mut prev = arena.pin_mut().get_game_state();
    let mut fires = 0;
    let mut max_z: f32 = 0.0;
    for _ in 0..200 {
        arena
            .pin_mut()
            .set_car_controls(id, CarControls { throttle: 1.0, boost: true, ..Default::default() })
            .unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        max_z = max_z.max(now.pos.z);
        if was.is_on_ground && !now.is_on_ground {
            fires += 1;
        }
        prev = cur;
    }
    println!(
        "corner climb, boost held, ZERO jump input: {fires} ground->air fires over 200 \
         decisions (13.3 s), max z={max_z:.0}\n"
    );
}

// ---------------------------------------------------------------------- bump
fn bump_case() {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(31));
    park_ball_away(&mut arena);
    let gs = arena.pin_mut().get_game_state();
    let (victim, bumper) = (gs.cars[0].id, gs.cars[1].id);

    let mut v = arena.pin_mut().get_car(victim);
    v.pos = Vec3::new(0.0, 0.0, 17.0);
    v.vel = Vec3::new(0.0, 0.0, 0.0);
    v.rot_mat = Angle { yaw: 0.0, pitch: 0.0, roll: 0.0 }.to_rotmat();
    arena.pin_mut().set_car(victim, v).unwrap();
    let mut b = arena.pin_mut().get_car(bumper);
    b.pos = Vec3::new(0.0, -900.0, 17.0);
    b.vel = Vec3::new(0.0, 1800.0, 0.0); // below supersonic: bump, not demo
    b.rot_mat = Angle { yaw: 1.5708, pitch: 0.0, roll: 0.0 }.to_rotmat();
    b.boost = 0.0;
    arena.pin_mut().set_car(bumper, b).unwrap();

    arena
        .pin_mut()
        .set_all_controls(&[
            (victim, CarControls::default()),
            (bumper, CarControls { throttle: 1.0, ..Default::default() }),
        ])
        .unwrap();
    arena.pin_mut().step(SKIP);
    let mut prev = arena.pin_mut().get_game_state();
    let mut fires = 0;
    for d in 0..40 {
        arena
            .pin_mut()
            .set_all_controls(&[
                (victim, CarControls::default()),
                (bumper, CarControls { throttle: 1.0, ..Default::default() }),
            ])
            .unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let vi = cur.cars.iter().position(|c| c.id == victim).unwrap();
        let wi = prev.cars.iter().position(|c| c.id == victim).unwrap();
        let (was, now) = (&prev.cars[wi].state, &cur.cars[vi].state);
        if was.is_on_ground && !now.is_on_ground {
            fires += 1;
            println!(
                "  bump decision {d:>3}: VICTIM GROUND->AIR FIRES  prev z={:.1} -> cur z={:.1}  \
                 demoed={} has_jumped={} jump_edge_action={}",
                was.pos.z, now.pos.z, now.is_demoed, now.has_jumped,
                now.last_controls.jump && !was.last_controls.jump
            );
        }
        prev = cur;
    }
    println!("bump (victim never presses jump): {fires} ground->air fires\n");
}

// -------------------------------------------------- action edge vs physics edge
fn edge_identity_case() {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(3));
    park_ball_away(&mut arena);
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = Vec3::new(0.0, -2000.0, 17.0);
    cs.vel = Vec3::new(0.0, 0.0, 0.0);
    arena.pin_mut().set_car(id, cs).unwrap();
    let mut prev = arena.pin_mut().get_game_state();

    // decision schedule: hold jump for 3 decisions, release 3, repeat. The
    // physics edge can only happen on the 0->1 boundary.
    let mut n_action_edge = 0;
    let mut n_hasjumped_edge = 0;
    let mut n_ground_air = 0;
    let mut n_agree = 0;
    for d in 0..60 {
        let jump = (d / 3) % 2 == 0;
        arena.pin_mut().set_car_controls(id, CarControls { jump, ..Default::default() }).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        let (was, now) = (&prev.cars[0].state, &cur.cars[0].state);
        let a = now.last_controls.jump && !was.last_controls.jump;
        let p = now.has_jumped && !was.has_jumped;
        let g = was.is_on_ground && !now.is_on_ground;
        n_action_edge += a as u32;
        n_hasjumped_edge += p as u32;
        n_ground_air += g as u32;
        n_agree += (a == p) as u32;
        if a || p || g {
            println!(
                "  d{d:>3} jump_in={jump}  action_edge={a}  has_jumped_edge={p}  ground_air={g}  \
                 prev(on_ground={} last_jump={})  cur(on_ground={} last_jump={} z={:.0})",
                was.is_on_ground, was.last_controls.jump,
                now.is_on_ground, now.last_controls.jump, now.pos.z
            );
        }
        prev = cur;
    }
    println!(
        "edge identity over 60 decisions: action_edges={n_action_edge} \
         has_jumped_edges={n_hasjumped_edge} ground_air={n_ground_air} agree={n_agree}/60\n"
    );
}

// --------------------------------------------------------------- spawn latching
fn spawn_case() {
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());

    // kickoff: what does the latched (pre-tick) state say?
    arena.pin_mut().reset_to_random_kickoff(Some(11));
    let gs = arena.pin_mut().get_game_state();
    for c in &gs.cars {
        println!(
            "kickoff latch: car {} z={:.1} on_ground={} has_jumped={} last_jump={}",
            c.id, c.state.pos.z, c.state.is_on_ground, c.state.has_jumped, c.state.last_controls.jump
        );
    }

    // random_reset: how many cars latch airborne, and does decision 1 fire?
    let b = RandomStateBounds::default();
    let mut rng = Pcg32::new(1234);
    let mut air_latched = 0;
    let mut total = 0;
    let mut first_dec_fires = 0;
    for _ in 0..200 {
        arena.pin_mut().reset_to_random_kickoff(Some(5));
        random_reset(arena.pin_mut(), &mut rng, &b);
        let prev = arena.pin_mut().get_game_state(); // == episode.rs prev_state latch
        for c in &prev.cars {
            total += 1;
            if !c.state.is_on_ground {
                air_latched += 1;
            }
        }
        let ids: Vec<u32> = prev.cars.iter().map(|c| c.id).collect();
        let ctl: Vec<(u32, CarControls)> =
            ids.iter().map(|&i| (i, CarControls { throttle: 1.0, ..Default::default() })).collect();
        arena.pin_mut().set_all_controls(&ctl).unwrap();
        arena.pin_mut().step(SKIP);
        let cur = arena.pin_mut().get_game_state();
        for (w, n) in prev.cars.iter().zip(cur.cars.iter()) {
            if w.state.is_on_ground && !n.state.is_on_ground {
                first_dec_fires += 1;
            }
        }
    }
    println!(
        "random_reset: {air_latched}/{total} cars latch is_on_ground=false ({:.1}%); \
         first-decision ground->air fires (no jump input) = {first_dec_fires}\n",
        100.0 * air_latched as f32 / total as f32
    );
}

fn main() {
    sim_init::ensure_init(None);
    println!("=== demo respawn ===");
    demo_case("grounded victim", false);
    demo_case("airborne victim", true);
    println!("=== wall drive-off (zero jump input) ===");
    wall_case();
    println!("=== corner climb (zero jump input) ===");
    corner_case();
    println!("=== bump (zero jump input by the victim) ===");
    bump_case();
    println!("=== action edge vs physics edge ===");
    edge_identity_case();
    println!("=== spawn latching ===");
    spawn_case();
}
