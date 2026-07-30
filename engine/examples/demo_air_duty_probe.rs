// THROWAWAY REVIEW PROBE (2026-07-30). The air-hang diff's new duty instrument
// `air_decisions` counts `!is_on_ground` unconditionally, with no demo guard --
// the neighbouring `boost_pickup` block needs one and the hang crossing argues
// explicitly why it does not. Question: what does a DEMOED car read for
// `is_on_ground`, and does a demo therefore inject fake airborne decisions?
//
// Run: setarch -R cargo run --example demo_air_duty_probe   (from engine/)
use construct_engine::reward::{compute_terms, RewardConfig, N_TERMS};
use construct_engine::sim_init::ensure_init;
use rocketsim_rs::math::{RotMat, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

/// air_decisions index is version-dependent; find it by NAME so this compiles
/// only against the new tree (the pre-change tree has no such term).
fn air_dec_idx() -> usize {
    construct_engine::reward::TERM_NAMES
        .iter()
        .position(|n| *n == "air_decisions")
        .expect("air_decisions")
}

fn run(label: &str, airborne_when_demoed: bool) {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(1));
    let id = arena.pin_mut().get_game_state().cars[0].id;
    let mut cs = arena.pin_mut().get_car(id);
    cs.pos = Vec3::new(0.0, -2000.0, if airborne_when_demoed { 900.0 } else { 17.0 });
    cs.vel = Vec3::new(0.0, 0.0, 0.0);
    cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    cs.rot_mat = RotMat {
        forward: Vec3::new(0.0, 1.0, 0.0),
        right: Vec3::new(-1.0, 0.0, 0.0),
        up: Vec3::new(0.0, 0.0, 1.0),
    };
    arena.pin_mut().set_car(id, cs).unwrap();
    // Park the ball far away so touch terms stay out of it.
    let mut b = arena.pin_mut().get_ball();
    b.pos = Vec3::new(0.0, 4000.0, 93.15);
    b.vel = Vec3::new(0.0, 0.0, 0.0);
    arena.pin_mut().set_ball(b);
    arena.pin_mut().step(8);

    arena.pin_mut().demolish_car(id).unwrap();
    let cfg = RewardConfig::load("../configs/reward_v9_aerial.toml").unwrap();
    let ai = air_dec_idx();
    let mut t = [0.0f64; N_TERMS];
    let mut prev = arena.pin_mut().get_game_state();
    let (mut demoed_decs, mut demoed_air) = (0u32, 0u32);
    for _ in 0..90u32 {
        arena.pin_mut().set_car_controls(id, CarControls::default()).unwrap();
        arena.pin_mut().step(8);
        let cur = arena.pin_mut().get_game_state();
        let before = t[ai];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        let st = cur.cars[0].state;
        if st.is_demoed {
            demoed_decs += 1;
            if t[ai] > before {
                demoed_air += 1;
            }
        }
        prev = cur;
    }
    println!(
        "{label:<28} demoed_decisions={demoed_decs:<3} \
         of which counted as AIRBORNE={demoed_air:<3} total_air_decisions={}",
        t[ai]
    );
}

fn main() {
    run("demoed while GROUNDED", false);
    run("demoed while AIRBORNE", true);
}
