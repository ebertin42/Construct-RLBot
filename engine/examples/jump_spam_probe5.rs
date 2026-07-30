// ADVERSARIAL PROBE part 5 (do not ship): re-measure the apex ladder the v9 tape
// quotes, with and without the kickoff ball in front of the car. The original probe
// (examples/aerial_reward_probe.rs section 2) parks the car at (0,-2000) facing the
// kickoff ball at (0,0,93.15) and boosts -- so a "hop + boost" apex may be a
// BALL COLLISION, not a boost apex.
// Run: cargo run --example jump_spam_probe5   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::Vec3;
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;

fn main() {
    sim_init::ensure_init(None);
    let tapes: [(&str, &dyn Fn(u32) -> CarControls); 4] = [
        ("single hop (jump 3 dec)", &|i: u32| CarControls { jump: i < 3, ..Default::default() }),
        ("double jump", &|i: u32| CarControls { jump: i < 3 || i == 8, ..Default::default() }),
        ("hop + boost (no rotate)", &|i: u32| CarControls { jump: i < 3, boost: i >= 3, ..Default::default() }),
        (
            "real aerial (jump,pitch,boost)",
            &|i: u32| CarControls {
                jump: i < 3,
                pitch: if (3..9).contains(&i) { 1.0 } else { 0.0 },
                boost: i >= 5,
                ..Default::default()
            },
        ),
    ];
    for ball_away in [false, true] {
        println!("--- ball {} ---", if ball_away { "moved to (0,4000,93) [clean]" } else { "at kickoff (0,0,93) [as originally measured]" });
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
            if ball_away {
                let mut b = arena.pin_mut().get_ball();
                b.pos = Vec3::new(0.0, 4000.0, 93.15);
                b.vel = Vec3::new(0.0, 0.0, 0.0);
                arena.pin_mut().set_ball(b);
            }
            let (mut apex, mut touched_before_apex, mut any_touch, mut apex_i) = (0.0f32, false, false, 0u32);
            let (mut apex_xy, mut apex_ground, mut air_apex, mut air_apex_i) = ((0.0f32, 0.0f32), false, 0.0f32, 0u32);
            let mut landed = false;
            let mut prev = arena.pin_mut().get_game_state();
            for i in 0..120u32 {
                arena.pin_mut().set_car_controls(id, tape(i)).unwrap();
                arena.pin_mut().step(SKIP);
                let cur = arena.pin_mut().get_game_state();
                let (h, hb) = (&cur.cars[0].state.ball_hit_info, &prev.cars[0].state.ball_hit_info);
                if h.is_valid && h.tick_count_when_hit != hb.tick_count_when_hit && h.tick_count_when_hit > prev.tick_count {
                    any_touch = true;
                }
                let st = &cur.cars[0].state;
                let z = st.pos.z;
                if z > apex {
                    apex = z;
                    apex_i = i;
                    apex_xy = (st.pos.x, st.pos.y);
                    apex_ground = st.is_on_ground;
                    touched_before_apex = any_touch;
                }
                if i > 0 && st.is_on_ground {
                    landed = true;
                }
                if !landed && z > air_apex {
                    air_apex = z;
                    air_apex_i = i;
                }
                prev = cur;
            }
            println!(
                "  {label:<32} apex z {apex:7.1} at dec {apex_i:>3} (xy {:6.0},{:6.0} on_ground {apex_ground})  \
                 FIRST-AIR-PHASE apex {air_apex:7.1} at dec {air_apex_i:>3}  touched before apex: {touched_before_apex}  any touch: {any_touch}",
                apex_xy.0, apex_xy.1
            );
        }
    }
}
