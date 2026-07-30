// ADVERSARIAL PROBE part 4 (do not ship): a per-transition payment is a payment
// per AIR PHASE, so its rate is 1/air_phase_duration. Measure the air-phase
// duration of every takeoff on the v9 tape's own ladder (hop .. full aerial) and
// turn each into transitions/min and reward/match. If the useless takeoff pays
// more per minute than the real one, the term ranks behaviour backwards.
// Run: cargo run --example jump_spam_probe4   (from engine/)
use construct_engine::sim_init;
use rocketsim_rs::math::Vec3;
use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

const SKIP: u32 = 8;
const HZ: f32 = 120.0 / SKIP as f32;

fn main() {
    sim_init::ensure_init(None);
    // (label, tape) -- tapes are the ones configs/reward_v9_aerial.toml's apex
    // figures were measured with, plus the 1-decision minimum hop.
    let tapes: [(&str, &dyn Fn(u32) -> CarControls); 5] = [
        ("minimum hop (jump 1 decision)", &|i: u32| CarControls { jump: i < 1, ..Default::default() }),
        ("full-height hop (jump 3 dec)", &|i: u32| CarControls { jump: i < 3, ..Default::default() }),
        ("double jump", &|i: u32| CarControls { jump: i < 3 || i == 8, ..Default::default() }),
        ("hop + boost", &|i: u32| CarControls { jump: i < 3, boost: i >= 3, ..Default::default() }),
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
    println!("air phase -> transitions/min -> reward/match, one takeoff per phase:");
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
        let mut b = arena.pin_mut().get_ball();
        b.pos = Vec3::new(0.0, 3000.0, 93.15);
        b.vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(b);

        let (mut apex, mut left_at, mut landed_at, mut boost_spent) = (0.0f32, -1i32, -1i32, 0.0f32);
        for i in 0..600u32 {
            arena.pin_mut().set_car_controls(id, tape(i)).unwrap();
            arena.pin_mut().step(SKIP);
            let st = arena.pin_mut().get_car(id);
            apex = apex.max(st.pos.z);
            if left_at < 0 && !st.is_on_ground {
                left_at = i as i32;
            }
            if left_at >= 0 && landed_at < 0 && st.is_on_ground && i as i32 > left_at {
                landed_at = i as i32;
                boost_spent = 100.0 - st.boost;
                break;
            }
        }
        // cycle = decisions from takeoff decision to the decision it is back on the
        // ground and can take off again
        let cyc_dec = (landed_at - left_at + 1).max(1) as f32;
        let per_min = HZ * 60.0 / cyc_dec;
        println!(
            "  {label:<32} apex z {apex:7.1}  air phase {:5.2} s ({:5.1} dec, {:5.0} ticks)  \
             -> {per_min:6.1} transitions/min  reward/match @4e-3 {:5.2} @8e-3 {:5.2}  boost spent {boost_spent:5.1}",
            cyc_dec / HZ,
            cyc_dec,
            cyc_dec * SKIP as f32,
            per_min * 0.004 * 5.0,
            per_min * 0.008 * 5.0,
        );
    }
}
