// INVESTIGATION PROBE (do not ship). By-construction demonstration that a
// finite-but-insane physics state (a blowup-ramp precursor) passes the collect
// path's `state_is_finite` containment, is written into obs, is paired with a
// reward, is NOT flagged terminated/truncated on the step it appears, and
// PERSISTS across many steps (a latch) until the unrelated no-touch truncation.
//
// Run: cargo run --release --example blowup_probe   (from engine/)
use construct_engine::episode::{state_is_sane, EpisodeArena, StepFlags};
use construct_engine::obs::OBS_SIZE;
use construct_engine::reward::RewardConfig;
use construct_engine::schema::Schema;
use construct_engine::sim_init;

fn max_abs(o: &[f32]) -> f32 {
    o.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}
fn all_finite(o: &[f32]) -> bool {
    o.iter().all(|x| x.is_finite())
}

fn main() {
    sim_init::ensure_init(None);
    // Load exactly what the live run-B trainer uses (reward_v2), from repo root.
    let sch = Schema::load("../schema/v0.toml").expect("schema");
    let cfg = RewardConfig::load("../configs/reward_v2.toml").expect("reward cfg");
    let pk = sch.normalization.pos_norm as f32;

    // 2v2 arena (multi-body squeeze regime).
    let mut arena = EpisodeArena::new(2, 2, sch.tick_skip, cfg, sch.normalization.clone(), 42);
    let n = arena.num_agents();

    // Inject a HUGE-BUT-FINITE "levitating ball" blowup: in-field x/y (so it does
    // NOT cross a goal line and score), but z ~ 5e8 uu (field ceiling ~2e3) rising
    // fast. Every component is finite (f32, well under 3.4e38), so `state_is_finite`
    // -- the only collect-path guard -- accepts it, while `state_is_sane` rejects it.
    arena.debug_place_ball([2000.0, 0.0, 5.0e8], [0.0, 0.0, 5.0e6]);

    let acts: Vec<i64> = vec![0; n]; // idle-ish action for all agents
    let mut rewards = vec![0.0f32; n];
    let mut flags = vec![StepFlags::default(); n];
    let mut final_obs = vec![0.0f32; n * OBS_SIZE];
    let mut obs = vec![0.0f32; n * OBS_SIZE];

    println!("agents={n}  pos_norm={pk:e}");
    println!("step |  ball.pos (uu)                    | ball.z(uu) | max|obs| | finite | sane | rew[0]     | term trunc");

    let mut first_term_or_trunc: Option<usize> = None;
    let mut poisoned_transitions = 0usize;
    let steps = 600usize; // enough to reach no-touch truncation (~450 steps @ tickskip8)
    for s in 0..steps {
        arena.step(&acts, &mut rewards, &mut flags, &mut final_obs);
        arena.write_obs(&mut obs);
        let gs = arena.game_state();
        let bp = gs.ball.pos;
        let finite = all_finite(&obs) && bp.x.is_finite() && bp.y.is_finite() && bp.z.is_finite();
        let sane = state_is_sane(&gs);
        let mo = max_abs(&obs);
        let term = flags[0].terminated;
        let trunc = flags[0].truncated;

        // A "poisoned transition" recorded into the PPO buffer = an insane-but-finite
        // state that is neither terminated nor truncated this step.
        if finite && !sane && !term && !trunc {
            poisoned_transitions += 1;
        }
        if (term || trunc) && first_term_or_trunc.is_none() {
            first_term_or_trunc = Some(s);
        }

        if s < 6 || (term || trunc) || s % 100 == 0 {
            println!(
                "{s:4} | [{:>11.2e},{:>11.2e},{:>11.2e}] | {:>10.2e} | {:>8.1e} | {finite:^6} | {sane:^4} | {:>10.3e} | {term:^4} {trunc:^5}",
                bp.x, bp.y, bp.z, bp.z, mo, rewards[0]
            );
        }
        if term || trunc {
            // step() has already reset the arena; the latch is over.
            println!("--- reset fired at step {s} (term={term} trunc={trunc}); ball now |z|={:.1}", arena.game_state().ball.pos.z);
            break;
        }
    }

    println!();
    println!("SUMMARY");
    println!("  poisoned (finite && !sane && !term && !trunc) transitions recorded: {poisoned_transitions}");
    match first_term_or_trunc {
        Some(s) => println!("  latch persisted {} steps before first term/trunc reset (@ tick_skip {}, ~{} game-ticks)", s + 1, sch.tick_skip, (s + 1) as u32 * sch.tick_skip),
        None => println!("  latch NEVER self-reset within {steps} steps (still finite-but-insane)"),
    }
}
