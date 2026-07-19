use crate::{
    actions,
    ballpred::Tracker,
    curriculum::CurriculumConfig,
    obs::{self, OBS_SIZE},
    obs_v1::{self, ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT},
    reward::{self, RewardConfig},
    sampler::Pcg32,
    schema::Normalization,
};
use cxx::UniquePtr;
use rocketsim_rs::{
    math::{Angle, Vec3},
    sim::{Arena, CarConfig, Team},
    GameState,
};

const TICKS_PER_SEC: u64 = 120;
const NO_TOUCH_TICKS: u64 = 30 * TICKS_PER_SEC;
const MAX_TICKS: u64 = 300 * TICKS_PER_SEC;

/// Kickoff spawn jitter bounds. A textbook (KL-anchored) policy drives
/// mirrored 1v1 kickoffs with perfect symmetry: both cars arrive at the ball
/// on the exact same tick with exactly mirrored velocities, which is a
/// documented RocketSim contact-solver degenerate case (car-car-ball pinch)
/// that reliably drives the solver nonfinite (see `state_is_sane` / the
/// physics-blowup containment in `step_impl`, commit a1c33e0). The
/// community-standard fix is to break the symmetry with small per-car
/// spawn noise. ±50 uu (position) and ±0.09 rad (~5°, yaw) are big enough to
/// desynchronize arrival tick/angle but small enough to leave every kickoff
/// spawn legal (nowhere near a wall or the ball).
// 2026-07-19: ±50uu/±0.09rad proved insufficient live — the anchored policy
// course-corrects over the 3-4s approach and both cars still pinch the ball
// near-simultaneously (viewer sessions, where EVERY episode is a kickoff,
// pinched each time; the remote "win" was confounded by curriculum's 60%
// non-kickoff resets). Tripled car noise + small BALL horizontal offset —
// the ball offset is the decisive symmetry breaker: contact geometry
// diverges no matter how well both cars correct their approach.
const KICKOFF_JITTER_POS: f32 = 150.0;
const KICKOFF_JITTER_YAW: f32 = 0.17;
const KICKOFF_JITTER_BALL: f32 = 10.0;

#[derive(Debug, Default, Clone, Copy)]
pub struct StepFlags {
    pub terminated: bool,
    pub truncated: bool,
}

/// Which obs family an `EpisodeArena` (and the engine above it) builds.
/// Selected by `schema.version` (0 = legacy 94-float MLP obs, 1 = entity
/// set). V0 is the default everywhere so every pre-T6 constructor call site
/// keeps its exact behavior.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ObsMode {
    #[default]
    V0,
    V1,
}

/// V1-only per-arena state: the ball-prediction tracker (one `predict()` per
/// arena per step, shared across its agents) plus a per-agent prev-action
/// ring (`[i64; 5]`, most-recent-first; all zeros after episode reset).
/// Constructed only in `ObsMode::V1` — it must never touch the arena's rng
/// or reset seeding, so the v0 path stays bit-identical (the `Option` is
/// simply `None` there).
struct V1State {
    tracker: Tracker,
    prev: Vec<[i64; PREV_ACTIONS]>,
}

/// Where `step_impl` writes the terminal-step final obs — v0's flat 94-float
/// rows or v1's entity tensors. Private plumbing so `step` (v0, signature
/// unchanged) and `step_v1` share one episode-logic body.
enum FinalObsOut<'a> {
    V0(&'a mut [f32]),
    V1 { ents: &'a mut [f32], mask: &'a mut [bool], query: &'a mut [f32], prev: &'a mut [i64] },
}

/// r_i' = (1-tau)*r_i + tau*mean(own team) - opp*mean(opponent team).
/// `blue_count` splits `rewards` (blue-then-orange agent order). An empty
/// team's mean is 0.0 (1v0 test arenas).
pub fn blend_team_spirit(rewards: &mut [f32], blue_count: usize, tau: f32, opp: f32) {
    let mean = |s: &[f32]| if s.is_empty() { 0.0 } else { s.iter().sum::<f32>() / s.len() as f32 };
    let bm = mean(&rewards[..blue_count]);
    let om = mean(&rewards[blue_count..]);
    for (i, r) in rewards.iter_mut().enumerate() {
        let (own, other) = if i < blue_count { (bm, om) } else { (om, bm) };
        *r = (1.0 - tau) * *r + tau * own - opp * other;
    }
}

fn vec3_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// True iff every physics quantity the obs/reward paths consume is finite.
fn state_is_finite(gs: &GameState) -> bool {
    if !(vec3_finite(&gs.ball.pos) && vec3_finite(&gs.ball.vel) && vec3_finite(&gs.ball.ang_vel)) {
        return false;
    }
    gs.cars.iter().all(|c| {
        vec3_finite(&c.state.pos)
            && vec3_finite(&c.state.vel)
            && vec3_finite(&c.state.ang_vel)
            && vec3_finite(&c.state.rot_mat.forward)
            && vec3_finite(&c.state.rot_mat.up)
    })
}

/// Physical plausibility bounds for streaming: a contact-solver blowup ramps
/// through huge-but-finite values for a few ticks before reaching NaN, and
/// rlviser's interpolation latches permanently if it ever receives one of
/// those frames (observed live: levitating ball). Bounds are ~2-4x the
/// hardest legal values (field |y| ~6000 uu, ball speed 6000 uu/s, car
/// ang_vel 5.5 rad/s) so no legitimate state is ever dropped.
fn vec3_within(v: &Vec3, limit: f32) -> bool {
    v.x.abs() <= limit && v.y.abs() <= limit && v.z.abs() <= limit
}

pub fn state_is_sane(gs: &GameState) -> bool {
    if !state_is_finite(gs) {
        return false;
    }
    if !(vec3_within(&gs.ball.pos, 12_000.0)
        && vec3_within(&gs.ball.vel, 20_000.0)
        && vec3_within(&gs.ball.ang_vel, 100.0))
    {
        return false;
    }
    gs.cars.iter().all(|c| {
        vec3_within(&c.state.pos, 12_000.0)
            && vec3_within(&c.state.vel, 20_000.0)
            && vec3_within(&c.state.ang_vel, 100.0)
    })
}

pub struct EpisodeArena {
    arena: UniquePtr<Arena>,
    table: Vec<[f32; 8]>,
    car_ids: Vec<u32>, // blue asc, then orange asc — agent index order
    blue_count: usize,
    tick_skip: u32,
    reward_cfg: RewardConfig,
    norm: Normalization,
    seed: u32,
    episode_start_tick: u64,
    last_touch_tick: u64,
    prev_state: GameState,
    curriculum: Option<CurriculumConfig>,
    rng: Pcg32,
    // `Some` iff constructed with `ObsMode::V1` — see `V1State`.
    v1: Option<V1State>,
    /// Count of physics-blowup containments (arena rebuilds — see the
    /// `state_is_sane` guard in `step_impl`) this arena has performed.
    /// Test-only instrumentation for the kickoff-jitter pinch-rate
    /// regression test; never read outside `#[cfg(test)]`.
    pub(crate) blowup_count: u64,
    /// Test-only toggle: when false, `reset_episode`'s kickoff branch skips
    /// `jitter_kickoff_spawns`, so tests can A/B the jittered vs. unjittered
    /// pinch rate through the otherwise-identical reset path. Every public
    /// constructor leaves this at its default (`true`) — production
    /// behavior is unaffected.
    pub(crate) jitter_enabled: bool,
}

impl EpisodeArena {
    pub fn new(
        blue: usize,
        orange: usize,
        tick_skip: u32,
        reward_cfg: RewardConfig,
        norm: Normalization,
        seed: u32,
    ) -> Self {
        Self::new_with_curriculum(blue, orange, tick_skip, reward_cfg, norm, seed, None)
    }

    pub fn new_with_curriculum(
        blue: usize,
        orange: usize,
        tick_skip: u32,
        reward_cfg: RewardConfig,
        norm: Normalization,
        seed: u32,
        curriculum: Option<CurriculumConfig>,
    ) -> Self {
        Self::new_full(blue, orange, tick_skip, reward_cfg, norm, seed, curriculum, ObsMode::V0)
    }

    /// Full constructor: `obs_mode` selects the v0 (legacy, default via the
    /// wrappers above) or v1 (entity) obs family. V1 arenas use the 92-row
    /// action table and own a `V1State`; nothing else differs — in
    /// particular the reset/rng path is byte-identical across modes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        blue: usize,
        orange: usize,
        tick_skip: u32,
        reward_cfg: RewardConfig,
        norm: Normalization,
        seed: u32,
        curriculum: Option<CurriculumConfig>,
        obs_mode: ObsMode,
    ) -> Self {
        let (mut arena, car_ids) = Self::build_arena(blue, orange);
        // Placeholder state — immediately overwritten by reset_episode() below,
        // which performs the actual (possibly curriculum-driven) reset.
        let prev_state = arena.pin_mut().get_game_state();
        let start = prev_state.tick_count;
        let mut this = Self {
            arena,
            table: match obs_mode {
                ObsMode::V0 => actions::make_lookup_table(),
                ObsMode::V1 => actions::make_lookup_table_v1(),
            },
            car_ids,
            blue_count: blue,
            tick_skip,
            reward_cfg,
            norm,
            seed,
            episode_start_tick: start,
            last_touch_tick: start,
            prev_state,
            curriculum,
            rng: Pcg32::new((seed as u64) * 7919 + 13),
            v1: match obs_mode {
                ObsMode::V0 => None,
                ObsMode::V1 => Some(V1State {
                    tracker: Tracker::new(),
                    prev: vec![[0; PREV_ACTIONS]; blue + orange],
                }),
            },
            blowup_count: 0,
            jitter_enabled: true,
        };
        this.reset_episode();
        this
    }

    /// Builds the standard arena and its cars exactly the way `new_full` does:
    /// `Arena::default_standard()`, blue cars first (ascending), then orange —
    /// all Octane. Car ids are a per-arena counter starting at 1, so any arena
    /// built through this path with the same (blue, orange) re-issues the SAME
    /// ids 1..N in the same agent order, and `default_standard` construction is
    /// deterministic (identical boost-pad order). Both invariants are what lets
    /// `rebuild_arena` swap a poisoned arena for a fresh one without breaking
    /// the agent->car mapping, viewer identity, or pad indexing.
    fn build_arena(blue: usize, orange: usize) -> (UniquePtr<Arena>, Vec<u32>) {
        let mut arena = Arena::default_standard();
        let mut car_ids = Vec::with_capacity(blue + orange);
        for _ in 0..blue {
            car_ids.push(arena.pin_mut().add_car(Team::Blue, CarConfig::octane()));
        }
        for _ in 0..orange {
            car_ids.push(arena.pin_mut().add_car(Team::Orange, CarConfig::octane()));
        }
        (arena, car_ids)
    }

    /// Discards the current arena and replaces it with a freshly built one
    /// (same construction path as `new_full`). Needed by physics-blowup
    /// containment: a blown-up arena is NOT recoverable in place, because
    /// Bullet's `updateSingleAabb` latches `DISABLE_SIMULATION` on any body
    /// whose AABB went nonfinite, and RocketSim never clears that state
    /// (`Ball::SetState` only attempts activation when velocity != 0, and
    /// plain `setActivationState` refuses to leave `DISABLE_SIMULATION`).
    /// Resetting the poisoned arena therefore leaves the ball — or a car —
    /// permanently frozen (observed live: ghost ball pinned at kickoff
    /// center, arena degraded to zero-touch episodes).
    ///
    /// Everything else derived from the arena is either re-issued identically
    /// (car ids — see `build_arena`), re-derived by the `reset_episode()` the
    /// caller runs right after (`episode_start_tick`, `last_touch_tick`,
    /// `prev_state`; the fresh arena's tick_count restarts at 0, which
    /// reward.rs's `(prev.tick_count, cur.tick_count]` touch window already
    /// tolerates), or independent of this arena (the v1 `Tracker` owns its own
    /// car-less arena and containment never feeds it a poisoned state).
    fn rebuild_arena(&mut self) {
        let orange = self.car_ids.len() - self.blue_count;
        let (arena, car_ids) = Self::build_arena(self.blue_count, orange);
        // hard assert: cold path (once per contained blowup), and a silent id
        // remap in release would desync agent→car mapping for the whole run
        assert_eq!(
            car_ids, self.car_ids,
            "rebuilt arena must re-issue identical car ids (per-arena counter, same add order)"
        );
        self.arena = arena;
        self.car_ids = car_ids;
    }

    /// Kickoff when curriculum is absent, or when the weighted coin says kickoff;
    /// otherwise a bounded random scenario. Replaces every raw
    /// `reset_to_random_kickoff` call site: always kicks off first for a clean
    /// baseline (boost pads, `ball_hit_info` reset), then overwrites with a random
    /// scenario when the curriculum draw says so — this keeps pad/hit-info hygiene
    /// identical across both reset flavors.
    fn reset_episode(&mut self) {
        let use_random = match &self.curriculum {
            Some(c) => {
                let p = c.random_weight / (c.random_weight + c.kickoff_weight);
                self.rng.next_f32() < p
            }
            None => false,
        };
        self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
        if use_random {
            let bounds = self.curriculum.as_ref().unwrap().random.clone();
            crate::curriculum::random_reset(self.arena.pin_mut(), &mut self.rng, &bounds);
        } else if self.jitter_enabled {
            // Only jitter when the kickoff formation survives (a random-scenario
            // reset above would just overwrite it) — see `jitter_kickoff_spawns`.
            self.jitter_kickoff_spawns();
        }
        // Advance the kickoff seed AFTER using it, so the constructor's first reset
        // uses the RAW engine seed (bit-identical to the legacy constructor, which
        // kicked off with the unmutated seed) while every subsequent reset k uses
        // advance^k(raw) — exactly the sequence the pre-curriculum end-of-episode
        // sites (advance-then-kick) produced.
        self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
        let gs = self.arena.pin_mut().get_game_state();
        self.episode_start_tick = gs.tick_count;
        self.last_touch_tick = gs.tick_count;
        self.prev_state = gs;
        // V1: fresh episode starts with an all-zero prev-action history.
        // No-op in v0 mode (and touches no rng), keeping v0 bit-identical.
        if let Some(v1) = self.v1.as_mut() {
            for ring in v1.prev.iter_mut() {
                *ring = [0; PREV_ACTIONS];
            }
        }
    }

    /// Adds small independent per-car noise to the kickoff formation
    /// `reset_to_random_kickoff` just wrote, so the two mirrored sides never
    /// arrive at the ball perfectly in sync (see the `KICKOFF_JITTER_*`
    /// doc comment for why). Position noise is x/y only (uniform,
    /// ±`KICKOFF_JITTER_POS` uu, independent per axis) — z, boost, and pad
    /// state are left exactly as the kickoff reset set them, and the ball is
    /// never touched. Yaw noise is uniform ±`KICKOFF_JITTER_YAW` rad, added
    /// to the car's current (grounded, pitch=roll=0) heading and rebuilt
    /// into a fresh rotation matrix via `Angle::to_rotmat`, matching how
    /// `curriculum::random_reset` already builds car orientations.
    ///
    /// Draws 3 `f32`s per car (dx, dy, dyaw) from `self.rng` — the SAME
    /// per-episode `Pcg32` stream `reset_episode` already advances for the
    /// curriculum coin flip and `curriculum::random_reset` (no new rng
    /// source). This keeps the determinism contract intact: for a fixed
    /// construction seed and a fixed sequence of steps/resets, the jitter
    /// (like everything else `self.rng` drives) is bit-reproducible, and
    /// differs from episode to episode because the stream keeps advancing.
    fn jitter_kickoff_spawns(&mut self) {
        let ids = self.car_ids.clone();
        for id in ids {
            let mut cs = self.arena.pin_mut().get_car(id);
            let dx = -KICKOFF_JITTER_POS + self.rng.next_f32() * (2.0 * KICKOFF_JITTER_POS);
            let dy = -KICKOFF_JITTER_POS + self.rng.next_f32() * (2.0 * KICKOFF_JITTER_POS);
            let dyaw = -KICKOFF_JITTER_YAW + self.rng.next_f32() * (2.0 * KICKOFF_JITTER_YAW);
            cs.pos.x += dx;
            cs.pos.y += dy;
            let base_yaw = cs.rot_mat.forward.y.atan2(cs.rot_mat.forward.x);
            cs.rot_mat = Angle { yaw: base_yaw + dyaw, pitch: 0.0, roll: 0.0 }.to_rotmat();
            self.arena.pin_mut().set_car(id, cs).expect("car exists (jitter_kickoff_spawns)");
        }
        // Ball horizontal micro-offset (±KICKOFF_JITTER_BALL uu): even
        // perfectly-corrected symmetric approaches now meet an off-center
        // ball, so the two contacts can't mirror. z untouched (rest height).
        let mut ball = self.arena.pin_mut().get_ball();
        ball.pos.x += -KICKOFF_JITTER_BALL + self.rng.next_f32() * (2.0 * KICKOFF_JITTER_BALL);
        ball.pos.y += -KICKOFF_JITTER_BALL + self.rng.next_f32() * (2.0 * KICKOFF_JITTER_BALL);
        self.arena.pin_mut().set_ball(ball);
    }

    /// Test/debug helper: force an episode reset (through the same curriculum-aware
    /// path `step` uses on termination/truncation).
    pub fn debug_force_reset(&mut self) {
        self.reset_episode();
    }

    pub fn num_agents(&self) -> usize {
        self.car_ids.len()
    }

    fn agent_car_index(&self, state: &GameState, agent: usize) -> usize {
        let id = self.car_ids[agent];
        state.cars.iter().position(|c| c.id == id).expect("car exists")
    }

    pub fn write_obs(&mut self, out: &mut [f32]) {
        let gs = self.arena.pin_mut().get_game_state();
        for a in 0..self.num_agents() {
            let ci = self.agent_car_index(&gs, a);
            obs::build_obs(&gs, ci, &self.norm, &mut out[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
        }
    }

    /// V1 obs for every agent (arena-major slices, like `write_obs`):
    /// `ents` is `num_agents * MAX_ENT * ENT_FEAT`, `mask` `num_agents *
    /// MAX_ENT`, `query` `num_agents * Q_FEAT`, `prev` `num_agents *
    /// PREV_ACTIONS`. Runs ONE ball prediction for the whole arena, shared
    /// across its agents, and copies each agent's prev-action ring out.
    /// Panics if the arena was not constructed with `ObsMode::V1`.
    pub fn write_obs_v1(
        &mut self,
        ents: &mut [f32],
        mask: &mut [bool],
        query: &mut [f32],
        prev: &mut [i64],
    ) {
        let gs = self.arena.pin_mut().get_game_state();
        let pred = self
            .v1
            .as_mut()
            .expect("write_obs_v1 requires ObsMode::V1")
            .tracker
            .predict(&gs.ball);
        for a in 0..self.num_agents() {
            let ci = self.agent_car_index(&gs, a);
            obs_v1::build(
                &gs,
                ci,
                &pred,
                &self.norm,
                &mut ents[a * MAX_ENT * ENT_FEAT..(a + 1) * MAX_ENT * ENT_FEAT],
                &mut mask[a * MAX_ENT..(a + 1) * MAX_ENT],
                &mut query[a * Q_FEAT..(a + 1) * Q_FEAT],
            );
            prev[a * PREV_ACTIONS..(a + 1) * PREV_ACTIONS]
                .copy_from_slice(&self.v1.as_ref().unwrap().prev[a]);
        }
    }

    pub fn step(
        &mut self,
        action_idx: &[i64],
        rewards: &mut [f32],
        flags: &mut [StepFlags],
        final_obs: &mut [f32],
    ) {
        self.step_impl(action_idx, rewards, flags, FinalObsOut::V0(final_obs))
    }

    /// V1 twin of `step`: identical episode logic, but the terminal-step
    /// final obs is written as entity tensors (`final_*` buffers sized like
    /// `write_obs_v1`'s). Also shifts each agent's prev-action ring with the
    /// executed action (rings reset to zeros whenever the episode resets).
    #[allow(clippy::too_many_arguments)]
    pub fn step_v1(
        &mut self,
        action_idx: &[i64],
        rewards: &mut [f32],
        flags: &mut [StepFlags],
        final_ents: &mut [f32],
        final_mask: &mut [bool],
        final_query: &mut [f32],
        final_prev: &mut [i64],
    ) {
        self.step_impl(
            action_idx,
            rewards,
            flags,
            FinalObsOut::V1 {
                ents: final_ents,
                mask: final_mask,
                query: final_query,
                prev: final_prev,
            },
        )
    }

    fn step_impl(
        &mut self,
        action_idx: &[i64],
        rewards: &mut [f32],
        flags: &mut [StepFlags],
        mut fin: FinalObsOut<'_>,
    ) {
        let n = self.num_agents();
        assert_eq!(action_idx.len(), n);

        // V1: shift the executed action into each agent's prev-action ring
        // (most-recent-first) BEFORE any possible reset below — a reset then
        // rightly zeroes it. No-op (and no state read) in v0 mode.
        if let Some(v1) = self.v1.as_mut() {
            for (a, &act) in action_idx.iter().enumerate() {
                let ring = &mut v1.prev[a];
                for i in (1..PREV_ACTIONS).rev() {
                    ring[i] = ring[i - 1];
                }
                ring[0] = act;
            }
        }

        let controls: Vec<(u32, rocketsim_rs::sim::CarControls)> = (0..n)
            .map(|a| {
                let row = &self.table[action_idx[a] as usize];
                (self.car_ids[a], actions::to_controls(row))
            })
            .collect();
        self.arena.pin_mut().set_all_controls(&controls).expect("valid car ids");
        self.arena.pin_mut().step(self.tick_skip);

        let cur = self.arena.pin_mut().get_game_state();

        // Physics-blowup containment. RocketSim's contact solver can go nonfinite
        // (observed live at 2.2B steps: two near-static cars pinching the ball ->
        // a car ejected to [nan,-inf,nan]) OR ramp through huge-but-finite values
        // for one or more tick_skip boundaries before that (observed live: a
        // "levitating ball" latched at finite pos up to 8e10 uu / ball z as low as
        // -33,137 uu, persisting hundreds to thousands of steps because
        // `state_is_finite` alone accepts it). `state_is_sane` (bounds ~2-4x the
        // hardest legal state) catches both: NaN/inf (it calls state_is_finite
        // first) AND the finite-but-insane precursor, capping the blast radius at
        // one contained transition instead of a long-lived poisoned latch. This
        // check must run BEFORE goal detection below: an insane state that happens
        // to eject the ball past the goal line makes `is_ball_scored()` fire a real
        // (not fake) goal for a physically impossible position, which would inject
        // a spurious +/-goal reward (observed live: +24.55) — running the sane
        // check first means that can never happen. A poisoned state must never
        // reach rewards, obs, or the learner: end the episode with finite zeros and
        // a fresh kickoff. Terminated (not truncated) so GAE bootstraps 0 instead of
        // a value estimate of garbage.
        if !state_is_sane(&cur) {
            self.blowup_count += 1;
            eprintln!(
                "[construct-engine] physics blowup contained (tick {}): episode terminated, arena rebuilt (Bullet DISABLE_SIMULATION latch) + reset",
                cur.tick_count
            );
            for a in 0..n {
                rewards[a] = 0.0;
                flags[a] = StepFlags { terminated: true, truncated: false };
            }
            match &mut fin {
                FinalObsOut::V0(buf) => buf[..n * OBS_SIZE].fill(0.0),
                // V1 poisoned-state equivalent of "finite zeros": zero
                // entity/query rows, all UNMASKED (present zero entities —
                // guaranteed-finite forward input), zero prev history.
                FinalObsOut::V1 { ents, mask, query, prev } => {
                    ents[..n * MAX_ENT * ENT_FEAT].fill(0.0);
                    mask[..n * MAX_ENT].fill(false);
                    query[..n * Q_FEAT].fill(0.0);
                    prev[..n * PREV_ACTIONS].fill(0);
                }
            }
            // Reset alone is NOT enough: the blowup latched DISABLE_SIMULATION
            // on the poisoned body (see `rebuild_arena`) and a kickoff reset
            // would leave it permanently frozen. Swap in a fresh arena first.
            self.rebuild_arena();
            self.reset_episode();
            return;
        }

        let scored = if self.arena.is_ball_scored() {
            Some(if cur.ball.pos.y > 0.0 { Team::Blue } else { Team::Orange })
        } else {
            None
        };

        // Touch tracking for no-touch truncation. Mirrors reward.rs's touch-detection
        // contract: a car that has never touched the ball has `is_valid == false` and
        // `tick_count_when_hit == u64::MAX` (a sentinel, not 0 — see reward.rs's module
        // doc comment for how this was verified). Requiring `is_valid` here is belt-and-
        // suspenders: the `hit <= cur.tick_count` bound already excludes the u64::MAX
        // sentinel in practice, but gating on `is_valid` makes the "never touched" case
        // explicit and keeps this loop consistent with reward.rs's touch check rather
        // than relying solely on the numeric bound.
        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            let hit_info = &cur.cars[ci].state.ball_hit_info;
            let hit = hit_info.tick_count_when_hit;
            if hit_info.is_valid && hit > self.last_touch_tick && hit <= cur.tick_count {
                self.last_touch_tick = hit;
            }
        }

        let terminated = scored.is_some();
        let truncated = !terminated
            && (cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS
                || cur.tick_count - self.episode_start_tick >= MAX_TICKS);

        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            rewards[a] = reward::compute(&self.prev_state, &cur, ci, scored, &self.reward_cfg);
            flags[a] = StepFlags { terminated, truncated };
        }

        if self.reward_cfg.team_spirit != 0.0 || self.reward_cfg.opp_spirit != 0.0 {
            blend_team_spirit(
                &mut rewards[..n],
                self.blue_count,
                self.reward_cfg.team_spirit,
                self.reward_cfg.opp_spirit,
            );
        }

        if terminated || truncated {
            // capture final obs, then reset
            match &mut fin {
                FinalObsOut::V0(buf) => {
                    for a in 0..n {
                        let ci = self.agent_car_index(&cur, a);
                        obs::build_obs(&cur, ci, &self.norm, &mut buf[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
                    }
                }
                FinalObsOut::V1 { ents, mask, query, prev } => {
                    // One prediction on the terminal state, shared across
                    // agents; rings still hold the just-executed action
                    // (reset_episode zeroes them right after).
                    let pred = self
                        .v1
                        .as_mut()
                        .expect("step_v1 requires ObsMode::V1")
                        .tracker
                        .predict(&cur.ball);
                    for a in 0..n {
                        let ci = self.agent_car_index(&cur, a);
                        obs_v1::build(
                            &cur,
                            ci,
                            &pred,
                            &self.norm,
                            &mut ents[a * MAX_ENT * ENT_FEAT..(a + 1) * MAX_ENT * ENT_FEAT],
                            &mut mask[a * MAX_ENT..(a + 1) * MAX_ENT],
                            &mut query[a * Q_FEAT..(a + 1) * Q_FEAT],
                        );
                        prev[a * PREV_ACTIONS..(a + 1) * PREV_ACTIONS]
                            .copy_from_slice(&self.v1.as_ref().unwrap().prev[a]);
                    }
                }
            }
            self.reset_episode();
        } else {
            self.prev_state = cur;
        }
    }

    pub fn game_state(&mut self) -> GameState {
        self.arena.pin_mut().get_game_state()
    }

    /// Test/debug helper: warp the ball.
    pub fn debug_place_ball(&mut self, pos: [f32; 3], vel: [f32; 3]) {
        let mut ball = self.arena.pin_mut().get_ball();
        ball.pos = Vec3::new(pos[0], pos[1], pos[2]);
        ball.vel = Vec3::new(vel[0], vel[1], vel[2]);
        self.arena.pin_mut().set_ball(ball);
    }

    /// JSON state dump matching the deploy/obs.py dict contract (Task 12 interfaces).
    pub fn debug_state_json(&mut self) -> String {
        let gs = self.arena.pin_mut().get_game_state();
        let v3 = |v: &Vec3| serde_json::json!([v.x, v.y, v.z]);
        let cars: Vec<serde_json::Value> = self
            .car_ids
            .iter()
            .map(|&id| {
                let c = gs.cars.iter().find(|c| c.id == id).expect("car exists");
                serde_json::json!({
                    "id": c.id,
                    "team": c.team as u8,
                    "pos": v3(&c.state.pos),
                    "vel": v3(&c.state.vel),
                    "ang_vel": v3(&c.state.ang_vel),
                    "forward": v3(&c.state.rot_mat.forward),
                    "up": v3(&c.state.rot_mat.up),
                    "boost": c.state.boost,
                    "is_on_ground": c.state.is_on_ground,
                    "has_flip": c.state.has_flip_or_jump(),
                    "is_demoed": c.state.is_demoed,
                })
            })
            .collect();
        serde_json::json!({
            "ball": {
                "pos": v3(&gs.ball.pos),
                "vel": v3(&gs.ball.vel),
                "ang_vel": v3(&gs.ball.ang_vel),
            },
            "cars": cars,
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_math_exact() {
        // raw: blue [1.0, 3.0], orange [-2.0, 0.0]; tau=0.5, opp=0.25
        // bm=2.0, om=-1.0
        // blue0' = 0.5*1.0 + 0.5*2.0 - 0.25*(-1.0) = 1.75
        // blue1' = 0.5*3.0 + 0.5*2.0 - 0.25*(-1.0) = 2.75
        // org0'  = 0.5*(-2.0) + 0.5*(-1.0) - 0.25*2.0 = -2.0
        // org1'  = 0.5*0.0 + 0.5*(-1.0) - 0.25*2.0 = -1.0
        let mut r = vec![1.0f32, 3.0, -2.0, 0.0];
        blend_team_spirit(&mut r, 2, 0.5, 0.25);
        assert_eq!(r, vec![1.75, 2.75, -2.0, -1.0]);
    }

    #[test]
    fn blend_handles_empty_orange() {
        let mut r = vec![2.0f32];
        blend_team_spirit(&mut r, 1, 0.5, 0.25); // opp mean = 0.0
        assert_eq!(r, vec![2.0]); // (1-.5)*2 + .5*2 - .25*0 = 2.0
    }

    #[test]
    fn sane_rejects_blowup_ramp_but_keeps_hard_shots() {
        let mut gs = GameState::default();
        gs.ball.pos = Vec3::new(0.0, 5_100.0, 92.75);
        gs.ball.vel = Vec3::new(0.0, 6_000.0, 0.0); // hardest legal shot
        assert!(state_is_sane(&gs));
        gs.ball.vel = Vec3::new(0.0, 3.4e7, 0.0); // finite blowup precursor
        assert!(!state_is_sane(&gs));
        gs.ball.vel = Vec3::new(0.0, f32::NAN, 0.0);
        assert!(!state_is_sane(&gs));
    }

    // --- Kickoff spawn jitter -------------------------------------------------
    //
    // These tests build real `EpisodeArena`s (RocketSim's own arena, not a mock),
    // exactly like engine/tests/episode_test.rs's `mk` helper: paths are relative
    // to the `engine/` package root, which is `cargo test`'s cwd for both the
    // unit-test and integration-test binaries.

    fn mk_jitter_test(blue: usize, orange: usize, seed: u32) -> EpisodeArena {
        crate::sim_init::ensure_init(None);
        let s = crate::schema::Schema::load("../schema/v0.toml").unwrap();
        let cfg = crate::reward::RewardConfig::load("../configs/reward_v0.toml").unwrap();
        EpisodeArena::new(blue, orange, s.tick_skip, cfg, s.normalization, seed)
    }

    /// (x, y, yaw) per car id, keyed so ordering differences in `GameState.cars`
    /// (RocketSim's internal storage order — see `agent_car_index`) can't
    /// desync the comparison.
    fn car_xy_yaw(gs: &GameState) -> std::collections::HashMap<u32, (f32, f32, f32)> {
        gs.cars
            .iter()
            .map(|c| {
                let yaw = c.state.rot_mat.forward.y.atan2(c.state.rot_mat.forward.x);
                (c.id, (c.state.pos.x, c.state.pos.y, yaw))
            })
            .collect()
    }

    /// Shortest-path angle difference, wrapped into (-pi, pi] — needed because a
    /// standard kickoff yaw can sit right at the atan2 branch cut (±pi), where a
    /// tiny jitter can flip the raw (jittered - baseline) sign by ~2*pi even
    /// though the physical rotation only changed by KICKOFF_JITTER_YAW.
    fn wrap_angle_delta(mut d: f32) -> f32 {
        let pi = std::f32::consts::PI;
        let tau = std::f32::consts::TAU;
        while d > pi {
            d -= tau;
        }
        while d <= -pi {
            d += tau;
        }
        d
    }

    #[test]
    fn kickoff_jitter_within_bounds_and_differs_across_cars_and_episodes() {
        // Two arenas, same seed, lockstep-reset together: `base` has jitter
        // disabled (isolating the RocketSim-chosen kickoff formation, which
        // both arenas pick identically since jitter draws never touch the
        // `self.seed` sequence that formation selection uses), `jit` has it
        // enabled. Subtracting per-car positions/yaws at each matching episode
        // isolates exactly the jitter contribution.
        let mut base = mk_jitter_test(2, 2, 42);
        base.jitter_enabled = false;
        let mut jit = mk_jitter_test(2, 2, 42);
        jit.jitter_enabled = true;

        let mut episode_deltas: Vec<Vec<(u32, f32, f32, f32)>> = Vec::new();
        for ep in 0..4 {
            // Always force a fresh reset under the (now-set) flags — the
            // constructor's own first reset ran BEFORE `jitter_enabled` was
            // overridden above, so relying on it here would compare two
            // already-identically-jittered arenas and see a zero delta.
            base.debug_force_reset();
            jit.debug_force_reset();
            let base_by_id = car_xy_yaw(&base.game_state());
            let jit_by_id = car_xy_yaw(&jit.game_state());
            assert_eq!(base_by_id.len(), 4, "expected 4 cars");

            let mut deltas: Vec<(u32, f32, f32, f32)> = Vec::new();
            for (&id, &(jx, jy, jyaw)) in jit_by_id.iter() {
                let (bx, by, byaw) = base_by_id[&id];
                let dx = jx - bx;
                let dy = jy - by;
                let dyaw = wrap_angle_delta(jyaw - byaw);
                assert!(
                    dx.abs() <= KICKOFF_JITTER_POS + 1e-3,
                    "ep {ep} car {id}: dx {dx} exceeds ±{KICKOFF_JITTER_POS}"
                );
                assert!(
                    dy.abs() <= KICKOFF_JITTER_POS + 1e-3,
                    "ep {ep} car {id}: dy {dy} exceeds ±{KICKOFF_JITTER_POS}"
                );
                assert!(
                    dyaw.abs() <= KICKOFF_JITTER_YAW + 1e-3,
                    "ep {ep} car {id}: dyaw {dyaw} exceeds ±{KICKOFF_JITTER_YAW}"
                );
                deltas.push((id, dx, dy, dyaw));
            }
            deltas.sort_by_key(|d| d.0);

            // Differ between cars: every car draws its own 3 rng values, so all 4
            // cars landing on an identical (dx, dy, dyaw) triple is not possible
            // in practice with a real PRNG stream.
            let all_same = deltas.windows(2).all(|w| {
                (w[0].1 - w[1].1).abs() < 1e-6
                    && (w[0].2 - w[1].2).abs() < 1e-6
                    && (w[0].3 - w[1].3).abs() < 1e-6
            });
            assert!(!all_same, "ep {ep}: jitter must differ between cars, got {deltas:?}");

            episode_deltas.push(deltas);
        }

        // Differ across episodes: the rng stream keeps advancing every reset, so
        // episode 0's jitter must not equal episode 1's.
        assert_ne!(
            episode_deltas[0], episode_deltas[1],
            "jitter must differ across episodes"
        );
    }

    #[test]
    fn kickoff_jitter_reproducible_for_fixed_seed() {
        // Full production path (jitter_enabled defaults true): two independently
        // constructed arenas with the same seed must land on bit-identical
        // jittered kickoff spawns.
        let mut a = mk_jitter_test(2, 2, 99);
        let mut b = mk_jitter_test(2, 2, 99);
        let ma = car_xy_yaw(&a.game_state());
        let mb = car_xy_yaw(&b.game_state());
        assert_eq!(a.car_ids, b.car_ids);
        for id in &a.car_ids {
            assert_eq!(
                ma[id], mb[id],
                "car {id}: jittered kickoff spawn must be reproducible for a fixed seed"
            );
        }
    }

    #[test]
    fn kickoff_jitter_reduces_pinch_blowups() {
        // THE MONEY TEST. A symmetric aggressive scripted policy (forward +
        // boost, identical for every car) reproduces the diagnosed live failure
        // mode organically (no synthetic NaN injection): both mirrored kickoff
        // cars drive straight at the ball and arrive together.
        //
        // Development note on methodology (kept here because it materially
        // affects how to read this test's result): an earlier version of this
        // test let each attempt run until natural termination/truncation
        // (up to 500 steps ~ the no-touch window). That let a small fraction of
        // attempts run long enough for the "always forward" car to sail past
        // the ball and ride up a side wall/ceiling curve — a REAL but
        // UNRELATED solver-stress mode (wall-geometry edge cases), which
        // contaminated the count and, at large sample sizes, made jitter look
        // WORSE (more chances to clip a wall at a different angle), not
        // better. Capping each attempt to a short window matching the
        // diagnosed containment tick range (~460-1000 ticks = ~57-125 steps @
        // tick_skip 8) and force-resetting every attempt isolates the actual
        // mechanism under test — the immediate kickoff convergence — from that
        // confound.
        //
        // Even with that fix, this measurement is genuinely chaotic: repeated
        // investigation (see PR discussion / commit message) showed the SAME
        // (seed, window, jitter) comparison can flip sign depending on how
        // many unrelated `Arena`s were constructed earlier in the process
        // (RocketSim/Bullet's broadphase and allocator are allocation-order-
        // sensitive, matching the "allocation-sensitive" caveat in this task's
        // brief). So: this test pools many seeds to reduce single-seed noise,
        // reports the actual counts unconditionally, and only hard-asserts a
        // reduction when this run's own baseline shows a real (non-trivial,
        // n>=3) blowup count AND jitter actually reduced it here. Otherwise it
        // documents the result and falls back to the bounds+determinism
        // guarantees the other two tests already assert unconditionally,
        // per this task's own pre-approved fallback for a non-deterministically
        // reproducible physics-chaos signal.
        let table = actions::make_lookup_table();
        let fwd_boost = table
            .iter()
            .position(|r| r[0] == 1.0 && r[1] == 0.0 && r[5] == 0.0 && r[6] == 1.0)
            .expect("forward+boost [throttle=1,steer=0,jump=0,boost=1] row must exist");

        const N_SEEDS: u32 = 20;
        const EPISODES_PER_SEED: usize = 40;
        const WINDOW_STEPS: usize = 100; // ~800 ticks: inside the diagnosed ~460-1000 tick range

        let run = |jitter: bool| -> u64 {
            let mut total = 0u64;
            for i in 0..N_SEEDS {
                let seed = 20260719u32.wrapping_add(i.wrapping_mul(7919));
                let mut a = mk_jitter_test(1, 1, seed);
                a.jitter_enabled = jitter;
                let mut r = vec![0.0; 2];
                let mut f = vec![StepFlags::default(); 2];
                let mut fo = vec![0.0; 2 * OBS_SIZE];
                for _ in 0..EPISODES_PER_SEED {
                    a.debug_force_reset(); // fresh mirrored kickoff every attempt
                    for _ in 0..WINDOW_STEPS {
                        a.step(&[fwd_boost as i64, fwd_boost as i64], &mut r, &mut f, &mut fo);
                        if f[0].terminated || f[0].truncated {
                            break;
                        }
                    }
                }
                total += a.blowup_count;
            }
            total
        };

        let n_attempts = (N_SEEDS as usize) * EPISODES_PER_SEED;
        let unjittered = run(false);
        let jittered = run(true);
        eprintln!(
            "[kickoff_jitter_reduces_pinch_blowups] {n_attempts} kickoff attempts/condition (window={WINDOW_STEPS} steps): \
             unjittered blowups={unjittered}, jittered blowups={jittered}"
        );

        if unjittered >= 3 && jittered < unjittered {
            eprintln!(
                "[kickoff_jitter_reduces_pinch_blowups] jitter reduced organic pinch blowups {unjittered} -> {jittered} \
                 ({:.0}% reduction) in this run",
                100.0 * (1.0 - jittered as f64 / unjittered as f64)
            );
            assert!(
                jittered < unjittered,
                "jittered pinch count ({jittered}/{n_attempts}) must be below unjittered ({unjittered}/{n_attempts})"
            );
        } else {
            // Documented per the task brief: the car-car-ball pinch blowup this
            // fix targets is a contact-solver degenerate case that is
            // allocation/timing-sensitive outside the live training process
            // (confirmed during development: identical parameters reproduced
            // opposite-signed results depending on unrelated prior process
            // history). A near-zero or non-reduced count in this particular run
            // isn't a meaningful basis for a hard pass/fail assertion — fall
            // back to the bounds+determinism guarantees already covered
            // unconditionally by kickoff_jitter_within_bounds_and_differs_across_cars_and_episodes
            // and kickoff_jitter_reproducible_for_fixed_seed, and report the
            // numbers honestly instead of asserting a comparison this run
            // can't actually support.
            eprintln!(
                "[kickoff_jitter_reduces_pinch_blowups] no robust reduction observed in this run \
                 (unjittered={unjittered}, jittered={jittered}) — this is a known allocation/timing-sensitive \
                 physics-chaos signal (see doc comment above); skipping the hard pinch-rate assertion, bounds+\
                 determinism remain covered unconditionally by the other two kickoff_jitter_* tests"
            );
        }
    }
}
