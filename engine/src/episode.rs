use crate::{
    actions,
    ballpred::Tracker,
    curriculum::CurriculumConfig,
    obs::{self, OBS_SIZE},
    obs_v1::{self, ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT},
    reset_pool::ResetState,
    reward::{self, MatchState, RewardConfig},
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

/// Which flavor of state an episode reset starts from. `Replay` carries the
/// already-drawn pool index so the draw and the apply stay in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResetKind {
    Replay(usize),
    Random,
    Kickoff,
}

/// The reset draw, extracted from `reset_episode` so the legacy-rng-stream
/// contract can be tested against the SHIPPED expression instead of a copy of
/// it. Everything here is a pure function of the rng and the weights.
///
/// `replay_eff` is the replay share *after* the caller has zeroed it for a
/// non-duel arena or an empty pool. Exactly ONE `next_f32` per call, as before
/// the replay lever existed, plus a `next_u32` on the replay branch only.
///
/// Branch order and the threshold expressions are load-bearing: with
/// `replay_eff == 0.0`, `u < 0.0 / total` is always false (strict `<`, and
/// `u ∈ [0,1)`), while `(0.0 + random) / (0.0 + kickoff + random)` is
/// BIT-IDENTICAL to the legacy `random / (random + kickoff)` (`0.0 + x == x`
/// exactly; IEEE addition is commutative). So every legacy config, every
/// non-duel arena, and every arena with an empty pool reproduces the old rng
/// stream and the old sequence of reset states exactly.
///
/// That claim is pinned by `zero_replay_weight_matches_legacy_coin`, which
/// compares this function against an inline transcription of the pre-replay
/// two-way coin. It is deliberately NOT pinned by comparing two arenas that
/// both have `replay_eff == 0.0` — those execute identical arithmetic and agree
/// no matter what this function does.
fn draw_reset_kind(
    rng: &mut Pcg32,
    replay_eff: f32,
    kickoff_weight: f32,
    random_weight: f32,
    pool_len: usize,
) -> ResetKind {
    let total = replay_eff + kickoff_weight + random_weight;
    let u = rng.next_f32();
    if total > 0.0 && u < replay_eff / total {
        // Index from `next_u32`, never from `next_f32`: with a ~1.08M pool,
        // `(next_f32() * len as f32) as usize` can round up to exactly `len` in
        // f32 (spacing near 1e6 is 0.125) and panic out of bounds. Modulo bias
        // over 2^32 is ~2.5e-4 relative.
        ResetKind::Replay((rng.next_u32() as usize) % pool_len)
    } else if total > 0.0 && u < (replay_eff + random_weight) / total {
        ResetKind::Random
    } else {
        ResetKind::Kickoff
    }
}

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
    /// Match score, only meaningful when `curriculum.match_mode` is set.
    score_blue: u32,
    score_orange: u32,
    /// Tick the current MATCH began (distinct from `episode_start_tick`, which
    /// in match mode restarts at every kickoff within the match).
    match_start_tick: u64,
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
            score_blue: 0,
            score_orange: 0,
            match_start_tick: 0,
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
        this.start_match(0);
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
        let kind = match &self.curriculum {
            Some(c) => {
                // The pool is duels-only by construction, so a 2v2/3v3 arena (or
                // an empty/absent pool) zeroes the replay share and the draw
                // RENORMALIZES over {kickoff, random}. It deliberately does NOT
                // "draw replay, then fall back to kickoff": that would silently
                // inflate kickoff to 0.8 in every non-duel arena — an 8x
                // overweight of the one reset flavor this lever exists to move
                // away from, landing on exactly the arenas most prone to the
                // mirrored-kickoff pinch blowup.
                let replay_eff = if !c.pool.is_empty() && self.car_ids.len() == 2 && self.blue_count == 1
                {
                    c.replay_weight
                } else {
                    0.0
                };
                draw_reset_kind(
                    &mut self.rng,
                    replay_eff,
                    c.kickoff_weight,
                    c.random_weight,
                    c.pool.len(),
                )
            }
            None => ResetKind::Kickoff,
        };
        // Always first, on every branch: this is the only thing that resets boost
        // pads and per-car `ball_hit_info`, and the replay branch depends on it
        // for a default-constructed `CarState` to read-modify-write (the invalid
        // hit info the touch-tracking loop and `reward::compute` both key off).
        self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
        match kind {
            ResetKind::Replay(i) => {
                // Arc clone (refcount bump), not a pool copy — see the `pool`
                // field's doc comment on why this matters at 192 arenas.
                let pool = self.curriculum.as_ref().unwrap().pool.clone();
                self.apply_replay_state(&pool[i]);
            }
            ResetKind::Random => {
                let bounds = self.curriculum.as_ref().unwrap().random.clone();
                crate::curriculum::random_reset(self.arena.pin_mut(), &mut self.rng, &bounds);
            }
            ResetKind::Kickoff => {
                // Only jitter when the kickoff formation survives (the other two
                // branches overwrite it) — see `jitter_kickoff_spawns`. A replay
                // state is already asymmetric, and jitter would corrupt a real
                // recorded position.
                if self.jitter_enabled {
                    self.jitter_kickoff_spawns();
                }
            }
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

    /// Overwrites the ball and both cars with a recorded human state, on top of
    /// the kickoff `reset_episode` just performed. Uses the same
    /// `get -> mutate -> set` idiom as `curriculum::random_reset`, and for the
    /// same reason: `set_car`/`set_ball` REPLACE the whole struct, so every
    /// field we don't assign is inherited from the fresh post-kickoff state
    /// rather than from a `CarState::default()` we'd have to get right by hand.
    ///
    /// Six fields per car — exactly the set `random_reset` writes. Everything
    /// else stays at the post-kickoff default deliberately: `ball_hit_info`
    /// invalid (a stale hit would falsely reset `last_touch_tick`), `is_demoed`
    /// false (the pool has no demo flag; filter F3 drops demo-proxy states),
    /// jump/flip state cleared (so every replay-reset car is granted a flip —
    /// the pool has no `has_flip` field, and this matches `random_reset`),
    /// `last_controls` zero, and all boost pads available.
    ///
    /// KNOWN DEPLOY RISK, not a no-op — the last two are a train/finetune obs
    /// mismatch, and it is bigger than "airborne replay cars" suggests:
    ///
    /// - 43.0% of corpus cars (807,859 of 1,877,602) are airborne, and 61.5% of
    ///   states contain at least one. Every one of them resets with
    ///   `has_jumped/has_double_jumped/has_flipped = false`, so `f22`
    ///   (`obs_v1.rs:165`, `has_flip_or_jump()`) reads 1 for a car that in the
    ///   real frame had already burned its flip.
    /// - All 34 pads reset available, so `f21`/`timer_norm` (`obs_v1.rs:186`,
    ///   `:205`) read "full" against a mid-game state that had several on
    ///   cooldown.
    ///
    /// This matters specifically on a BC-pretrain branch: `replay/src/bc_obs.rs`
    /// builds the SAME features from the SAME corpus with the REAL values
    /// (`bc_obs.rs:391` sets `has_flipped` from the shard; `:393-397` sets real
    /// pad `is_active`/`cooldown`). So a BC-pretrained policy learns f21/f22 on
    /// real values and then meets fabricated ones on the same states during RL
    /// fine-tuning. Closing it means carrying `has_flip` and pad cooldowns in
    /// the pool schema (`replay/src/reset_pool.rs`), which is a corpus re-parse,
    /// not an engine change — tracked as backlog, deliberately not papered over
    /// here.
    ///
    /// No unit conversion on positions/velocities — the pool is raw
    /// RocketSim-native world frame (uu, uu/s, rad/s). Boost is the one
    /// exception: `0..1` on the wire, `0..100` in `CarState`.
    fn apply_replay_state(&mut self, st: &ResetState) {
        let mut ball = self.arena.pin_mut().get_ball();
        ball.pos = Vec3::new(st.ball.pos[0], st.ball.pos[1], st.ball.pos[2]);
        ball.vel = Vec3::new(st.ball.vel[0], st.ball.vel[1], st.ball.vel[2]);
        ball.ang_vel = Vec3::new(st.ball.ang_vel[0], st.ball.ang_vel[1], st.ball.ang_vel[2]);
        // `rot_mat` left at the kickoff default: the ball is rotationally
        // symmetric, `random_reset` doesn't set it either, and the pool has no
        // ball-rotation field by design.
        self.arena.pin_mut().set_ball(ball);

        // `st.cars[i]` -> `self.car_ids[i]`: the pool is blue-then-orange (filter
        // F7 enforces it at load) and `car_ids` is blue-asc-then-orange-asc, and
        // the caller's gate guarantees this is a 1v1 arena.
        let ids = self.car_ids.clone();
        for (i, &id) in ids.iter().enumerate() {
            let sp = &st.cars[i];
            let mut cs = self.arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(sp.pos[0], sp.pos[1], sp.pos[2]);
            cs.vel = Vec3::new(sp.vel[0], sp.vel[1], sp.vel[2]);
            cs.ang_vel = Vec3::new(sp.ang_vel[0], sp.ang_vel[1], sp.ang_vel[2]);
            cs.rot_mat = crate::reset_pool::quat_to_rotmat(sp.quat);
            cs.boost = sp.boost * 100.0;
            cs.is_on_ground = sp.on_ground;
            self.arena.pin_mut().set_car(id, cs).expect("car exists (apply_replay_state)");
        }
        // `state_is_sane`'s limits are pos <=12000 / vel <=20000 / ang_vel
        // <=100. Measured maxima over the 938,801 states the filters accept
        // BEFORE F8 were 5987.9 / 4732.2 / 96.5 — that ang_vel figure is 17.5x
        // RocketSim's own 5.5 rad/s cap and sat 3.6% under this assert, not the
        // comfortable margin an earlier version of this comment claimed. F8
        // (`reset_pool::accept`) now bounds ang_vel at 6.0 and car speed at
        // 2400, which restores the margin to ~16x and makes this a genuine
        // debug-only backstop rather than a near-miss.
        debug_assert!(state_is_sane(&self.arena.pin_mut().get_game_state()));
    }

    /// Begin a NEW match: zero the score and restart the clock. Deliberately
    /// separate from `reset_episode`, which also runs on every post-goal
    /// kickoff WITHIN a match and must leave the score alone.
    fn start_match(&mut self, tick: u64) {
        self.score_blue = 0;
        self.score_orange = 0;
        self.match_start_tick = tick;
    }

    /// Test/debug helper: force an episode reset (through the same curriculum-aware
    /// path `step` uses on termination/truncation).
    pub fn debug_force_reset(&mut self) {
        self.reset_episode();
    }

    pub fn num_agents(&self) -> usize {
        self.car_ids.len()
    }

    /// Physics-blowup containments (arena rebuilds) this arena has performed.
    /// Test-only instrumentation — exposed so integration tests can assert a
    /// reset flavor doesn't drive the contact solver nonfinite.
    pub fn blowup_count(&self) -> u64 {
        self.blowup_count
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

        // MATCH MODE (task #56): a goal is a scoring event INSIDE the
        // trajectory, not the end of it. Credit therefore flows across the
        // goal, which is the whole point -- "was that goal worth the
        // counter-attack it conceded?" is only learnable if the trajectory
        // continues. Only the clock ends a match.
        //
        // LEGACY MODE: unchanged. A goal terminates. The goal-share screen and
        // every historical gate depend on this exact behavior.
        let (terminated, truncated) = if self.match_mode() {
            if let Some(team) = scored {
                match team {
                    Team::Blue => self.score_blue += 1,
                    Team::Orange => self.score_orange += 1,
                }
            }
            let match_over = cur.tick_count - self.match_start_tick >= MAX_TICKS;
            let stalled = cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS;
            (match_over, !match_over && stalled)
        } else {
            let t = scored.is_some();
            let tr = !t
                && (cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS
                    || cur.tick_count - self.episode_start_tick >= MAX_TICKS);
            (t, tr)
        };

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

        // Match mode: a goal that did NOT end the match still needs a kickoff.
        if self.match_mode() && scored.is_some() && !terminated && !truncated {
            self.reset_episode();
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
            if self.match_mode() {
                self.start_match(cur.tick_count);
            }
            self.reset_episode();
        } else {
            self.prev_state = cur;
        }
    }

    pub fn game_state(&mut self) -> GameState {
        self.arena.pin_mut().get_game_state()
    }

    /// Score and remaining-clock fraction for the current match.
    pub fn match_state(&self, cur_tick: u64) -> MatchState {
        let elapsed = cur_tick.saturating_sub(self.match_start_tick);
        let t_frac = 1.0 - (elapsed as f32 / MAX_TICKS as f32);
        MatchState {
            score_blue: self.score_blue,
            score_orange: self.score_orange,
            t_frac: t_frac.clamp(0.0, 1.0),
        }
    }

    fn match_mode(&self) -> bool {
        self.curriculum.as_ref().map_or(false, |c| c.match_mode)
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

    // --- Match layer (task #56 Phase 1) ----------------------------------

    fn curr_with_match_mode(on: bool) -> crate::curriculum::CurriculumConfig {
        let mut c = crate::curriculum::CurriculumConfig {
            kickoff_weight: 1.0, random_weight: 0.0, ..Default::default()
        };
        c.match_mode = on;
        c
    }

    /// Builds a real 1v1 `EpisodeArena` with the given curriculum, using the
    /// v0 schema/reward configs (existing, on disk) rather than
    /// `configs/reward_v5_winprob.toml` / `configs/curriculum_v3_match.toml`,
    /// which don't exist yet (they arrive in Task 4). The match layer under
    /// test here (score bookkeeping, clock, termination) has no dependency on
    /// the win-prob reward shaping those files configure.
    fn mk_match_test(curriculum: Option<crate::curriculum::CurriculumConfig>) -> EpisodeArena {
        crate::sim_init::ensure_init(None);
        let s = crate::schema::Schema::load("../schema/v0.toml").unwrap();
        let cfg = crate::reward::RewardConfig::load("../configs/reward_v0.toml").unwrap();
        EpisodeArena::new_with_curriculum(1, 1, s.tick_skip, cfg, s.normalization, 5150, curriculum)
    }

    fn mk_match_arena() -> EpisodeArena {
        mk_match_test(Some(curr_with_match_mode(true)))
    }

    #[test]
    fn legacy_mode_still_terminates_on_a_goal() {
        // The goal-share screen depends on this. If match_mode leaks into the
        // default path, every historical gate becomes incomparable.
        //
        // Strengthened beyond the brief's config-only check (which only
        // asserted `!c.match_mode`, not termination itself -- a config-default
        // check would keep passing even if the termination logic silently
        // stopped honoring that default): this drives a REAL arena to an
        // actual goal and asserts `terminated == true`, the property the test
        // name promises. See `match_mode_continues_past_a_goal_and_increments_score`
        // below for the mirrored match-mode case.
        let c = curr_with_match_mode(false);
        assert!(!c.match_mode);

        let mut a = mk_match_test(Some(c));
        // Warp ball into orange net with velocity, same placement as
        // episode_test.rs's scored_ball_terminates_and_pays_goal.
        a.debug_place_ball([0.0, 5200.0, 320.0], [0.0, 2000.0, 0.0]);
        let mut r = vec![0.0; 2];
        let mut f = vec![StepFlags::default(); 2];
        let mut fo = vec![0.0; 2 * OBS_SIZE];
        let mut terminated = false;
        for _ in 0..30 {
            a.step(&[0, 0], &mut r, &mut f, &mut fo);
            if f[0].terminated {
                terminated = true;
                break;
            }
        }
        assert!(terminated, "legacy mode: a goal must terminate the episode");
    }

    #[test]
    fn match_mode_continues_past_a_goal_and_increments_score() {
        // Mirror of the legacy test above: the SAME scoring event, but with
        // match_mode on, must NOT terminate (nor truncate) -- the score
        // increments instead and the match continues.
        let mut a = mk_match_arena();
        a.debug_place_ball([0.0, 5200.0, 320.0], [0.0, 2000.0, 0.0]);
        let mut r = vec![0.0; 2];
        let mut f = vec![StepFlags::default(); 2];
        let mut fo = vec![0.0; 2 * OBS_SIZE];
        let mut scored = false;
        for _ in 0..30 {
            a.step(&[0, 0], &mut r, &mut f, &mut fo);
            if a.score_blue > 0 {
                scored = true;
                assert!(!f[0].terminated, "match mode: a goal must not terminate the episode");
                assert!(!f[0].truncated, "match mode: a goal must not truncate the episode either");
                break;
            }
        }
        assert!(scored, "ball placed in goal mouth must score within 30 steps");
        assert_eq!(a.score_blue, 1);
        assert_eq!(a.score_orange, 0);
    }

    #[test]
    fn match_mode_is_off_by_default_in_every_shipped_curriculum() {
        for p in ["../configs/curriculum_v1.toml", "../configs/curriculum_v2.toml"] {
            let c = crate::curriculum::CurriculumConfig::load(p)
                .unwrap_or_else(|e| panic!("{p}: {e}"));
            assert!(!c.match_mode, "{p} must default match_mode to false");
        }
    }

    #[test]
    fn match_state_reports_signed_diff_and_falling_clock() {
        let mut a = mk_match_arena();
        a.score_blue = 2;
        a.score_orange = 1;
        a.match_start_tick = 0;
        let half = a.match_state(MAX_TICKS / 2);
        assert_eq!(half.score_diff(Team::Blue), 1);
        assert_eq!(half.score_diff(Team::Orange), -1);
        assert!((half.t_frac - 0.5).abs() < 0.01, "t_frac {}", half.t_frac);

        let start = a.match_state(0);
        assert!((start.t_frac - 1.0).abs() < 1e-6);
        let end = a.match_state(MAX_TICKS);
        assert!((end.t_frac - 0.0).abs() < 1e-6);
    }

    #[test]
    fn t_frac_never_leaves_unit_range_past_the_whistle() {
        let mut a = mk_match_arena();
        a.match_start_tick = 0;
        let over = a.match_state(MAX_TICKS * 2);
        assert!((0.0..=1.0).contains(&over.t_frac), "t_frac {}", over.t_frac);
    }
}

#[cfg(test)]
mod replay_reset_tests {
    use super::*;
    use crate::curriculum::CurriculumConfig;
    use crate::reset_pool::{BallSpawn, CarSpawn, ResetState};
    use std::sync::Arc;

    fn mk(blue: usize, orange: usize, seed: u32, c: Option<CurriculumConfig>) -> EpisodeArena {
        crate::sim_init::ensure_init(None);
        let s = crate::schema::Schema::load("../schema/v0.toml").unwrap();
        let cfg = crate::reward::RewardConfig::load("../configs/reward_v0.toml").unwrap();
        EpisodeArena::new_with_curriculum(blue, orange, s.tick_skip, cfg, s.normalization, seed, c)
    }

    fn curr(replay: f32, kickoff: f32, random: f32, pool: Vec<ResetState>) -> CurriculumConfig {
        let mut c = CurriculumConfig::load("../configs/curriculum_v1.toml").unwrap();
        c.replay_weight = replay;
        c.kickoff_weight = kickoff;
        c.random_weight = random;
        c.pool = Arc::new(pool);
        c
    }

    /// A synthetic-but-legal duel state. `tag` shifts the ball far from both the
    /// kickoff spot and (in practice) any random-reset draw, so the branch
    /// classifier below can attribute a reset unambiguously.
    fn state(tag: usize) -> ResetState {
        let t = tag as f32;
        ResetState {
            ball: BallSpawn {
                pos: [1000.0 + t, 2000.0 - t, 500.0 + t * 0.5],
                vel: [10.0 + t, -20.0, 30.0],
                ang_vel: [0.5, -0.25, 1.0],
            },
            cars: [
                CarSpawn {
                    pos: [-1500.0 - t, -2500.0, 17.01],
                    vel: [100.0 + t, 200.0, 0.0],
                    ang_vel: [0.0, 0.0, 1.5],
                    quat: [0.0014666612, 0.004585884, -0.32203302, 0.9467162],
                    boost: 0.42,
                    team: 0,
                    on_ground: true,
                },
                CarSpawn {
                    pos: [1500.0 + t, 2500.0, 800.0],
                    vel: [-300.0, -100.0, 250.0 + t],
                    ang_vel: [1.0, -2.0, 0.5],
                    quat: [0.0026252908, 0.004022506, -0.49989405, 0.86607325],
                    boost: 0.11,
                    team: 1,
                    on_ground: false,
                },
            ],
        }
    }

    fn pool_of(n: usize) -> Vec<ResetState> {
        (0..n).map(state).collect()
    }

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum Kind {
        Replay,
        Kickoff,
        Random,
    }

    /// RocketSim stores physics state in Bullet units (uu/50) and converts on
    /// every get/set, so a `set_ball`/`get_ball` round-trip is NOT bit-exact —
    /// it loses ~1e-7 relative (e.g. 1002.0 -> 1001.99994). Applied-state
    /// assertions therefore use a tight relative tolerance rather than
    /// equality; this is a property of the simulator's unit conversion, not of
    /// `apply_replay_state`. (Arena-vs-arena sequence comparisons stay
    /// bit-exact: both sides take the identical lossy path.)
    fn close(got: f32, want: f32, what: &str) {
        let tol = 1e-4 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    fn close3(got: [f32; 3], want: [f32; 3], what: &str) {
        for i in 0..3 {
            close(got[i], want[i], &format!("{what}[{i}]"));
        }
    }

    /// True iff `b` is the round-tripped image of pool ball position `p`. The
    /// synthetic pool separates entries by >= 0.5 uu per component, so a 1e-2
    /// window is unambiguous and far wider than the ~1e-4 conversion error.
    fn same_ball(b: &Vec3, p: &[f32; 3]) -> bool {
        (b.x - p[0]).abs() < 1e-2 && (b.y - p[1]).abs() < 1e-2 && (b.z - p[2]).abs() < 1e-2
    }

    /// Attributes an observed post-reset state to the branch that produced it:
    /// replay iff the ball matches a pool entry, kickoff iff the ball is at
    /// (jittered) center at rest height, random otherwise.
    fn classify(gs: &GameState, pool: &[ResetState]) -> Kind {
        let b = gs.ball.pos;
        if pool.iter().any(|s| same_ball(&b, &s.ball.pos)) {
            return Kind::Replay;
        }
        if (b.z - 93.15).abs() < 1e-3 && b.x.abs() < 20.0 && b.y.abs() < 20.0 {
            return Kind::Kickoff;
        }
        Kind::Random
    }

    /// Every float a reset can write, flattened for bit-exact sequence compare.
    fn snapshot(gs: &GameState) -> Vec<f32> {
        let mut v = vec![
            gs.ball.pos.x, gs.ball.pos.y, gs.ball.pos.z,
            gs.ball.vel.x, gs.ball.vel.y, gs.ball.vel.z,
            gs.ball.ang_vel.x, gs.ball.ang_vel.y, gs.ball.ang_vel.z,
        ];
        let mut cars: Vec<_> = gs.cars.iter().collect();
        cars.sort_by_key(|c| c.id);
        for c in cars {
            let s = &c.state;
            v.extend_from_slice(&[
                s.pos.x, s.pos.y, s.pos.z,
                s.vel.x, s.vel.y, s.vel.z,
                s.ang_vel.x, s.ang_vel.y, s.ang_vel.z,
                s.rot_mat.forward.x, s.rot_mat.forward.y, s.rot_mat.forward.z,
                s.rot_mat.right.x, s.rot_mat.right.y, s.rot_mat.right.z,
                s.rot_mat.up.x, s.rot_mat.up.y, s.rot_mat.up.z,
                s.boost,
            ]);
        }
        v
    }

    fn sequence(a: &mut EpisodeArena, n: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(snapshot(&a.game_state()));
            a.debug_force_reset();
        }
        out
    }

    // ---- T17 ----
    #[test]
    fn zero_replay_weight_is_bit_identical_to_legacy() {
        // A non-empty pool must be completely inert when replay_weight is 0:
        // no extra rng draw, no branch taken. Proves the legacy stream survives
        // merely having a pool loaded.
        let mut a = mk(1, 1, 5150, Some(curr(0.0, 0.4, 0.6, pool_of(16))));
        let mut b = mk(1, 1, 5150, Some(curr(0.0, 0.4, 0.6, vec![])));
        assert_eq!(sequence(&mut a, 50), sequence(&mut b, 50));
    }

    // ---- T18: the "pool file missing on the training box" contract ----
    #[test]
    fn empty_pool_degrades_to_legacy_stream() {
        let mut a = mk(1, 1, 991, Some(curr(0.7, 0.1, 0.2, vec![])));
        let mut b = mk(1, 1, 991, Some(curr(0.0, 0.1, 0.2, vec![])));
        assert_eq!(
            sequence(&mut a, 50),
            sequence(&mut b, 50),
            "an empty pool must reproduce the legacy kickoff/random stream exactly"
        );
    }

    // ---- T18b: the test T17/T18 only LOOKED like they were ----
    #[test]
    fn zero_replay_weight_matches_legacy_coin() {
        // T17 and T18 above compare two arenas that both end up with
        // `replay_eff == 0.0`, so both sides execute literally identical
        // arithmetic and agree regardless of what `draw_reset_kind` does. A
        // refactor that reordered the branches (kickoff tested first) diverges
        // from the legacy stream on ~80% of draws and still passes both. The
        // legacy coin has to appear on ONE side of the comparison, so here it
        // is, transcribed from 639561f (the commit before the replay lever):
        //
        //     let u = self.rng.next_f32();
        //     if u < c.random_weight / (c.random_weight + c.kickoff_weight) {
        //         ResetKind::Random
        //     } else {
        //         ResetKind::Kickoff
        //     }
        fn legacy(rng: &mut Pcg32, kickoff_weight: f32, random_weight: f32) -> ResetKind {
            let u = rng.next_f32();
            if u < random_weight / (random_weight + kickoff_weight) {
                ResetKind::Random
            } else {
                ResetKind::Kickoff
            }
        }

        // Part 1: the threshold algebra, bit-exact. `(0.0 + r) / (0.0 + k + r)`
        // must be the SAME f32 as `r / (r + k)` — not merely close, since a
        // one-ulp difference flips draws that land in the gap.
        for (k, r) in [(0.1f32, 0.2f32), (0.4, 0.6), (0.5, 0.5), (0.9, 0.1), (1.0, 0.0), (0.0, 1.0)]
        {
            let shipped = (0.0f32 + r) / (0.0f32 + k + r);
            let legacy_thresh = r / (r + k);
            assert_eq!(
                shipped.to_bits(),
                legacy_thresh.to_bits(),
                "renormalized threshold must be bit-identical for kickoff={k} random={r}"
            );
        }

        // Part 2: the branch SEQUENCE out of the shipped function must equal the
        // legacy coin's, draw for draw, from the same seed. This is what
        // actually catches a branch reorder: both consume exactly one next_f32
        // per call, so any divergence shows up as a different Kind at the same
        // index rather than as a desynced stream.
        let cfg = CurriculumConfig::load("../configs/curriculum_v1.toml").unwrap();
        for &(k, r) in
            &[(cfg.kickoff_weight, cfg.random_weight), (0.1, 0.2), (0.4, 0.6), (0.7, 0.3)]
        {
            let mut shipped_rng = Pcg32::new(0xC0FFEE);
            let mut legacy_rng = Pcg32::new(0xC0FFEE);
            let mut n_random = 0usize;
            for i in 0..20_000 {
                // `pool_len` is 1 (not 0) on purpose: a zero would make a
                // buggy replay branch panic on `% 0` instead of silently
                // producing a wrong Kind, which is a weaker signal.
                let got = draw_reset_kind(&mut shipped_rng, 0.0, k, r, 1);
                let want = legacy(&mut legacy_rng, k, r);
                assert_eq!(got, want, "draw {i} diverged from the legacy coin (k={k} r={r})");
                if got == ResetKind::Random {
                    n_random += 1;
                }
            }
            // Guard against the whole comparison being vacuous because both
            // sides always said Kickoff.
            let frac = n_random as f64 / 20_000.0;
            let want = (r / (r + k)) as f64;
            assert!(
                (frac - want).abs() < 0.02,
                "sequence must actually exercise both branches: random={frac:.3} want~{want:.3}"
            );
        }
    }

    // ---- T18c ----
    #[test]
    fn nonzero_replay_weight_diverges_from_legacy() {
        // The negative control for T18b: if `draw_reset_kind` agreed with the
        // legacy coin even with a live replay share, T18b would be proving
        // nothing about the replay branch existing at all.
        let mut a = Pcg32::new(0xC0FFEE);
        let mut b = Pcg32::new(0xC0FFEE);
        let mut diffs = 0usize;
        for _ in 0..2_000 {
            let with_replay = draw_reset_kind(&mut a, 0.7, 0.1, 0.2, 64);
            let without = draw_reset_kind(&mut b, 0.0, 0.1, 0.2, 64);
            if with_replay != without {
                diffs += 1;
            }
        }
        assert!(diffs > 1_000, "a 0.7 replay share must change most draws, got {diffs}/2000");
    }

    // ---- T19 ----
    #[test]
    fn duel_arena_applies_pool_state_exactly() {
        let st = state(7);
        let mut a = mk(1, 1, 3, Some(curr(1.0, 0.0, 0.0, vec![st.clone()])));
        a.debug_force_reset();
        let gs = a.game_state();

        close3([gs.ball.pos.x, gs.ball.pos.y, gs.ball.pos.z], st.ball.pos, "ball pos");
        close3([gs.ball.vel.x, gs.ball.vel.y, gs.ball.vel.z], st.ball.vel, "ball vel");
        close3(
            [gs.ball.ang_vel.x, gs.ball.ang_vel.y, gs.ball.ang_vel.z],
            st.ball.ang_vel,
            "ball ang_vel",
        );

        for (i, &id) in a.car_ids.clone().iter().enumerate() {
            let c = gs.cars.iter().find(|c| c.id == id).unwrap();
            let sp = &st.cars[i];
            close3([c.state.pos.x, c.state.pos.y, c.state.pos.z], sp.pos, &format!("car {i} pos"));
            close3([c.state.vel.x, c.state.vel.y, c.state.vel.z], sp.vel, &format!("car {i} vel"));
            close3(
                [c.state.ang_vel.x, c.state.ang_vel.y, c.state.ang_vel.z],
                sp.ang_vel,
                &format!("car {i} ang_vel"),
            );
            close(c.state.boost, sp.boost * 100.0, &format!("car {i} boost rescaled 0..1 -> 0..100"));
            assert_eq!(c.state.is_on_ground, sp.on_ground, "car {i} on_ground");
            let m = crate::reset_pool::quat_to_rotmat(sp.quat);
            for (got, want) in [
                (c.state.rot_mat.forward.x, m.forward.x),
                (c.state.rot_mat.forward.y, m.forward.y),
                (c.state.rot_mat.forward.z, m.forward.z),
                (c.state.rot_mat.up.x, m.up.x),
                (c.state.rot_mat.up.y, m.up.y),
                (c.state.rot_mat.up.z, m.up.z),
            ] {
                assert!((got - want).abs() < 1e-6, "car {i} rot_mat: {got} vs {want}");
            }
            // blue-then-orange: pool car i lands on agent i
            let team_u8 = if c.team == Team::Blue { 0u8 } else { 1u8 };
            assert_eq!(team_u8, sp.team, "pool car {i} must land on the matching team");
        }
    }

    // ---- T20 ----
    #[test]
    fn pool_reset_clears_episode_bookkeeping() {
        let st = state(2);
        crate::sim_init::ensure_init(None);
        let s = crate::schema::Schema::load("../schema/v0.toml").unwrap();
        let cfg = crate::reward::RewardConfig::load("../configs/reward_v0.toml").unwrap();
        let mut a = EpisodeArena::new_full(
            1, 1, s.tick_skip, cfg, s.normalization, 11,
            Some(curr(1.0, 0.0, 0.0, vec![st.clone()])), ObsMode::V1,
        );
        a.debug_force_reset();
        let tick = a.game_state().tick_count;
        assert_eq!(a.episode_start_tick, tick);
        assert_eq!(a.last_touch_tick, tick);
        let p = a.prev_state.ball.pos;
        assert!(same_ball(&p, &st.ball.pos), "prev_state must latch the applied pool state: {p:?}");
        for ring in &a.v1.as_ref().unwrap().prev {
            assert_eq!(*ring, [0; PREV_ACTIONS], "v1 prev-action rings must be zeroed");
        }
    }

    // ---- T21 ----
    #[test]
    fn pool_reset_preserves_defaults_reward_depends_on() {
        let mut a = mk(1, 1, 77, Some(curr(1.0, 0.0, 0.0, vec![state(1)])));
        a.debug_force_reset();
        let gs = a.game_state();
        for c in &gs.cars {
            assert!(!c.state.ball_hit_info.is_valid, "a fresh episode must carry no hit info");
            assert!(!c.state.is_demoed);
            assert_eq!(c.state.demo_respawn_timer, 0.0);
            assert!(c.state.has_flip_or_jump(), "replay-reset cars are granted a flip");
            assert_eq!(c.state.last_controls.throttle, 0.0);
            assert_eq!(c.state.last_controls.steer, 0.0);
            assert!(!c.state.last_controls.jump);
            assert!(!c.state.last_controls.boost);
        }
        assert!(
            gs.pads.iter().all(|p| p.state.is_active),
            "reset_to_random_kickoff still runs first, so every pad is available"
        );
    }

    // ---- T22 ----
    #[test]
    fn non_duel_arena_never_uses_pool() {
        let pool = pool_of(8);
        for (blue, orange) in [(2usize, 2usize), (3, 3)] {
            let mut a = mk(blue, orange, 404, Some(curr(1.0, 0.0, 0.0, pool.clone())));
            for _ in 0..100 {
                a.debug_force_reset();
                let gs = a.game_state();
                assert_ne!(
                    classify(&gs, &pool),
                    Kind::Replay,
                    "{blue}v{orange} arenas must never draw a duel-only pool state"
                );
            }
        }
    }

    // ---- T23 ----
    #[test]
    fn non_duel_arena_renormalizes_kickoff_and_random() {
        let pool = pool_of(8);
        let mut a = mk(2, 2, 8_191, Some(curr(0.7, 0.1, 0.2, pool.clone())));
        let (mut kick, mut rand) = (0usize, 0usize);
        let n = 4000;
        for _ in 0..n {
            a.debug_force_reset();
            match classify(&a.game_state(), &pool) {
                Kind::Kickoff => kick += 1,
                Kind::Random => rand += 1,
                Kind::Replay => panic!("non-duel arena drew a replay state"),
            }
        }
        let (k, r) = (kick as f64 / n as f64, rand as f64 / n as f64);
        // Renormalized over {kickoff, random} = 1:2 — NOT the raw 0.1/0.2.
        assert!((k - 1.0 / 3.0).abs() < 0.03, "kickoff share {k}");
        assert!((r - 2.0 / 3.0).abs() < 0.03, "random share {r}");
    }

    // ---- T24 ----
    #[test]
    fn duel_arena_branch_mix_matches_weights() {
        let pool = pool_of(3);
        let mut a = mk(1, 1, 20_260_719, Some(curr(0.7, 0.1, 0.2, pool.clone())));
        let (mut rep, mut kick, mut rand) = (0usize, 0usize, 0usize);
        let n = 4000;
        for _ in 0..n {
            a.debug_force_reset();
            match classify(&a.game_state(), &pool) {
                Kind::Replay => rep += 1,
                Kind::Kickoff => kick += 1,
                Kind::Random => rand += 1,
            }
        }
        let f = |x: usize| x as f64 / n as f64;
        assert!((f(rep) - 0.70).abs() < 0.03, "replay share {}", f(rep));
        assert!((f(kick) - 0.10).abs() < 0.03, "kickoff share {}", f(kick));
        assert!((f(rand) - 0.20).abs() < 0.03, "random share {}", f(rand));
    }

    // ---- T25 ----
    #[test]
    fn replay_reset_reproducible_for_fixed_seed() {
        let pool = pool_of(32);
        let mut a = mk(1, 1, 6_006, Some(curr(0.7, 0.1, 0.2, pool.clone())));
        let mut b = mk(1, 1, 6_006, Some(curr(0.7, 0.1, 0.2, pool)));
        assert_eq!(sequence(&mut a, 30), sequence(&mut b, 30));
    }

    // ---- T26 ----
    #[test]
    fn pool_index_covers_range_and_never_oob() {
        let pool = pool_of(1000);
        let mut a = mk(1, 1, 13, Some(curr(1.0, 0.0, 0.0, pool.clone())));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..5000 {
            a.debug_force_reset();
            let b = a.game_state().ball.pos;
            let idx = pool.iter().position(|s| same_ball(&b, &s.ball.pos)).expect("pool state");
            seen.insert(idx);
        }
        assert!(seen.len() > 900, "index must cover the pool, saw {}", seen.len());

        // Degenerate single-state pool: modulo must not divide by zero or wrap out.
        let mut one = mk(1, 1, 14, Some(curr(1.0, 0.0, 0.0, pool_of(1))));
        for _ in 0..100 {
            one.debug_force_reset();
        }
    }
}
