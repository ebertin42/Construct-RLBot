//! Necto's `NectoObsBuilder`. Same tensor widths as Nexto (q32 / kv24) and the
//! same `_norm` / `_invert` vectors, but the CONTENT differs in four ways that
//! are invisible from the shapes -- feeding it Nexto's observation produces a bot
//! driving on nonsense (this happened; see docs/training-journal.md 2026-07-22):
//!
//!   1. Entity order is `[ball, players.., boosts..]` -- the BALL IS FIRST and
//!      players are offset by one (Nexto puts players first).
//!   2. The relative transform subtracts position AND linear velocity (cols 5:11)
//!      and does NOT rotate into the self heading (Nexto subtracts position only,
//!      then rotates five vector triples by `atan2(fw.x, fw.y)`).
//!   3. `is_teammate` / `is_opponent` are keyed to BLUE, then the two columns are
//!      SWAPPED for an orange viewer (Nexto keys them to the viewer directly).
//!   4. Column 21 is a STATEFUL timer -- a per-pad respawn timer and a per-car
//!      demo timer carried across frames -- not a binary flag.
//!
//! Reproduces deploy/external/necto/necto_obs.py; golden-tested against that real
//! builder in necto_obs_test.rs.
use crate::obs_advanced::{rlgym_to_canon, BOOST_LOCATIONS};
use rocketsim_rs::sim::Team;
use rocketsim_rs::GameState;

pub use crate::obs_nexto::{NEXTO_KV as NECTO_KV, NEXTO_Q as NECTO_Q};
pub const NECTO_PADS: usize = 34;
/// Necto's builder defaults to tick_skip 8, which is also our engine's.
const TICK_SKIP: f32 = 8.0;

// column indices (shared with Nexto)
const IS_MAIN: usize = 0;
const IS_MATE: usize = 1;
const IS_OPP: usize = 2;
const IS_BALL: usize = 3;
const IS_BOOST: usize = 4;
const POS: usize = 5;
const LIN_VEL: usize = 8;
const FW: usize = 11;
const UP: usize = 14;
const ANG_VEL: usize = 17;
const BOOST: usize = 20;
const TIMER: usize = 21;
const ON_GROUND: usize = 22;
const HAS_FLIP: usize = 23;

#[inline]
fn norm_at(i: usize) -> f32 {
    match i {
        0..=4 => 1.0,
        5..=10 => 2300.0,
        11..=16 => 1.0,
        17..=19 => 5.5,
        _ => 1.0,
    }
}

#[inline]
fn invert_at(i: usize) -> f32 {
    if !(POS..BOOST).contains(&i) {
        return 1.0;
    }
    match (i - POS) % 3 {
        0 | 1 => -1.0,
        _ => 1.0,
    }
}

/// Per-ARENA stateful timers. Necto carries these across frames, so they must be
/// kept per arena (not per car) and advanced exactly once per FRAME.
///
/// "Once per frame" is the whole point, and it was WRONG until 2026-07-26. The
/// reference `NectoObsBuilder.build_obs` is called once per frame for the whole
/// team; ours is called once per CAR, so in a 2v2/3v3 arena the timers advanced
/// 2-3 times per frame and each car saw a DIFFERENT observation of the same
/// world. Measured, same frame, three cars: pad col21 read
/// [1.0, 0.99333, 0.98667] and demo col21 [0.30, 0.29333, 0.28667] -- i.e. cars
/// 2 and 3 were fed a boost-pad respawn clock that had silently run forward.
/// Invisible at 1v1 (one car per team per frame), which is why it survived
/// until team foreign arenas became expressible.
///
/// The fix keys the advance on `state.tick_count` and caches the values EMITTED
/// for that frame, rather than just skipping the update: the boost timer emits
/// its value BEFORE its own decrement (reproducing the reference's ordering), so
/// "don't advance" and "emit what the first car emitted" are not the same thing
/// -- they differ by exactly the TICK_SKIP/1200 = 0.00667 seen in the measurement
/// above.
///
/// NOTE (unchanged, and deliberately so): with `decision_period > 1` a held
/// frame returns before `build_necto_obs` runs at all, so the timers advance
/// once per THINKING frame, not once per simulated frame. That is pre-existing
/// behaviour that every rung of the existing difficulty ladder was measured
/// against; do not "fix" it in the same change as this one.
pub struct NectoTimers {
    boost: [f32; NECTO_PADS],
    demo: Vec<f32>,
    /// Values emitted for `last_tick`'s frame (see the struct doc): the boost
    /// timer is emitted pre-decrement, so this cannot be recomputed from the
    /// live state after the fact.
    boost_emit: [f32; NECTO_PADS],
    demo_emit: Vec<f32>,
    /// `state.tick_count` of the frame the caches above describe. `None` before
    /// the first build (and after an episode boundary clears the whole struct).
    last_tick: Option<u64>,
}

impl NectoTimers {
    pub fn new(n_players: usize) -> Self {
        Self {
            boost: [0.0; NECTO_PADS],
            demo: vec![0.0; n_players],
            boost_emit: [0.0; NECTO_PADS],
            demo_emit: vec![0.0; n_players],
            last_tick: None,
        }
    }

    /// Advance one frame and latch what this frame emits. `order` is the
    /// ascending-car-id player order, which is frame-global (identical for
    /// every car of the frame), so the demo timers stay index-stable.
    fn advance(&mut self, state: &GameState, order: &[usize]) {
        let n_players = order.len();
        if self.demo.len() < n_players {
            self.demo.resize(n_players, 0.0);
        }
        if self.demo_emit.len() < n_players {
            self.demo_emit.resize(n_players, 0.0);
        }
        for pi in 0..n_players {
            // demo timer: reset to 3 when it hits 0, else count down. This is
            // what the reference does (it never reads is_demoed) -- odd, but
            // reproduced.
            let t = &mut self.demo[pi];
            if *t <= 0.0 {
                *t = 3.0;
            } else {
                *t = (*t - TICK_SKIP / 120.0).max(0.0);
            }
            self.demo_emit[pi] = *t;
        }
        let perm = rlgym_to_canon(&state.pads);
        for j in 0..NECTO_PADS {
            let active = state.pads[perm[j]].state.is_active;
            let is_big = BOOST_LOCATIONS[j][2] > 72.0;
            // new grab: pad available while its timer is zero
            if active && self.boost[j] == 0.0 {
                self.boost[j] = 0.4 + 0.6 * (is_big as u8 as f32);
            }
            self.boost[j] *= active as u8 as f32;
            self.boost_emit[j] = self.boost[j]; // emitted BEFORE the decrement
            self.boost[j] -= TICK_SKIP / 1200.0;
            if self.boost[j] < 0.0 {
                self.boost[j] = 0.0;
            }
        }
    }
}

pub fn n_entities(n_players: usize) -> usize {
    1 + n_players + NECTO_PADS
}

/// Build Necto's (q, kv, mask) for `state.cars[car_idx]`, advancing `timers` one
/// frame IF this is the first car of that frame (keyed on `state.tick_count`;
/// see `NectoTimers`). Players are ordered by ascending car id (rlgym_compat
/// sorts them so).
pub fn build_necto_obs(
    state: &GameState,
    car_idx: usize,
    prev_action: &[f32; 8],
    timers: &mut NectoTimers,
    q: &mut [f32],
    kv: &mut [f32],
    mask: &mut [bool],
) {
    let n_players = state.cars.len();
    let n_ent = n_entities(n_players);
    assert_eq!(q.len(), NECTO_Q);
    assert_eq!(kv.len(), n_ent * NECTO_KV);
    assert_eq!(mask.len(), n_ent);
    kv.fill(0.0);
    q.fill(0.0);
    mask.fill(false);

    let mut order: Vec<usize> = (0..n_players).collect();
    order.sort_by_key(|&i| state.cars[i].id);
    let self_row = 1 + order.iter().position(|&i| i == car_idx).expect("car in state");
    let orange = state.cars[car_idx].team == Team::Orange;

    // ONE advance per frame, however many cars of this arena ask for an obs.
    if timers.last_tick != Some(state.tick_count) {
        timers.advance(state, &order);
        timers.last_tick = Some(state.tick_count);
    }

    // --- ball (entity 0) ---
    {
        let b = &state.ball;
        kv[IS_BALL] = 1.0;
        kv[POS] = b.pos.x;
        kv[POS + 1] = b.pos.y;
        kv[POS + 2] = b.pos.z;
        kv[LIN_VEL] = b.vel.x;
        kv[LIN_VEL + 1] = b.vel.y;
        kv[LIN_VEL + 2] = b.vel.z;
        kv[ANG_VEL] = b.ang_vel.x;
        kv[ANG_VEL + 1] = b.ang_vel.y;
        kv[ANG_VEL + 2] = b.ang_vel.z;
    }

    // --- players (entities 1..1+n) ---
    for (pi, &ci) in order.iter().enumerate() {
        let c = &state.cars[ci];
        let s = &c.state;
        let o = (1 + pi) * NECTO_KV;
        // keyed to BLUE, swapped later for an orange viewer
        if c.team == Team::Blue {
            kv[o + IS_MATE] = 1.0;
        } else {
            kv[o + IS_OPP] = 1.0;
        }
        kv[o + POS] = s.pos.x;
        kv[o + POS + 1] = s.pos.y;
        kv[o + POS + 2] = s.pos.z;
        kv[o + LIN_VEL] = s.vel.x;
        kv[o + LIN_VEL + 1] = s.vel.y;
        kv[o + LIN_VEL + 2] = s.vel.z;
        kv[o + FW] = s.rot_mat.forward.x;
        kv[o + FW + 1] = s.rot_mat.forward.y;
        kv[o + FW + 2] = s.rot_mat.forward.z;
        kv[o + UP] = s.rot_mat.up.x;
        kv[o + UP + 1] = s.rot_mat.up.y;
        kv[o + UP + 2] = s.rot_mat.up.z;
        kv[o + ANG_VEL] = s.ang_vel.x;
        kv[o + ANG_VEL + 1] = s.ang_vel.y;
        kv[o + ANG_VEL + 2] = s.ang_vel.z;
        kv[o + BOOST] = s.boost / 100.0; // rlgym stores a 0..1 fraction
        kv[o + ON_GROUND] = s.is_on_ground as u8 as f32;
        kv[o + HAS_FLIP] = s.has_flip_or_jump() as u8 as f32;
        // demo timer latched by `NectoTimers::advance` for THIS frame -- every
        // car of the frame must read the same number.
        kv[o + TIMER] = timers.demo_emit[pi] / 10.0;
    }

    // --- boost pads (entities 1+n..) in rlgym BOOST_LOCATIONS order ---
    for (j, loc) in BOOST_LOCATIONS.iter().enumerate().take(NECTO_PADS) {
        let is_big = loc[2] > 72.0;
        let o = (1 + n_players + j) * NECTO_KV;
        kv[o + IS_BOOST] = 1.0;
        kv[o + POS] = loc[0];
        kv[o + POS + 1] = loc[1];
        kv[o + POS + 2] = loc[2];
        kv[o + BOOST] = 0.12 + 0.88 * (is_big as u8 as f32);
        kv[o + TIMER] = timers.boost_emit[j]; // pre-decrement value, per frame
    }

    // --- normalise ---
    for row in 0..n_ent {
        let o = row * NECTO_KV;
        for i in 0..NECTO_KV {
            kv[o + i] /= norm_at(i);
        }
    }

    // --- is_main, then orange team-swap + inversion ---
    kv[self_row * NECTO_KV + IS_MAIN] = 1.0;
    if orange {
        for row in 0..n_ent {
            let o = row * NECTO_KV;
            kv.swap(o + IS_MATE, o + IS_OPP);
            for i in 0..NECTO_KV {
                kv[o + i] *= invert_at(i);
            }
        }
    }

    // --- query = self row ++ previous action ---
    let so = self_row * NECTO_KV;
    q[..NECTO_KV].copy_from_slice(&kv[so..so + NECTO_KV]);
    q[NECTO_KV..NECTO_Q].copy_from_slice(prev_action);

    // --- relative: subtract self POSITION AND VELOCITY; no rotation ---
    let base: Vec<f32> = q[POS..LIN_VEL + 3].to_vec();
    for row in 0..n_ent {
        let o = row * NECTO_KV;
        for (k, b) in base.iter().enumerate() {
            kv[o + POS + k] -= *b;
        }
    }
}
