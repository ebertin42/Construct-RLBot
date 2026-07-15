//! Obs v1 entity-set builder (T3, entity-transformer plan). See
//! `docs/superpowers/plans/2026-07-16-entity-transformer-obs-v1.md`, "Obs v1
//! layout" section, for the authoritative spec this file implements.
//!
//! ## Pad API (rocketsim_rs 0.37 investigation)
//! `GameState` (from `Arena::get_game_state`) already carries
//! `pads: Vec<BoostPad>` where `BoostPad { config: BoostPadConfig { position:
//! Vec3, is_big: bool }, state: BoostPadState { is_active: bool, cooldown:
//! f32, .. } }` — no separate `pads_from_arena` helper is needed (the plan
//! anticipated one might be, in case v0 didn't expose pads; it turns out
//! `GameState` already does). For a standard soccar arena `pads.len() == 34`
//! in a FIXED arena-construction order confirmed from RocketSim's C++
//! (`Arena.cpp`, boost pad init loop): indices 0..6 are the 6 big pads (in
//! `RLConst::BoostPads::LOCS_BIG_SOCCAR` order), indices 6..34 are the 28
//! small pads (`LOCS_SMALL_SOCCAR` order). This module doesn't hardcode that
//! split — it filters on `config.is_big` at runtime, which is robust to
//! either fact.
//!
//! `BoostPadState::cooldown` is RocketSim's actual "seconds until respawn"
//! countdown (confirmed in `BoostPad.cpp`: decremented each tick while > 0,
//! `is_active = (cooldown == 0)`) — exactly the "timer" the plan wants.
//! Per the task brief, normalization is always `/10.0` (the big-pad
//! cooldown), even for small pads (whose real max cooldown is 4s, so their
//! normalized timer tops out at 0.4) — a deliberate simplification, not a bug.
//!
//! ## Scoreboard decision
//! `GameState` has no score or match-clock fields anywhere in rocketsim_rs
//! (confirmed: no `score`/`blue_score`/`orange_score`/`time_remaining` field
//! on `GameState`, `Arena`, or any car/ball state). Episode-based training has
//! no scoreboard to report. Per the task brief we do NOT invent state: the 3
//! scoreboard floats in the query row (idx 60..63) are always written 0.0.
//! The deploy adapter (a later, separate task) is responsible for filling
//! real score/time when this net runs in an actual match.
//!
//! ## Mirror-permutation approach
//! Mirroring negates x,y of every vector (play-as-blue convention, reusing
//! `obs::mir`) — including pad positions. But a pad ENTITY ROW's canonical
//! slot `k` (defined by blue's fixed arena order) must, for orange, describe
//! whichever OTHER physical pad now sits at that slot's mirrored location —
//! otherwise the row's position and its timer/is_active would describe two
//! different physical pads. We precompute a `[usize; PAD_COUNT]` involution
//! once (`mirror_pad_perm`, cached in a `OnceLock` since pad positions never
//! move): `perm[i]` = index of the pad whose position is nearest
//! `(-pos[i].x, -pos[i].y, pos[i].z)`. For canonical slot `k` sourced from
//! arena-order pad `g`, the row's SOURCE pad is `perm[g]` when mirroring
//! (else `g`), and position/timer/is_active are all read from that one
//! source pad (`mir(pads[src].config.position, mirror)`,
//! `timer_norm(pads[src])`, `pads[src].state.is_active`) — never mixing `g`'s
//! position with `perm[g]`'s timer. Because `perm` is an (approximate)
//! involution over exact-negation pad coordinates, `mir(pos_perm[g], true) ==
//! pos_g`, so orange's slot `k` position numerically matches blue's slot `k`
//! position (verified by the `orange_obs_mirrors_blue_at_kickoff` test) while
//! the timer legitimately differs (it's a different physical pad's dynamic
//! state). The SAME permutation is applied to the 34-element pad-timer vector
//! in the query row (`query[26+k] = timer_norm(pads[mirror ? perm[k] : k])`)
//! — there position isn't emitted, only the reindexed timer.
//!
//! ## prev_actions
//! `build()` intentionally does NOT take a `prev` parameter. Per the plan
//! ("Query/self row" section) prev-5 actions are embedded and summed into the
//! POOLED embedding inside the net (T5/T4), not written into the entity/query
//! tensors here — so there is nothing for this function to consume. They ride
//! separately through the engine's per-agent action-history ring buffer,
//! wired up in T6.

use std::sync::OnceLock;

use rocketsim_rs::{math::Vec3, sim::Team, BoostPad, GameState};

use crate::ballpred::BallSnap;
use crate::obs::mir;
use crate::schema::Normalization;

pub const MAX_ENT: usize = 17;
pub const ENT_FEAT: usize = 26;
pub const Q_FEAT: usize = 64;
pub const PREV_ACTIONS: usize = 5;

/// Total boost pads in a standard soccar arena (6 big + 28 small). See the
/// module doc's "Pad API" section.
const PAD_COUNT: usize = 34;
const MAX_MATES: usize = 2;
const MAX_OPPS: usize = 3;
const MAX_BIG_PADS: usize = 6;
const NUM_PRED: usize = 4;

// Entity slot layout (fixed order, see module + plan doc):
// [0]            self
// [1..3)         mates (asc car id, ≤2)
// [3..6)         opps (asc car id, ≤3)
// [6]            ball
// [7..13)        big pads (arena order, ≤6)
// [13..17)       ball-pred (ascending horizon, 4)
const SELF_IDX: usize = 0;
const MATES_START: usize = 1;
const OPPS_START: usize = MATES_START + MAX_MATES; // 3
const BALL_IDX: usize = OPPS_START + MAX_OPPS; // 6
const PADS_START: usize = BALL_IDX + 1; // 7
const PRED_START: usize = PADS_START + MAX_BIG_PADS; // 13

/// One 26-float entity row, assembled field-by-field then written with
/// `write` — keeps every index literal in one place instead of scattered
/// `out[i] = ...` lines, and matches the plan's "Entity row: 26 floats" table
/// 1:1 (see doc comment above each field).
#[derive(Debug, Clone, Copy, Default)]
struct EntityRow {
    /// [0..5): IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_PAD. Ball-pred entities
    /// deliberately get an all-zero one-hot here (see module doc: they are
    /// identified by a nonzero `horizon`, not a type flag — the plan's type
    /// one-hot table only lists 5 real categories).
    onehot: [f32; 5],
    pos: [f32; 3],
    vel: [f32; 3],
    ang_vel: [f32; 3],
    /// cars only; zero for ball/pad/pred.
    fwd: [f32; 3],
    /// cars only; zero for ball/pad/pred.
    up: [f32; 3],
    /// boost 0..1 (cars) / pad big-timer 0..1 (pads) / 0 (ball, pred).
    f20: f32,
    /// on_ground (cars) / pad is-available flag (pads) / 0 (ball, pred).
    f21: f32,
    /// has_flip (cars only) / 0 elsewhere.
    f22: f32,
    /// demoed (cars only) / 0 elsewhere.
    f23: f32,
    /// ball-pred horizon tau/2.0; 0 for every real (non-predicted) entity.
    horizon: f32,
}

impl EntityRow {
    fn write(&self, row: &mut [f32]) {
        debug_assert_eq!(row.len(), ENT_FEAT);
        row[0..5].copy_from_slice(&self.onehot);
        row[5..8].copy_from_slice(&self.pos);
        row[8..11].copy_from_slice(&self.vel);
        row[11..14].copy_from_slice(&self.ang_vel);
        row[14..17].copy_from_slice(&self.fwd);
        row[17..20].copy_from_slice(&self.up);
        row[20] = self.f20;
        row[21] = self.f21;
        row[22] = self.f22;
        row[23] = self.f23;
        row[24] = self.horizon;
        row[25] = 0.0; // reserved
    }
}

#[inline]
fn scaled(v: [f32; 3], k: f32) -> [f32; 3] {
    [v[0] * k, v[1] * k, v[2] * k]
}

/// Builds a car (self/mate/opp) entity row.
fn car_row(onehot: [f32; 5], s: &rocketsim_rs::sim::CarState, mirror: bool, pk: f32, vk: f32, ak: f32) -> EntityRow {
    EntityRow {
        onehot,
        pos: scaled(mir(s.pos, mirror), pk),
        vel: scaled(mir(s.vel, mirror), vk),
        ang_vel: scaled(mir(s.ang_vel, mirror), ak),
        fwd: mir(s.rot_mat.forward, mirror),
        up: mir(s.rot_mat.up, mirror),
        f20: s.boost / 100.0,
        f21: s.is_on_ground as u8 as f32,
        f22: s.has_flip_or_jump() as u8 as f32,
        f23: s.is_demoed as u8 as f32,
        horizon: 0.0,
    }
}

fn ball_row(pos: Vec3, vel: Vec3, ang_vel: Vec3, mirror: bool, pk: f32, vk: f32, ak: f32) -> EntityRow {
    EntityRow {
        onehot: [0.0, 0.0, 0.0, 1.0, 0.0],
        pos: scaled(mir(pos, mirror), pk),
        vel: scaled(mir(vel, mirror), vk),
        ang_vel: scaled(mir(ang_vel, mirror), ak),
        ..Default::default()
    }
}

fn pad_row(pos: Vec3, mirror: bool, pk: f32, timer_norm: f32, is_active: bool) -> EntityRow {
    EntityRow {
        onehot: [0.0, 0.0, 0.0, 0.0, 1.0],
        pos: scaled(mir(pos, mirror), pk),
        f20: timer_norm,
        f21: is_active as u8 as f32,
        ..Default::default()
    }
}

fn pred_row(snap: &BallSnap, mirror: bool, pk: f32, vk: f32, horizon: f32) -> EntityRow {
    EntityRow {
        onehot: [0.0, 0.0, 0.0, 0.0, 0.0],
        pos: scaled(mir(snap.pos, mirror), pk),
        vel: scaled(mir(snap.vel, mirror), vk),
        horizon,
        ..Default::default()
    }
}

/// 0.0 if the pad is active/available, else `cooldown / 10.0` (see module
/// doc: normalization is always by the big-pad cooldown constant, per brief).
#[inline]
fn timer_norm(pad: &BoostPad) -> f32 {
    if pad.state.is_active { 0.0 } else { pad.state.cooldown / 10.0 }
}

/// Precomputes, once, the pad-mirror involution: `perm[i]` = index of the pad
/// whose position is nearest `(-pos[i].x, -pos[i].y, pos[i].z)`. Cached in a
/// process-wide `OnceLock` because pad positions are static for the lifetime
/// of a standard-soccar arena config (only `state.is_active`/`cooldown`
/// change tick to tick) — see module doc's "Mirror-permutation approach".
fn mirror_pad_perm(pads: &[BoostPad]) -> &'static [usize; PAD_COUNT] {
    static PERM: OnceLock<[usize; PAD_COUNT]> = OnceLock::new();
    PERM.get_or_init(|| {
        let n = pads.len().min(PAD_COUNT);
        let mut perm = [0usize; PAD_COUNT];
        for i in 0..n {
            let pi = pads[i].config.position;
            let (tx, ty) = (-pi.x, -pi.y);
            let mut best = i;
            let mut best_d2 = f32::MAX;
            for (j, pad) in pads.iter().enumerate().take(n) {
                let pj = pad.config.position;
                let (dx, dy) = (pj.x - tx, pj.y - ty);
                let d2 = dx * dx + dy * dy;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = j;
                }
            }
            perm[i] = best;
        }
        // Any slots beyond `n` (defensive only — standard soccar always has
        // PAD_COUNT pads) map to themselves.
        for i in n..PAD_COUNT {
            perm[i] = i;
        }
        perm
    })
}

/// Builds the entity/mask/query tensors for one car's obs v1. See the plan's
/// "Obs v1 layout" section and this module's doc comment for the exact
/// contract. `ents` must be `MAX_ENT * ENT_FEAT` long, `mask` `MAX_ENT` long,
/// `query` `Q_FEAT` long. Deliberately takes no `prev` parameter — see module
/// doc's "prev_actions" section.
pub fn build(
    state: &GameState,
    car_idx: usize,
    pred: &[BallSnap; NUM_PRED],
    n: &Normalization,
    ents: &mut [f32],
    mask: &mut [bool],
    query: &mut [f32],
) {
    assert_eq!(ents.len(), MAX_ENT * ENT_FEAT);
    assert_eq!(mask.len(), MAX_ENT);
    assert_eq!(query.len(), Q_FEAT);
    ents.fill(0.0);
    query.fill(0.0);
    mask.fill(true); // default: absent/masked; explicitly un-masked below

    let me = &state.cars[car_idx];
    let mirror = me.team == Team::Orange;
    let (pk, vk, ak) = (n.pos_norm as f32, n.vel_norm as f32, n.ang_vel_norm as f32);

    // --- self ---
    let self_row = car_row([1.0, 0.0, 0.0, 0.0, 0.0], &me.state, mirror, pk, vk, ak);
    self_row.write(&mut ents[SELF_IDX * ENT_FEAT..(SELF_IDX + 1) * ENT_FEAT]);
    mask[SELF_IDX] = false;

    // --- mates / opps (asc car id) ---
    let mut mates: Vec<&rocketsim_rs::CarInfo> = state
        .cars
        .iter()
        .enumerate()
        .filter(|(i, c)| *i != car_idx && c.team == me.team)
        .map(|(_, c)| c)
        .collect();
    mates.sort_by_key(|c| c.id);
    for (slot, c) in mates.into_iter().take(MAX_MATES).enumerate() {
        let row = car_row([0.0, 1.0, 0.0, 0.0, 0.0], &c.state, mirror, pk, vk, ak);
        let idx = MATES_START + slot;
        row.write(&mut ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT]);
        mask[idx] = false;
    }

    let mut opps: Vec<&rocketsim_rs::CarInfo> =
        state.cars.iter().filter(|c| c.team != me.team).collect();
    opps.sort_by_key(|c| c.id);
    for (slot, c) in opps.into_iter().take(MAX_OPPS).enumerate() {
        let row = car_row([0.0, 0.0, 1.0, 0.0, 0.0], &c.state, mirror, pk, vk, ak);
        let idx = OPPS_START + slot;
        row.write(&mut ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT]);
        mask[idx] = false;
    }

    // --- ball ---
    let b = &state.ball;
    let brow = ball_row(b.pos, b.vel, b.ang_vel, mirror, pk, vk, ak);
    brow.write(&mut ents[BALL_IDX * ENT_FEAT..(BALL_IDX + 1) * ENT_FEAT]);
    mask[BALL_IDX] = false;

    // --- big pads (fixed arena order) ---
    let pads = &state.pads;
    let perm = mirror_pad_perm(pads);
    let big_indices: Vec<usize> =
        pads.iter().enumerate().filter(|(_, p)| p.config.is_big).map(|(i, _)| i).collect();
    for (slot, &g) in big_indices.iter().take(MAX_BIG_PADS).enumerate() {
        // Both the position AND the dynamic state (timer/is_active) must come
        // from the SAME source pad: for orange that's `perm[g]` (the pad that
        // physically sits at the mirrored location), not `g` itself — mixing
        // g's raw position with perm[g]'s timer would put a pad's dynamic
        // state at the wrong mirrored coordinates. See module doc's
        // "Mirror-permutation approach".
        let src = if mirror { perm[g] } else { g };
        let pad_src = &pads[src];
        let row =
            pad_row(pad_src.config.position, mirror, pk, timer_norm(pad_src), pad_src.state.is_active);
        let idx = PADS_START + slot;
        row.write(&mut ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT]);
        mask[idx] = false;
    }

    // --- ball prediction (ascending horizon) ---
    for (slot, snap) in pred.iter().enumerate() {
        let horizon = (slot as f32 + 1.0) * 0.25; // 0.25, 0.5, 0.75, 1.0
        let row = pred_row(snap, mirror, pk, vk, horizon);
        let idx = PRED_START + slot;
        row.write(&mut ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT]);
        mask[idx] = false;
    }

    // --- query row: self (26) + 34 pad timers + 3 scoreboard (zero) + 1 reserved ---
    self_row.write(&mut query[0..ENT_FEAT]);
    let n_pads = pads.len().min(PAD_COUNT);
    for k in 0..n_pads {
        let src = if mirror { perm[k] } else { k };
        query[ENT_FEAT + k] = timer_norm(&pads[src]);
    }
    // query[ENT_FEAT + PAD_COUNT .. ENT_FEAT + PAD_COUNT + 3] = scoreboard: no
    // score/time state exists in this engine (see module doc) -> left 0.0.
    // query[63] = reserved -> left 0.0.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;
    use crate::sim_init::ensure_init;
    use rocketsim_rs::sim::{Arena, CarConfig};

    fn norm() -> Normalization {
        Schema::load("../schema/v1.toml").unwrap().normalization
    }

    fn kickoff_1v1() -> GameState {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(7));
        arena.pin_mut().get_game_state()
    }

    fn zero_pred() -> [BallSnap; NUM_PRED] {
        [BallSnap::default(); NUM_PRED]
    }

    #[test]
    fn layout_golden_1v1_kickoff() {
        let gs = kickoff_1v1();
        let pred = zero_pred();
        let nrm = norm();
        let mut ents = [0.0f32; MAX_ENT * ENT_FEAT];
        let mut mask = [false; MAX_ENT];
        let mut query = [0.0f32; Q_FEAT];
        build(&gs, 0, &pred, &nrm, &mut ents, &mut mask, &mut query);

        // self: IS_SELF one-hot, present
        assert_eq!(&ents[0..5], &[1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(!mask[SELF_IDX]);

        // 1v1: no mates, one opponent (asc id -> car 1) in opps[0], rest masked
        assert!(mask[MATES_START]);
        assert!(mask[MATES_START + 1]);
        assert!(!mask[OPPS_START], "the single opponent must be present");
        let opp_row = &ents[OPPS_START * ENT_FEAT..(OPPS_START + 1) * ENT_FEAT];
        assert_eq!(&opp_row[0..5], &[0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(mask[OPPS_START + 1]);
        assert!(mask[OPPS_START + 2]);

        // ball: IS_BALL one-hot, present, position matches normalized arena ball
        assert!(!mask[BALL_IDX]);
        let ball_row_out = &ents[BALL_IDX * ENT_FEAT..(BALL_IDX + 1) * ENT_FEAT];
        assert_eq!(&ball_row_out[0..5], &[0.0, 0.0, 0.0, 1.0, 0.0]);
        let pk = nrm.pos_norm as f32;
        assert!((ball_row_out[5] - gs.ball.pos.x * pk).abs() < 1e-5);
        assert!((ball_row_out[6] - gs.ball.pos.y * pk).abs() < 1e-5);
        assert!((ball_row_out[7] - gs.ball.pos.z * pk).abs() < 1e-5);

        // 6 big pads: IS_PAD one-hot, present, is_big positions (|x| or |y| large & z ~73)
        for slot in 0..6 {
            let idx = PADS_START + slot;
            assert!(!mask[idx], "pad slot {slot} must be present");
            let row = &ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT];
            assert_eq!(&row[0..5], &[0.0, 0.0, 0.0, 0.0, 1.0], "pad slot {slot} one-hot");
        }

        // 4 ball-pred: present, horizon feature exactly 0.25/0.5/0.75/1.0, all-zero type one-hot
        let expected_horizons = [0.25f32, 0.5, 0.75, 1.0];
        for slot in 0..4 {
            let idx = PRED_START + slot;
            assert!(!mask[idx], "pred slot {slot} must be present");
            let row = &ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT];
            assert_eq!(&row[0..5], &[0.0, 0.0, 0.0, 0.0, 0.0], "pred slot {slot} type one-hot");
            assert!((row[24] - expected_horizons[slot]).abs() < 1e-6, "pred slot {slot} horizon");
        }

        assert!(ents.iter().all(|x| x.is_finite()));
        assert!(query.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn orange_obs_mirrors_blue_at_kickoff() {
        // Kickoff spawns are 180deg-rotation symmetric, exactly the same
        // invariant obs.rs's mirror test relies on.
        let gs = kickoff_1v1();
        let pred = zero_pred();
        let nrm = norm();
        let (mut eb, mut eo) = ([0.0f32; MAX_ENT * ENT_FEAT], [0.0f32; MAX_ENT * ENT_FEAT]);
        let (mut mb, mut mo) = ([false; MAX_ENT], [false; MAX_ENT]);
        let (mut qb, mut qo) = ([0.0f32; Q_FEAT], [0.0f32; Q_FEAT]);
        build(&gs, 0, &pred, &nrm, &mut eb, &mut mb, &mut qb);
        build(&gs, 1, &pred, &nrm, &mut eo, &mut mo, &mut qo);
        for i in 0..eb.len() {
            assert!((eb[i] - eo[i]).abs() < 1e-4, "ent idx {i}: {} vs {}", eb[i], eo[i]);
        }
        assert_eq!(mb, mo);
        for i in 0..qb.len() {
            assert!((qb[i] - qo[i]).abs() < 1e-4, "query idx {i}: {} vs {}", qb[i], qo[i]);
        }
    }

    #[test]
    fn nonexistent_mate_and_opp_slots_are_zero_and_masked() {
        let gs = kickoff_1v1();
        let pred = zero_pred();
        let nrm = norm();
        let mut ents = [0.0f32; MAX_ENT * ENT_FEAT];
        let mut mask = [false; MAX_ENT];
        let mut query = [0.0f32; Q_FEAT];
        build(&gs, 0, &pred, &nrm, &mut ents, &mut mask, &mut query);
        for idx in [MATES_START, MATES_START + 1, OPPS_START + 1, OPPS_START + 2] {
            assert!(mask[idx], "slot {idx} must be masked");
            let row = &ents[idx * ENT_FEAT..(idx + 1) * ENT_FEAT];
            assert!(row.iter().all(|&x| x == 0.0), "slot {idx} must be zero row");
        }
    }
}
