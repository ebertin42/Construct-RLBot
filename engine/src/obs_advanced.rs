//! Immortal's RLGym AdvancedObs (107 floats, 1v1) built from the engine GameState.
//! Distinct from `obs.rs` (our 94-float v0 obs): different normalizers
//! (pos/vel /2300, ang_vel /PI), includes 34 boost-pad states + previous action,
//! and uses the forward/up basis (not Euler). Reproduces
//! rlmarlbot/immortal/obs/advanced_obs.py byte-exact -- see immortal_obs_test.rs.
//!
//! ## Boost pads
//! Immortal's obs consumes `state.boost_pads` in rlgym's BOOST_LOCATIONS order,
//! which is NOT the engine's RocketSim canonical order (6 big then 28 small). We
//! reorder canonical -> rlgym by nearest-xy position match (`rlgym_to_canon`).
//! Orange uses `inverted_boost_pads` = the rlgym array reversed (rlgym order is
//! reverse-antisymmetric within ~2uu, so reverse == physical mirror):
//!   obs_pad[j] = state.pads[ rlgym_to_canon[ (33-j) if orange else j ] ].is_active
use crate::obs::mir;
use rocketsim_rs::math::Vec3;
use rocketsim_rs::sim::Team;
use rocketsim_rs::{BoostPad, GameState};
use std::sync::OnceLock;

pub const ADV_OBS_SIZE: usize = 107;
const POS_STD: f32 = 2300.0;
const ANG_STD: f32 = std::f32::consts::PI;

/// rlgym BOOST_LOCATIONS (rlgym_compat.common_values, verified) -- the order
/// Immortal's obs `pads` are in.
pub(crate) const BOOST_LOCATIONS: [[f32; 3]; 34] = [
    [0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0], [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0], [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0], [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0], [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0], [-2048.0, 1036.0, 70.0], [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0], [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0], [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0], [940.0, 3308.0, 70.0], [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
];

/// perm[j] = engine (canonical) pad index whose position is nearest
/// BOOST_LOCATIONS[j] (xy). Static -- pad positions never move -- cached
/// process-wide (standard soccar always has 34 pads).
pub(crate) fn rlgym_to_canon(pads: &[BoostPad]) -> &'static [usize; 34] {
    static PERM: OnceLock<[usize; 34]> = OnceLock::new();
    PERM.get_or_init(|| {
        let mut perm = [0usize; 34];
        for (j, loc) in BOOST_LOCATIONS.iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f32::MAX;
            for (i, p) in pads.iter().enumerate() {
                let pp = p.config.position;
                let d = (pp.x - loc[0]).powi(2) + (pp.y - loc[1]).powi(2);
                if d < bd { bd = d; best = i; }
            }
            perm[j] = best;
        }
        perm
    })
}

struct W<'a> { out: &'a mut [f32], i: usize }
impl W<'_> {
    #[inline]
    fn v3(&mut self, v: [f32; 3], k: f32) {
        self.out[self.i] = v[0] * k;
        self.out[self.i + 1] = v[1] * k;
        self.out[self.i + 2] = v[2] * k;
        self.i += 3;
    }
    #[inline]
    fn f(&mut self, x: f32) { self.out[self.i] = x; self.i += 1; }
}

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 { Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z) }

#[inline]
fn mir_vec(v: Vec3, m: bool) -> Vec3 { let a = mir(v, m); Vec3::new(a[0], a[1], a[2]) }

/// Append one player's 25-float block. Returns the mirrored (pos, vel) so the
/// caller can compute the enemy rel-extras.
fn add_player(w: &mut W, car: &rocketsim_rs::CarInfo, ball_p: Vec3, ball_v: Vec3, m: bool)
    -> (Vec3, Vec3)
{
    let s = &car.state;
    let pos = mir_vec(s.pos, m);
    let vel = mir_vec(s.vel, m);
    w.v3(mir(sub(ball_p, s.pos), m), 1.0 / POS_STD); // rel_pos
    w.v3(mir(sub(ball_v, s.vel), m), 1.0 / POS_STD); // rel_vel
    w.v3(mir(s.pos, m), 1.0 / POS_STD);
    w.v3(mir(s.rot_mat.forward, m), 1.0);
    w.v3(mir(s.rot_mat.up, m), 1.0);
    w.v3(mir(s.vel, m), 1.0 / POS_STD);
    w.v3(mir(s.ang_vel, m), 1.0 / ANG_STD);
    w.f(s.boost / 100.0);
    w.f(s.is_on_ground as u8 as f32);
    w.f(s.has_flip_or_jump() as u8 as f32);
    w.f(s.is_demoed as u8 as f32);
    (pos, vel)
}

/// Shared body of the AdvancedObs builders. Writes ball 9 + prev_action 8 +
/// pads 34 + self 25 = 76 floats, then a 31-float block per car in `others`
/// (25 player floats + 6 rel-extras). `others` holds indices into
/// `state.cars`, already in the order the net expects to read them.
///
/// Only the `others` list is car-count-dependent -- ball, pads, prev_action
/// and self are identical at every team size, which is what makes a truncated
/// obs well-defined at all.
fn build_advanced_obs_inner(
    state: &GameState,
    car_idx: usize,
    others: &[usize],
    prev_action: &[f32; 8],
    out: &mut [f32],
) {
    let me = &state.cars[car_idx];
    let m = me.team == Team::Orange;
    let mut w = W { out, i: 0 };

    let b = &state.ball;
    w.v3(mir(b.pos, m), 1.0 / POS_STD);
    w.v3(mir(b.vel, m), 1.0 / POS_STD);
    w.v3(mir(b.ang_vel, m), 1.0 / ANG_STD);
    for k in 0..8 { w.f(prev_action[k]); }

    let perm = rlgym_to_canon(&state.pads);
    for j in 0..34 {
        let rj = if m { 33 - j } else { j };
        w.f(state.pads[perm[rj]].state.is_active as u8 as f32);
    }

    let (self_pos, self_vel) = add_player(&mut w, me, b.pos, b.vel, m);

    for &oi in others {
        let (opos, ovel) = add_player(&mut w, &state.cars[oi], b.pos, b.vel, m);
        w.v3([opos.x - self_pos.x, opos.y - self_pos.y, opos.z - self_pos.z], 1.0 / POS_STD);
        w.v3([ovel.x - self_vel.x, ovel.y - self_vel.y, ovel.z - self_vel.z], 1.0 / POS_STD);
    }
}

/// Build Immortal's 107-float AdvancedObs for `state.cars[car_idx]` given the
/// previous action (8 floats). `out` must be `ADV_OBS_SIZE` long.
///
/// UNCHANGED BEHAVIOUR: shows every other car, in ally-then-enemy order. That
/// only fits `ADV_OBS_SIZE` when there is exactly one other car (1v1); above
/// that this runs off the end of `out` and panics, exactly as before. Use
/// `build_advanced_obs_one_other` for team arenas.
pub fn build_advanced_obs(state: &GameState, car_idx: usize, prev_action: &[f32; 8], out: &mut [f32]) {
    assert_eq!(out.len(), ADV_OBS_SIZE);
    let me = &state.cars[car_idx];
    // allies (same team, ascending id) then enemies -- 1v1 has only the enemy.
    let mut others: Vec<usize> = (0..state.cars.len()).filter(|i| *i != car_idx).collect();
    others.sort_by_key(|&i| (state.cars[i].team != me.team, state.cars[i].id));
    build_advanced_obs_inner(state, car_idx, &others, prev_action, out);
    debug_assert_eq!(ADV_OBS_SIZE, 107);
}

/// The index into `state.cars` of the OPPONENT of `car_idx` that is closest to
/// the ball, or `None` if `car_idx` has no opponents.
///
/// The selector for the truncated obs below. `near-ball` rather than
/// `near-self` or `lowest-id` because it is self-correcting against the
/// obvious exploit: with `near-self` our net can drive car B at the ball from
/// range while car A parks next to the bot and hogs the visible slot, whereas
/// with `near-ball` whichever of our cars is contesting the ball IS the
/// visible one, by construction -- the hidden car is always the one least able
/// to punish at that instant. `lowest-id` is additionally degenerate when the
/// bot drives several cars: they would all watch the same opponent (measured
/// as the worst selector, immortal 3v3 fc3 facing 0.484 vs 0.519 near-ball).
pub fn nearest_opponent_to_ball(state: &GameState, car_idx: usize) -> Option<usize> {
    let my_team = state.cars[car_idx].team;
    let b = state.ball.pos;
    state
        .cars
        .iter()
        .enumerate()
        .filter(|(i, c)| *i != car_idx && c.team != my_team)
        .min_by(|(_, a), (_, c)| {
            let d = |p: Vec3| {
                let (dx, dy, dz) = (p.x - b.x, p.y - b.y, p.z - b.z);
                dx * dx + dy * dy + dz * dz
            };
            d(a.state.pos).total_cmp(&d(c.state.pos))
        })
        .map(|(i, _)| i)
}

/// Build the 107-float AdvancedObs for `state.cars[car_idx]` showing EXACTLY
/// ONE other car, `other_idx`. `out` must be `ADV_OBS_SIZE` long.
///
/// WHY THIS EXISTS. Immortal's and Element's extracted weights have a hard
/// 107-wide input (immortal `net.0.weight` [512,107], element `fc1.weight`
/// [256,107]) and 107 = ball 9 + prev_action 8 + pads 34 + self 25 + 31 *
/// (other cars), i.e. exactly ONE other car. Upstream RLMarlbot's
/// `advanced_obs.py` loops over all other cars with no padding and no
/// truncation, so upstream would emit 169 floats into a 107-wide net at 2v2
/// and crash -- the limitation is a property of THE EXTRACTED ARTIFACT, not of
/// the bot's competence. Nothing stops us feeding it a truncated obs so it can
/// drive a car in a team arena.
///
/// This matters because difficulty is wildly uneven across our opponent set
/// (at a shared p4: immortal 0.896, element 0.677, nexto 0.146) and the two
/// EASIEST bots were excluded from exactly the 2v2/3v3 blocks where the net is
/// weakest and the rung controller has nowhere easier to go. They are also
/// ~5x cheaper per car-forward than nexto at 3v3 (0.23-0.26 ms vs 1.29 ms).
///
/// MEASURED (probe over 4000 steps/row x 3 seeds vs a random opponent): a
/// truncated element/immortal at 2v2 and 3v3 is indistinguishable from the
/// same bot at 1v1 on every behavioural axis -- goals/600s 58+-5 vs 61+-5
/// (element 3v3 vs 1v1), ball-facing 0.570 vs 0.555, approach 0.579 vs 0.531,
/// no spin (ground |ang_vel.z| 0.8-1.1 rad/s, nowhere near the 5.5 cap), no
/// idling, zero insane physics states. Against nexto (which sees the whole
/// arena) the truncated bots track a NON-truncated necto control cell for
/// cell as team size grows, so the collapse with team size is the cost of
/// playing nexto, not the cost of truncation.
///
/// `other_idx` MUST be an opponent. The 31-float block lands at offset 76..107,
/// the slot the net was trained to read as the ENEMY; passing a teammate
/// silently writes an ally into the enemy slot (measured as consistently the
/// worst selector). Use `nearest_opponent_to_ball` to choose it.
///
/// At 1v1 this is byte-identical to `build_advanced_obs` (there is only one
/// other car and it is the enemy), which is what keeps every existing 1v1
/// element/immortal bench row comparable.
pub fn build_advanced_obs_one_other(
    state: &GameState,
    car_idx: usize,
    other_idx: usize,
    prev_action: &[f32; 8],
    out: &mut [f32],
) {
    assert_eq!(out.len(), ADV_OBS_SIZE);
    debug_assert_ne!(other_idx, car_idx);
    debug_assert_ne!(
        state.cars[other_idx].team, state.cars[car_idx].team,
        "the shown car occupies the ENEMY slot; showing a teammate mis-types it"
    );
    build_advanced_obs_inner(state, car_idx, &[other_idx], prev_action, out);
}
