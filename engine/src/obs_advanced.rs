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

/// Build Immortal's 107-float AdvancedObs for `state.cars[car_idx]` given the
/// previous action (8 floats). `out` must be `ADV_OBS_SIZE` long.
pub fn build_advanced_obs(state: &GameState, car_idx: usize, prev_action: &[f32; 8], out: &mut [f32]) {
    assert_eq!(out.len(), ADV_OBS_SIZE);
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

    // allies (same team, ascending id) then enemies -- 1v1 has only the enemy.
    let mut others: Vec<&rocketsim_rs::CarInfo> = state.cars.iter()
        .enumerate().filter(|(i, _)| *i != car_idx).map(|(_, c)| c).collect();
    others.sort_by_key(|c| (c.team != me.team, c.id));
    for c in others {
        let (opos, ovel) = add_player(&mut w, c, b.pos, b.vel, m);
        w.v3([opos.x - self_pos.x, opos.y - self_pos.y, opos.z - self_pos.z], 1.0 / POS_STD);
        w.v3([ovel.x - self_vel.x, ovel.y - self_vel.y, ovel.z - self_vel.z], 1.0 / POS_STD);
    }
    debug_assert_eq!(w.i, ADV_OBS_SIZE);
}
