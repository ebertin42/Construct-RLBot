use crate::schema::Normalization;
use rocketsim_rs::{
    math::Vec3,
    sim::Team,
    GameState,
};

pub const OBS_SIZE: usize = 94;
pub const MAX_OTHERS: usize = 5;

/// Play-as-blue mirroring: orange negates x,y of every vector (z/height is
/// unaffected — mirroring is a horizontal flip through the field center).
/// `pub(crate)` so `obs_v1` reuses the exact v0 convention instead of
/// duplicating it (see plan's "Obs v1 layout": "mirroring = play-as-blue
/// ... exactly as v0's `mir`").
#[inline]
pub(crate) fn mir(v: Vec3, mirror: bool) -> [f32; 3] {
    if mirror { [-v.x, -v.y, v.z] } else { [v.x, v.y, v.z] }
}

struct W<'a> {
    out: &'a mut [f32],
    i: usize,
}
impl W<'_> {
    #[inline]
    fn v3(&mut self, v: [f32; 3], k: f32) {
        self.out[self.i] = v[0] * k;
        self.out[self.i + 1] = v[1] * k;
        self.out[self.i + 2] = v[2] * k;
        self.i += 3;
    }
    #[inline]
    fn f(&mut self, x: f32) {
        self.out[self.i] = x;
        self.i += 1;
    }
}

pub fn build_obs(state: &GameState, car_idx: usize, n: &Normalization, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let me = &state.cars[car_idx];
    let mirror = me.team == Team::Orange;
    let (pk, vk, ak) = (n.pos_norm as f32, n.vel_norm as f32, n.ang_vel_norm as f32);
    let ms = &me.state;
    let mut w = W { out, i: 0 };

    // self [0:19]
    w.v3(mir(ms.pos, mirror), pk);
    w.v3(mir(ms.rot_mat.forward, mirror), 1.0);
    w.v3(mir(ms.rot_mat.up, mirror), 1.0);
    w.v3(mir(ms.vel, mirror), vk);
    w.v3(mir(ms.ang_vel, mirror), ak);
    w.f(ms.boost / 100.0);
    w.f(ms.is_on_ground as u8 as f32);
    w.f(ms.has_flip_or_jump() as u8 as f32);
    w.f(ms.is_demoed as u8 as f32);

    // ball [19:34]
    let b = &state.ball;
    w.v3(mir(b.pos, mirror), pk);
    w.v3(mir(b.vel, mirror), vk);
    w.v3(mir(b.ang_vel, mirror), ak);
    let rel_p = Vec3::new(b.pos.x - ms.pos.x, b.pos.y - ms.pos.y, b.pos.z - ms.pos.z);
    let rel_v = Vec3::new(b.vel.x - ms.vel.x, b.vel.y - ms.vel.y, b.vel.z - ms.vel.z);
    w.v3(mir(rel_p, mirror), pk);
    w.v3(mir(rel_v, mirror), vk);

    // others [34:94] — teammates (asc id) then opponents (asc id)
    let mut others: Vec<&rocketsim_rs::CarInfo> = state
        .cars
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != car_idx)
        .map(|(_, c)| c)
        .collect();
    others.sort_by_key(|c| (c.team != me.team, c.id));
    for c in others.into_iter().take(MAX_OTHERS) {
        w.v3(mir(c.state.pos, mirror), pk);
        w.v3(mir(c.state.vel, mirror), vk);
        w.v3(mir(c.state.rot_mat.forward, mirror), 1.0);
        w.f(c.state.boost / 100.0);
        w.f((c.team == me.team) as u8 as f32);
        w.f((!c.state.is_demoed) as u8 as f32);
    }
    // remaining slots stay zero (out.fill above)
}
