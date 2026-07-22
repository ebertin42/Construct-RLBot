//! Test-only helpers to construct a `GameState` from plain fixture data, so golden
//! tests can feed known car/ball/pad values. Dependency-free (no serde) -- the
//! integration test parses JSON and passes primitives here. Compiled into the lib
//! unconditionally (it only depends on rocketsim_rs, already a dep) so integration
//! tests -- which link the non-test lib -- can reach it.
use crate::sim_init::ensure_init;
use rocketsim_rs::math::{RotMat, Vec3};
use rocketsim_rs::sim::{Arena, CarConfig, CarState, Team};
use rocketsim_rs::GameState;

pub struct FixtureCar {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub boost: f32,
    pub on_ground: bool,
    pub has_flip: bool,
    pub is_demoed: bool,
    pub team_orange: bool,
}

#[inline]
fn v(a: [f32; 3]) -> Vec3 { Vec3::new(a[0], a[1], a[2]) }

/// Build a `GameState`: a real standard arena (for valid 34 boost pads), then
/// overwrite the cars (with clean `CarState::default()` bases so
/// `has_flip_or_jump()` follows the verified mapping
/// `on_ground || (!has_flipped && !has_double_jumped)`), the ball, and pad
/// activity. Pad activity is assigned by nearest position so it is robust to the
/// arena's pad ordering.
pub fn build_state(
    cars: &[FixtureCar],
    ball_pos: [f32; 3], ball_vel: [f32; 3], ball_ang: [f32; 3],
    pad_positions: &[[f32; 3]], pad_active: &[bool],
) -> GameState {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    for c in cars {
        let team = if c.team_orange { Team::Orange } else { Team::Blue };
        let _ = arena.pin_mut().add_car(team, CarConfig::octane());
    }
    let mut gs = arena.pin_mut().get_game_state();
    // get_game_state() does NOT preserve add order (cars come back id-descending),
    // so assign each fixture car's state to the gs car of the matching team, then
    // sort by team so gs.cars[i] lines up with fixture car i (cars[0]=blue,
    // cars[1]=orange -> sorted ascending [Blue, Orange]).
    for c in cars {
        let j = gs
            .cars
            .iter()
            .position(|gc| (gc.team == Team::Orange) == c.team_orange)
            .expect("no gs car of matching team");
        let mut s = CarState::default();
        s.pos = v(c.pos);
        s.vel = v(c.vel);
        s.ang_vel = v(c.ang_vel);
        let fwd = v(c.forward);
        let up = v(c.up);
        let right = Vec3::new(
            up.y * fwd.z - up.z * fwd.y,
            up.z * fwd.x - up.x * fwd.z,
            up.x * fwd.y - up.y * fwd.x,
        );
        s.rot_mat = RotMat { forward: fwd, right, up };
        s.boost = c.boost * 100.0;
        s.is_on_ground = c.on_ground;
        s.has_double_jumped = false;
        s.has_flipped = if c.on_ground { false } else { !c.has_flip };
        s.is_demoed = c.is_demoed;
        gs.cars[j].state = s;
    }
    gs.cars.sort_by_key(|c| c.team as u8);
    gs.ball.pos = v(ball_pos);
    gs.ball.vel = v(ball_vel);
    gs.ball.ang_vel = v(ball_ang);
    for (i, pos) in pad_positions.iter().enumerate() {
        let mut best = 0usize;
        let mut bd = f32::MAX;
        for (k, p) in gs.pads.iter().enumerate() {
            let pp = p.config.position;
            let d = (pp.x - pos[0]).powi(2) + (pp.y - pos[1]).powi(2);
            if d < bd {
                bd = d;
                best = k;
            }
        }
        gs.pads[best].state.is_active = pad_active[i];
    }
    gs
}
