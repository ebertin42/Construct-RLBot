//! 120 Hz reconstruction: steps RocketSim between consecutive replay frames
//! (typically 30 Hz) to fill in the physics states the replay itself never
//! records, porting VirxEC/replay-to-rocketsim's tick-stepping approach.
//!
//! Verified (this task, against the `sample.replay` fixture): `frames::extract_frames`
//! already reports positions in RocketSim's native uu (field |x| up to ~4096,
//! |y| up to ~5980 including out-of-bounds/celebration frames, z in
//! `[0, ~1861]`), and quaternions are already unit-norm. **No unit scale
//! correction is applied** — replay and RocketSim units agree directly.
//!
//! ## Algorithm
//! For each consecutive frame pair `(t, t+1)`:
//! 1. Snap the ball and every car to frame `t`'s authoritative rigid body (+
//!    boost, + on_ground).
//! 2. Set `CarControls` for every car: throttle/steer/handbrake/jump straight
//!    from the `CarFrame`, and pitch/yaw/roll from
//!    [`crate::pyr::estimate_pyr`] over `[frame t, frame t+1]`. These controls
//!    are held constant for the whole sub-step interval (the replay only
//!    samples inputs at `fps` Hz to begin with).
//! 3. `step(1)` in a loop of `N = round(120 * dt)` (dt = `1/fps`, ~4 for 30
//!    fps replays), capturing one [`Tick`] after every single tick.
//! 4. Loop back to step 1 with `t+1`: this **snaps** the arena back to the
//!    next frame's authoritative state rather than trusting whatever
//!    RocketSim's simulation produced, so integration error never
//!    accumulates across the whole replay — each 30 Hz interval's 120 Hz
//!    ticks are an independent, physically-plausible interpolation.
//!
//! Any tick whose ball/car state falls outside `engine::episode::state_is_sane`'s
//! bounds (pos/vel/ang_vel <= 12000/20000/100) is dropped rather than emitted.

use rocketsim_rs::{
    cxx,
    math::{RotMat, Vec3},
    sim::{Arena, CarConfig, CarControls, Team},
};

use crate::{
    frames::{CarFrame, ReplayFrames, RigidFrame},
    pyr::estimate_pyr,
};

/// One 120 Hz physics tick: ball + every player's car, reconstructed by
/// stepping RocketSim between the replay's native frames.
pub struct Tick {
    pub ball: RigidFrame,
    pub cars: Vec<CarFrame>,
    /// Per-car applied control action for this tick, same order as `cars`.
    /// Layout: `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`
    /// (jump/boost/handbrake as 0.0/1.0). Held constant across every 120 Hz
    /// tick in the 30 Hz interval this tick belongs to — the replay only
    /// samples inputs at `frames.fps` Hz to begin with, so this is the same
    /// `CarControls` value `set_all_controls` was given for the interval,
    /// not a re-derived value. Needed downstream by IDM/BC, which train on
    /// (state, action) pairs rather than physics state alone.
    pub actions: Vec<[f32; 8]>,
}

pub struct Reconstructed {
    pub ticks: Vec<Tick>,
    pub player_teams: Vec<u8>,
    /// Source replay frame rate the reconstruction was built from (i.e. the
    /// `target_fps` originally passed to `extract_frames`) — carried through
    /// for shard sidecar metadata.
    pub fps: u32,
}

/// Sanity bounds mirroring `engine::episode::state_is_sane` — see
/// `engine/src/episode.rs`'s `state_is_sane`/`vec3_within` (pos/vel/ang_vel
/// magnitude caps ~2-4x the hardest legal in-game values). A contained
/// contact-solver blowup can ramp through huge-but-finite values before
/// going NaN; any tick this coarse dies here instead of reaching a shard.
const POS_LIMIT: f32 = 12_000.0;
const VEL_LIMIT: f32 = 20_000.0;
const ANG_VEL_LIMIT: f32 = 100.0;

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn within3(v: [f32; 3], limit: f32) -> bool {
    v.iter().all(|x| x.abs() <= limit)
}

fn rigid_frame_is_sane(rf: &RigidFrame) -> bool {
    finite3(rf.pos)
        && finite3(rf.vel)
        && finite3(rf.ang_vel)
        && rf.quat.iter().all(|x| x.is_finite())
        && within3(rf.pos, POS_LIMIT)
        && within3(rf.vel, VEL_LIMIT)
        && within3(rf.ang_vel, ANG_VEL_LIMIT)
}

fn tick_is_sane(tick: &Tick) -> bool {
    rigid_frame_is_sane(&tick.ball) && tick.cars.iter().all(|c| rigid_frame_is_sane(&c.rb))
}

// --- quat <-> RotMat -------------------------------------------------------
//
// Duplicated (not imported) from `pyr.rs`'s confirmed axis convention —
// `RotMat { forward, right, up }` are the images of local x/y/z under the
// quaternion's active rotation (`rocketsim_rs::glam_ext` builds
// `Mat3::from_cols(forward, right, up)`, i.e. col0=forward=R*x, col1=right=
// R*y, col2=up=R*z). `pyr.rs`'s `rotate_vector` helper is private to that
// module, so it's re-derived here rather than exposed cross-module for a
// two-line function.

/// Rotate `v` by unit quaternion `q = [x, y, z, w]` (active rotation,
/// `v' = q * v * q^-1`), via the standard cross-product expansion.
fn rotate_vector(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];
    let t = [
        2.0 * (qv[1] * v[2] - qv[2] * v[1]),
        2.0 * (qv[2] * v[0] - qv[0] * v[2]),
        2.0 * (qv[0] * v[1] - qv[1] * v[0]),
    ];
    let cross_qv_t = [
        qv[1] * t[2] - qv[2] * t[1],
        qv[2] * t[0] - qv[0] * t[2],
        qv[0] * t[1] - qv[1] * t[0],
    ];
    [
        v[0] + qw * t[0] + cross_qv_t[0],
        v[1] + qw * t[1] + cross_qv_t[1],
        v[2] + qw * t[2] + cross_qv_t[2],
    ]
}

/// Build a RocketSim `RotMat` (forward/right/up, in world space) from a
/// `[x, y, z, w]` quaternion.
fn quat_to_rotmat(q: [f32; 4]) -> RotMat {
    let f = rotate_vector(q, [1.0, 0.0, 0.0]);
    let r = rotate_vector(q, [0.0, 1.0, 0.0]);
    let u = rotate_vector(q, [0.0, 0.0, 1.0]);
    RotMat {
        forward: Vec3::new(f[0], f[1], f[2]),
        right: Vec3::new(r[0], r[1], r[2]),
        up: Vec3::new(u[0], u[1], u[2]),
    }
}

/// Inverse of `quat_to_rotmat`: recover a `[x, y, z, w]` quaternion from a
/// RocketSim `RotMat`, via the standard trace/largest-diagonal matrix ->
/// quaternion construction (Shepperd's method). `RotMat.{forward,right,up}`
/// are read as the matrix's columns 0/1/2 respectively, matching
/// `quat_to_rotmat`'s construction.
fn rotmat_to_quat(m: &RotMat) -> [f32; 4] {
    let (m00, m10, m20) = (m.forward.x, m.forward.y, m.forward.z);
    let (m01, m11, m21) = (m.right.x, m.right.y, m.right.z);
    let (m02, m12, m22) = (m.up.x, m.up.y, m.up.z);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).max(0.0).sqrt() * 2.0; // s = 4*qw
        [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).max(0.0).sqrt() * 2.0; // s = 4*qx
        [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).max(0.0).sqrt() * 2.0; // s = 4*qy
        [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).max(0.0).sqrt() * 2.0; // s = 4*qz
        [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    }
}

fn vec3_to_arr(v: &Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Overwrite the arena's ball state with `rf` (frame `t`'s authoritative
/// rigid body). Mirrors `engine/src/curriculum.rs`'s get -> mutate -> set
/// idiom, so untouched `BallState` fields (heatseeker/dropshot info, tick
/// bookkeeping) are preserved rather than clobbered.
fn set_ball_state(arena: &mut cxx::UniquePtr<Arena>, rf: &RigidFrame) {
    let mut b = arena.pin_mut().get_ball();
    b.pos = Vec3::new(rf.pos[0], rf.pos[1], rf.pos[2]);
    b.vel = Vec3::new(rf.vel[0], rf.vel[1], rf.vel[2]);
    b.ang_vel = Vec3::new(rf.ang_vel[0], rf.ang_vel[1], rf.ang_vel[2]);
    b.rot_mat = quat_to_rotmat(rf.quat);
    arena.pin_mut().set_ball(b);
}

/// Overwrite car `car_id`'s state with `cf` (frame `t`'s authoritative car
/// frame). `CarState.boost` is `0..100`; `CarFrame.boost` is `0..1`.
fn set_car_state(arena: &mut cxx::UniquePtr<Arena>, car_id: u32, cf: &CarFrame) -> Result<(), String> {
    let mut cs = arena.pin_mut().get_car(car_id);
    cs.pos = Vec3::new(cf.rb.pos[0], cf.rb.pos[1], cf.rb.pos[2]);
    cs.vel = Vec3::new(cf.rb.vel[0], cf.rb.vel[1], cf.rb.vel[2]);
    cs.ang_vel = Vec3::new(cf.rb.ang_vel[0], cf.rb.ang_vel[1], cf.rb.ang_vel[2]);
    cs.rot_mat = quat_to_rotmat(cf.rb.quat);
    cs.boost = (cf.boost * 100.0).clamp(0.0, 100.0);
    cs.is_on_ground = cf.on_ground;
    arena.pin_mut().set_car(car_id, cs).map_err(|e: rocketsim_rs::NoCarFound| e.to_string())
}

/// Reconstructs a dense 120 Hz tick stream from `frames` (Task 2's typed,
/// fixed-fps replay data) by stepping a fresh RocketSim `Arena` between each
/// consecutive pair of frames and snapping back to the replay's authoritative
/// state at every frame boundary.
pub fn reconstruct_120hz(frames: &ReplayFrames) -> Result<Reconstructed, String> {
    let num_frames = frames.ball.len();
    let num_players = frames.player_teams.len();

    if frames.cars.len() != num_frames {
        return Err(format!(
            "ball/cars frame count mismatch: {num_frames} ball frames vs {} car-frame rows",
            frames.cars.len()
        ));
    }
    for (t, row) in frames.cars.iter().enumerate() {
        if row.len() != num_players {
            return Err(format!(
                "frame {t}: expected {num_players} cars (player_teams length), got {}",
                row.len()
            ));
        }
    }

    if num_frames < 2 {
        return Ok(Reconstructed {
            ticks: Vec::new(),
            player_teams: frames.player_teams.clone(),
            fps: frames.fps,
        });
    }

    if frames.fps == 0 {
        return Err("frames.fps must be > 0".to_string());
    }

    crate::sim_init::ensure_init(None);
    let mut arena = Arena::default_standard();
    let car_ids: Vec<u32> = frames
        .player_teams
        .iter()
        .map(|&team| {
            let t = if team == 0 { Team::Blue } else { Team::Orange };
            arena.pin_mut().add_car(t, CarConfig::octane())
        })
        .collect();

    let dt = 1.0 / frames.fps as f32;
    let n_substeps = (120.0 * dt).round().max(1.0) as u32;

    let mut ticks: Vec<Tick> = Vec::with_capacity((num_frames - 1) * n_substeps as usize);

    for t in 0..(num_frames - 1) {
        // 1. Snap ball + all cars to frame t's authoritative state.
        set_ball_state(&mut arena, &frames.ball[t]);
        for (p, &car_id) in car_ids.iter().enumerate() {
            set_car_state(&mut arena, car_id, &frames.cars[t][p])?;
        }

        // 2. Controls for this interval: throttle/steer/handbrake/jump come
        // straight from frame t's CarFrame; pitch/yaw/roll are the analytic
        // inversion over [frame t, frame t+1]. `on_ground` gates the pyr
        // estimate — the sim car was just set to frame t's `on_ground`
        // above, so that's the "sim state" value used here (there is no
        // richer signal available pre-step). Boost-held is approximated by
        // a boost-amount drop between frame t and frame t+1 (CarFrame does
        // not carry an explicit boost-active flag).
        let mut controls: Vec<CarControls> = Vec::with_capacity(num_players);
        for (p, _) in car_ids.iter().enumerate() {
            let cf = &frames.cars[t][p];
            let next_rb = &frames.cars[t + 1][p].rb;
            let pyr = estimate_pyr(&cf.rb, next_rb, dt, cf.on_ground);
            let boosting = frames.cars[t + 1][p].boost < cf.boost - 1e-4;
            controls.push(CarControls {
                throttle: cf.throttle,
                steer: cf.steer,
                pitch: pyr[0],
                yaw: pyr[1],
                roll: pyr[2],
                jump: cf.jump_active,
                boost: boosting,
                handbrake: cf.handbrake,
            });
        }
        let pairs: Vec<(u32, CarControls)> =
            car_ids.iter().zip(controls.iter()).map(|(&id, &c)| (id, c)).collect();
        arena.pin_mut().set_all_controls(&pairs).map_err(|e| e.to_string())?;

        // Per-car 8-dim action vector for this interval (constant across all
        // of its 120 Hz ticks) — see `Tick::actions` doc comment for layout.
        let actions: Vec<[f32; 8]> = controls
            .iter()
            .map(|c| {
                [
                    c.throttle,
                    c.steer,
                    c.pitch,
                    c.yaw,
                    c.roll,
                    if c.jump { 1.0 } else { 0.0 },
                    if c.boost { 1.0 } else { 0.0 },
                    if c.handbrake { 1.0 } else { 0.0 },
                ]
            })
            .collect();

        // 3. Step one 120 Hz tick at a time, capturing state after each.
        for _ in 0..n_substeps {
            arena.pin_mut().step(1);
            let gs = arena.pin_mut().get_game_state();

            let ball = RigidFrame {
                pos: vec3_to_arr(&gs.ball.pos),
                vel: vec3_to_arr(&gs.ball.vel),
                ang_vel: vec3_to_arr(&gs.ball.ang_vel),
                quat: rotmat_to_quat(&gs.ball.rot_mat),
            };

            let mut cars_out: Vec<CarFrame> = Vec::with_capacity(num_players);
            for (p, &car_id) in car_ids.iter().enumerate() {
                let ci = gs
                    .cars
                    .iter()
                    .find(|c| c.id == car_id)
                    .ok_or_else(|| format!("car id {car_id} missing from post-step game state"))?;
                let cs = &ci.state;
                let applied = controls[p];
                cars_out.push(CarFrame {
                    rb: RigidFrame {
                        pos: vec3_to_arr(&cs.pos),
                        vel: vec3_to_arr(&cs.vel),
                        ang_vel: vec3_to_arr(&cs.ang_vel),
                        quat: rotmat_to_quat(&cs.rot_mat),
                    },
                    boost: (cs.boost / 100.0).clamp(0.0, 1.0),
                    team: frames.player_teams[p],
                    throttle: applied.throttle,
                    steer: applied.steer,
                    handbrake: applied.handbrake,
                    jump_active: cs.is_jumping,
                    dodge_active: cs.is_flipping,
                    on_ground: cs.is_on_ground,
                    demoed: cs.is_demoed,
                });
            }

            let tick = Tick { ball, cars: cars_out, actions: actions.clone() };
            if tick_is_sane(&tick) {
                ticks.push(tick);
            }
        }
    }

    Ok(Reconstructed { ticks, player_teams: frames.player_teams.clone(), fps: frames.fps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq4(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps || (x + y).abs() < eps)
    }

    #[test]
    fn quat_rotmat_round_trip() {
        let cases = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.7071068, 0.0, 0.7071068],
            [0.2705981, 0.2705981, 0.6532815, 0.6532815],
        ];
        for q in cases {
            let rm = quat_to_rotmat(q);
            let back = rotmat_to_quat(&rm);
            assert!(approx_eq4(q, back, 1e-3), "round trip failed: {q:?} -> {back:?}");
        }
    }

    #[test]
    fn identity_quat_gives_identity_rotmat() {
        let rm = quat_to_rotmat([0.0, 0.0, 0.0, 1.0]);
        assert!((rm.forward.x - 1.0).abs() < 1e-6);
        assert!((rm.right.y - 1.0).abs() < 1e-6);
        assert!((rm.up.z - 1.0).abs() < 1e-6);
    }
}
