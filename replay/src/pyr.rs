//! Analytic pitch/yaw/roll estimation from consecutive rigid-body frames.
//!
//! Replays only record a car's *analog* throttle/steer inputs, not the
//! aerial pitch/yaw/roll axes — those are reconstructed here from the
//! physical effect they have on angular velocity, by inverting RocketSim's
//! (ballistic) torque model:
//!
//!   omega_dot = T * input + D * omega        (per body-local axis)
//!
//! Solving for `input`:
//!
//!   input = (omega_dot - D * omega) / T
//!
//! Ported from rlgym-tools' `inverse_aerial_controls` / Rolv-Arild's
//! `replay-pretraining` `inverse_aerial_controls.py`.
//!
//! ## Axis convention (confirmed, not the plan's tentative guess)
//!
//! `rocketsim_rs::glam_ext` builds a car's world rotation matrix as
//! `Mat3::from_cols(forward, right, up)`, i.e. the body-local **x** axis is
//! forward, **y** is right, and **z** is up (see
//! `rocketsim_rs-0.37.0/src/glam_ext.rs` — `CarOrientation { forward, right,
//! up }` mapped to `x_axis`/`y_axis`/`z_axis`). That is the standard
//! aviation roll/pitch/yaw convention:
//!   - **roll**  spins about the forward axis  -> local **x**
//!   - **pitch** spins about the right axis    -> local **y**
//!   - **yaw**   spins about the up axis        -> local **z**
//!
//! This is also consistent with the torque-constant magnitudes below
//! (`|T_r| > |T_p| > |T_y|`): roll is the fastest analog rotation in Rocket
//! League, pitch is medium, yaw is slowest — matching roll=x, pitch=y,
//! yaw=z, not the plan sketch's pitch=x/yaw=z/roll=y ordering.

use crate::frames::RigidFrame;

/// Torque coefficients (rad/s^2 at full analog input), per body-local axis.
const T_R: f32 = -36.07956616966136; // roll  (about local x / forward)
const T_P: f32 = -12.146176938276769; // pitch (about local y / right)
const T_Y: f32 = 8.91962804287785; // yaw   (about local z / up)

/// Aerodynamic drag coefficients, per body-local axis. Roll has ~no drag.
const D_R: f32 = 0.0;
const D_P: f32 = -2.798194258050845;
const D_Y: f32 = -1.886491900437232;

/// Rotate a vector by a unit quaternion `[x, y, z, w]` (active rotation:
/// `v' = q * v * q^-1`), using the standard cross-product expansion (avoids
/// building a full quaternion-multiply).
fn rotate_vector(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];

    // t = 2 * cross(qv, v)
    let t = [
        2.0 * (qv[1] * v[2] - qv[2] * v[1]),
        2.0 * (qv[2] * v[0] - qv[0] * v[2]),
        2.0 * (qv[0] * v[1] - qv[1] * v[0]),
    ];
    // v' = v + qw * t + cross(qv, t)
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

/// Rotate a world-frame vector into the body-local frame described by unit
/// quaternion `q` (i.e. apply the inverse/conjugate rotation).
fn world_to_local(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let conj = [-q[0], -q[1], -q[2], q[3]];
    rotate_vector(conj, v)
}

/// Estimate the `[pitch, yaw, roll]` analog inputs (each clamped to
/// `[-1, 1]`) that would produce the observed change in angular velocity
/// between `prev` and `cur`, `dt` seconds apart.
///
/// Aerial torque inputs have no effect while the car is on the ground (the
/// ground has its own, unrelated turning model), so `on_ground` short-
/// circuits to `[0, 0, 0]`.
pub fn estimate_pyr(prev: &RigidFrame, cur: &RigidFrame, dt: f32, on_ground: bool) -> [f32; 3] {
    if on_ground {
        return [0.0, 0.0, 0.0];
    }

    let local_prev = world_to_local(prev.quat, prev.ang_vel);
    let local_cur = world_to_local(cur.quat, cur.ang_vel);

    let omega_dot = [
        (local_cur[0] - local_prev[0]) / dt,
        (local_cur[1] - local_prev[1]) / dt,
        (local_cur[2] - local_prev[2]) / dt,
    ];

    // Drag term uses the local angular velocity at `cur` (the end of the
    // interval the derivative was estimated over).
    let roll = (omega_dot[0] - D_R * local_cur[0]) / T_R;
    let pitch = (omega_dot[1] - D_P * local_cur[1]) / T_P;
    let yaw = (omega_dot[2] - D_Y * local_cur[2]) / T_Y;

    [pitch.clamp(-1.0, 1.0), yaw.clamp(-1.0, 1.0), roll.clamp(-1.0, 1.0)]
}
