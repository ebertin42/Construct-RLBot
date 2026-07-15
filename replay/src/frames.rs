//! Frame extraction: turn a parsed `.replay` into typed, fixed-fps
//! `ReplayFrames` (ball + per-car rigid body/boost/partial-input state).
//!
//! Uses subtr-actor 1.2.0's typed `ReplayDataCollector` (not `NDArrayCollector`)
//! wrapped in a `FrameRateDecorator` to resample at `target_fps`. The typed path
//! (`FrameData { ball_data, players: Vec<(PlayerId, PlayerData)>, .. }`) is
//! mapped field-by-field into the structs below rather than relying on
//! `NDArrayCollector`'s column-index registry.

use subtr_actor::{
    BallFrame, FrameRateDecorator, PlayerFrame, ReplayDataCollector, ReplayProcessor,
};

/// A single rigid-body physics state (ball or car), in raw replay units
/// (uu, uu/s, rad/s) — NOT normalized observation space.
#[derive(Debug, Clone)]
pub struct RigidFrame {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
    /// Quaternion in `[x, y, z, w]` order (matches `boxcars::Quaternion`'s
    /// field order).
    pub quat: [f32; 4],
}

impl RigidFrame {
    fn zero() -> Self {
        RigidFrame {
            pos: [0.0; 3],
            vel: [0.0; 3],
            ang_vel: [0.0; 3],
            quat: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn from_rigid_body(rb: &boxcars::RigidBody) -> Self {
        let lv = rb.linear_velocity.unwrap_or(boxcars::Vector3f { x: 0.0, y: 0.0, z: 0.0 });
        let av = rb.angular_velocity.unwrap_or(boxcars::Vector3f { x: 0.0, y: 0.0, z: 0.0 });
        RigidFrame {
            pos: [rb.location.x, rb.location.y, rb.location.z],
            vel: [lv.x, lv.y, lv.z],
            ang_vel: [av.x, av.y, av.z],
            quat: [rb.rotation.x, rb.rotation.y, rb.rotation.z, rb.rotation.w],
        }
    }
}

/// A single player's per-frame state: rigid body + boost + partial replay
/// inputs + action flags.
#[derive(Debug, Clone)]
pub struct CarFrame {
    pub rb: RigidFrame,
    /// Boost amount normalized to `0..1` (subtr-actor reports `0..255`).
    pub boost: f32,
    pub team: u8,
    pub throttle: f32,
    pub steer: f32,
    pub handbrake: bool,
    pub jump_active: bool,
    pub dodge_active: bool,
    pub on_ground: bool,
    pub demoed: bool,
}

impl CarFrame {
    /// A zero-valued car frame for a player who has no data yet (not spawned,
    /// or demoed with no prior observation to carry forward).
    fn absent(team: u8) -> Self {
        CarFrame {
            rb: RigidFrame::zero(),
            boost: 0.0,
            team,
            throttle: 0.0,
            steer: 0.0,
            handbrake: false,
            jump_active: false,
            dodge_active: false,
            on_ground: false,
            demoed: true,
        }
    }

    /// Same physical/control state as `self` but flagged as carried-forward
    /// (player absent this frame; last-known state retained).
    fn carried_forward(&self) -> Self {
        CarFrame {
            demoed: true,
            ..self.clone()
        }
    }
}

/// Fixed-fps, per-frame replay data: ball + stable-order per-player cars.
#[derive(Debug, Clone)]
pub struct ReplayFrames {
    pub fps: u32,
    pub ball: Vec<RigidFrame>,
    /// `cars[t]` is the per-player vector at frame `t`; every entry has the
    /// same length as `player_teams` (stable first-seen player order).
    pub cars: Vec<Vec<CarFrame>>,
    /// Stable per-player-index team id, in first-seen order.
    pub player_teams: Vec<u8>,
}

/// Maps a raw replay-input byte (throttle/steer, ~128 neutral/centered,
/// 0..255) to a signed `-1..1` range. Byte 0 maps to -128/127 (~-1.008)
/// before clamping -- clamped here so callers never see a value outside
/// `[-1.0, 1.0]`.
fn signed_input_from_byte(byte: u8) -> f32 {
    ((byte as f32 - 128.0) / 127.0).clamp(-1.0, 1.0)
}

/// Raw boost byte (0..255) to `0..1`.
fn normalize_boost(raw: f32) -> f32 {
    (raw / 255.0).clamp(0.0, 1.0)
}

/// A player is grounded when their up-vector's implied on-ground state can't
/// be read directly from subtr-actor's `PlayerFrame` (it doesn't expose one);
/// approximate via jump/dodge/double-jump activity: a player mid-jump or
/// mid-dodge is airborne. This is a coarse signal — Task 3/4 refine actions
/// via analytic pitch/yaw/roll and RocketSim tick-stepping, which don't
/// depend on this flag being exact.
fn approximate_on_ground(jump_active: bool, double_jump_active: bool, dodge_active: bool) -> bool {
    !(jump_active || double_jump_active || dodge_active)
}

/// Extracts typed, fixed-fps replay frames from raw `.replay` bytes.
///
/// Uses subtr-actor's typed `ReplayDataCollector` (ball_data/players) wrapped
/// in `FrameRateDecorator::new_from_fps(target_fps, ..)` to resample the
/// replay's native ~30 Hz network frames down to (or up to) `target_fps`
/// evenly spaced samples. Players are indexed in stable first-seen order;
/// a player absent in a given frame (not yet spawned, or demoed with no
/// prior data) carries forward their last-known `CarFrame` with `demoed`
/// set to `true`.
pub fn extract_frames(bytes: &[u8], target_fps: u32) -> Result<ReplayFrames, String> {
    if target_fps == 0 {
        return Err("target_fps must be > 0".to_string());
    }

    let replay = boxcars::ParserBuilder::new(bytes)
        .must_parse_network_data()
        .parse()
        .map_err(|e| format!("parse: {e}"))?;

    let mut collector = ReplayDataCollector::new();
    let mut processor = ReplayProcessor::new(&replay).map_err(|e| e.variant.to_string())?;
    {
        let mut decorated = FrameRateDecorator::new_from_fps(target_fps as f32, &mut collector);
        processor
            .process(&mut decorated)
            .map_err(|e| e.variant.to_string())?;
    }
    let frame_data = collector.get_frame_data();

    let num_frames = frame_data.ball_data.frame_count();

    let ball: Vec<RigidFrame> = frame_data
        .ball_data
        .frames()
        .iter()
        .map(|bf| match bf {
            BallFrame::Data { rigid_body } => RigidFrame::from_rigid_body(rigid_body),
            BallFrame::Empty => RigidFrame::zero(),
        })
        .collect();

    let num_players = frame_data.players.len();
    // Stable per-player team, updated as real per-frame team info becomes
    // available (first-seen order matches frame_data.players' order, which
    // subtr-actor already maintains as players are first encountered).
    let mut player_teams: Vec<u8> = vec![0; num_players];

    // Per-player last-known CarFrame, used to carry state forward across
    // frames where the player is absent (PlayerFrame::Empty).
    let mut last_known: Vec<Option<CarFrame>> = vec![None; num_players];

    let mut cars: Vec<Vec<CarFrame>> = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let mut frame_cars: Vec<CarFrame> = Vec::with_capacity(num_players);
        for (p_idx, (_player_id, player_data)) in frame_data.players.iter().enumerate() {
            let player_frame = player_data.frames().get(t);
            match player_frame {
                Some(PlayerFrame::Data {
                    rigid_body,
                    boost_amount,
                    boost_active: _,
                    powerslide_active,
                    jump_active,
                    double_jump_active,
                    dodge_active,
                    player_name: _,
                    team,
                    is_team_0,
                    camera: _,
                    input,
                }) => {
                    let team_u8 = if let Some(is_zero) = is_team_0 {
                        if *is_zero { 0 } else { 1 }
                    } else if let Some(t) = team {
                        (*t).max(0) as u8
                    } else {
                        player_teams[p_idx]
                    };
                    player_teams[p_idx] = team_u8;

                    let on_ground =
                        approximate_on_ground(*jump_active, *double_jump_active, *dodge_active);

                    let car_frame = CarFrame {
                        rb: RigidFrame::from_rigid_body(rigid_body),
                        boost: normalize_boost(*boost_amount),
                        team: team_u8,
                        throttle: input
                            .throttle
                            .map(signed_input_from_byte)
                            .unwrap_or(0.0),
                        steer: input.steer.map(signed_input_from_byte).unwrap_or(0.0),
                        handbrake: *powerslide_active,
                        jump_active: *jump_active,
                        dodge_active: *dodge_active,
                        on_ground,
                        demoed: false,
                    };
                    last_known[p_idx] = Some(car_frame.clone());
                    frame_cars.push(car_frame);
                }
                Some(PlayerFrame::Empty) | None => {
                    let carried = match &last_known[p_idx] {
                        Some(prev) => prev.carried_forward(),
                        None => CarFrame::absent(player_teams[p_idx]),
                    };
                    frame_cars.push(carried);
                }
            }
        }
        cars.push(frame_cars);
    }

    Ok(ReplayFrames {
        fps: target_fps,
        ball,
        cars,
        player_teams,
    })
}
