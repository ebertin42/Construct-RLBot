use construct_replay::frames::RigidFrame;
use construct_replay::pyr::estimate_pyr;

#[test]
fn zero_angvel_change_gives_zero_torque_inputs() {
    let r = RigidFrame { pos: [0.0; 3], vel: [0.0; 3], ang_vel: [0.0; 3], quat: [0.0, 0.0, 0.0, 1.0] };
    let out = estimate_pyr(&r, &r, 1.0 / 120.0, false);
    assert!(out.iter().all(|x| x.abs() < 1e-3), "no rotation -> ~zero inputs, got {out:?}");
    assert!(out.iter().all(|x| (-1.0..=1.0).contains(x)));
}

/// A pure-roll angular-acceleration case (nonzero angular velocity only
/// about local x, with an identity orientation so world == local frame)
/// must produce a dominant `roll` output with `pitch`/`yaw` ~= 0. This
/// pins down the axis mapping (roll <-> local x) rather than just the
/// all-zero trivial case above.
#[test]
fn pure_roll_angaccel_gives_dominant_roll_output() {
    let prev = RigidFrame {
        pos: [0.0; 3],
        vel: [0.0; 3],
        ang_vel: [0.0, 0.0, 0.0],
        quat: [0.0, 0.0, 0.0, 1.0],
    };
    let cur = RigidFrame {
        pos: [0.0; 3],
        vel: [0.0; 3],
        ang_vel: [0.1, 0.0, 0.0], // angular velocity picked up about local x (forward/roll axis)
        quat: [0.0, 0.0, 0.0, 1.0],
    };
    let [pitch, yaw, roll] = estimate_pyr(&prev, &cur, 1.0 / 120.0, false);

    assert!(pitch.abs() < 1e-3, "expected ~zero pitch, got {pitch}");
    assert!(yaw.abs() < 1e-3, "expected ~zero yaw, got {yaw}");
    assert!(roll.abs() > 0.1, "expected a clearly nonzero roll, got {roll}");
    assert!(
        roll.abs() > pitch.abs() && roll.abs() > yaw.abs(),
        "roll should dominate for a pure roll angular-acceleration: pitch={pitch} yaw={yaw} roll={roll}"
    );
    assert!([pitch, yaw, roll].iter().all(|x| (-1.0..=1.0).contains(x)));
}
