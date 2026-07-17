//! TDD for Task B2 (`replay/src/project.rs`): context-gated projection of an
//! 8-dim replay control vector onto the nearest row of the 92-entry action
//! table (`construct_engine::actions::make_lookup_table_v1`).

use construct_engine::actions::make_lookup_table_v1;
use construct_replay::project::project_action;

/// Every one of the 92 table rows, given a compatible context (grounded for
/// ground rows, airborne+has_flip for jump/stall rows), must project back to
/// ITSELF at cost 0 — the projection's cost model (including its
/// dodge-vs-stall raw-channel tiebreak) is designed so self-match is always
/// the unique minimum; see `project.rs`'s module doc.
#[test]
fn identity_every_row_projects_to_itself() {
    let table = make_lookup_table_v1();
    for (i, row) in table.iter().enumerate() {
        let on_ground = i < 24; // rows 0..24 are the ground rows.
        let got = project_action(row, on_ground, true, 1.0, &table);
        assert_eq!(
            got, i,
            "row {i} {row:?} (on_ground={on_ground}) projected to {got} {:?} instead of itself",
            table[got]
        );
    }
}

#[test]
fn grounded_steer_rounds_to_nearest_table_value() {
    let table = make_lookup_table_v1();
    // throttle=1, steer=0.7 (nearest table steer is 1, not 0), boost pressed.
    let action = [1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let idx = project_action(&action, true, true, 1.0, &table);
    assert!(idx < 24, "expected a ground row, got index {idx}");
    assert_eq!(
        table[idx],
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "expected throttle=1 steer=1 boost=1 handbrake=0 ground row, got {:?}",
        table[idx]
    );
}

#[test]
fn boost_excluded_from_cost_when_boost_amount_is_zero() {
    let table = make_lookup_table_v1();
    let action_boost_pressed = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let action_boost_released = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let idx_pressed = project_action(&action_boost_pressed, true, true, 0.0, &table);
    let idx_released = project_action(&action_boost_released, true, true, 0.0, &table);
    assert_eq!(
        idx_pressed, idx_released,
        "boost_amount<=0 must make boost pressed/released project identically"
    );
    // Tiny tiebreak prefers row.boost==0 when boost is unscored.
    assert_eq!(table[idx_pressed], [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn aerial_no_jump_matches_pitch_and_nearest_yaw() {
    let table = make_lookup_table_v1();
    let action = [0.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0];
    let idx = project_action(&action, false, true, 1.0, &table);
    assert!(idx >= 24 && idx < 90, "expected an aerial row, got index {idx}");
    let row = table[idx];
    assert_eq!(row[2], -1.0, "pitch must match exactly");
    assert!(row[3] == 0.0 || row[3] == 1.0, "yaw 0.5 must round to the nearest table value (0 or 1)");
    assert_eq!(row[4], 0.0, "roll must match exactly (action roll is 0)");
    assert_eq!(row[5], 0.0, "jump must be 0 (not pressed in the action)");
}

#[test]
fn stall_input_maps_to_a_stall_row() {
    let table = make_lookup_table_v1();
    // yaw=1, roll=-1 (opposing diagonal input, dodgeDir cancels to zero) —
    // distinct from a *plain* jump (pitch=yaw=roll=0), which also has a
    // zero-magnitude dodgeDir but represents "no direction pressed" rather
    // than "opposing directions pressed", hence must resolve to a stall row
    // (90/91) not a pure-jump row.
    let action = [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0];
    let idx = project_action(&action, false, true, 1.0, &table);
    assert!(idx == 90 || idx == 91, "expected a stall row (90/91), got index {idx} = {:?}", table[idx]);
}

#[test]
fn dodge_direction_forward_flip_aligns_pitch_negative() {
    let table = make_lookup_table_v1();
    // pitch=-1 (forward), jump pressed, airborne+has_flip -> forward-dodge row.
    let action = [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let idx = project_action(&action, false, true, 1.0, &table);
    let row = table[idx];
    assert_eq!(
        row,
        [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "expected the forward-flip row (pitch=-1, boost=0), got {row:?}"
    );
}

#[test]
fn grounded_pure_jump_press_reaches_a_pure_jump_row() {
    let table = make_lookup_table_v1();
    // Grounded, jump pressed, no boost -- a normal jump-off. Ground rows
    // never have jump=1, so this must escape to one of the table's two
    // "pure jump" (pitch=yaw=roll=0) rows rather than a ground row.
    let action = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let idx = project_action(&action, true, true, 1.0, &table);
    let row = table[idx];
    assert_eq!(row[5], 1.0, "expected a jump=1 row");
    assert_eq!((row[2], row[3], row[4]), (0.0, 0.0, 0.0), "expected a pure-jump row (no dodge direction)");
    assert_eq!(row[6], 0.0, "boost not pressed in the action");
}

#[test]
fn airborne_jump_without_flip_available_is_treated_like_not_pressed() {
    let table = make_lookup_table_v1();
    // Jump held but has_flip=false (already used up in the air): the press
    // does nothing in-sim, so this must behave like jump not pressed at all
    // (excluding jump/stall rows), matching only pitch/yaw/roll.
    let action = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let idx = project_action(&action, false, false, 1.0, &table);
    let row = table[idx];
    assert_eq!(row[5], 0.0, "expected a jump=0 row when has_flip is false");
    assert_eq!(row[2], 1.0, "pitch must still match");
}
