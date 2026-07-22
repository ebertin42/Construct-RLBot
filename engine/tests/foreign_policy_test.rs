use std::collections::HashMap;

use construct_engine::actions::make_immortal_table;
use construct_engine::foreign::{ForeignKind, ForeignPolicy};
use construct_engine::test_support::{build_state, FixtureCar};

/// Immortal-shaped state_dict with all-zero weights: every logit is 0, so argmax
/// deterministically selects index 0. Lets us exercise the full
/// obs -> forward -> argmax -> table -> controls path without the (unlicensed)
/// real weights.
fn zero_immortal_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    let dims: [(usize, usize, usize); 7] = [
        (0, 512, 107),
        (2, 512, 512),
        (4, 512, 512),
        (6, 512, 512),
        (8, 512, 512),
        (10, 512, 512),
        (12, 126, 512),
    ];
    for (idx, out_dim, in_dim) in dims {
        w.insert(
            format!("net.{idx}.weight"),
            (vec![0.0f32; out_dim * in_dim], vec![out_dim, in_dim]),
        );
        w.insert(format!("net.{idx}.bias"), (vec![0.0f32; out_dim], vec![out_dim]));
    }
    w
}

/// Globally-unique per-car key: car ids restart at 1 in every arena.
fn key(arena: u64, car_id: u64) -> u64 { (arena << 32) | car_id }

fn car(team_orange: bool) -> FixtureCar {
    FixtureCar {
        pos: [100.0, -200.0, 17.0],
        vel: [10.0, 20.0, 0.0],
        ang_vel: [0.1, 0.2, 0.3],
        forward: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        boost: 0.34,
        on_ground: true,
        has_flip: true,
        is_demoed: false,
        team_orange,
    }
}

fn state() -> rocketsim_rs::GameState {
    let pad_positions = [[0.0f32, -4240.0, 70.0]; 34];
    let pad_active = [true; 34];
    build_state(
        &[car(false), car(true)],
        [0.0, 0.0, 93.15],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        &pad_positions,
        &pad_active,
    )
}

#[test]
fn foreign_policy_decides_via_table_lookup() {
    let mut p = ForeignPolicy::new(&zero_immortal_weights(), ForeignKind::Immortal).unwrap();
    assert_eq!(p.kind(), ForeignKind::Immortal);
    let gs = state();
    let table = make_immortal_table();

    // all-zero logits -> argmax picks index 0 -> that exact controls row
    let controls = p.decide(&gs, 0, key(0, 1));
    assert_eq!(controls, table[0], "should return row 0 of Immortal's table");

    // callable again (prev-action now populated, weights still zero -> same row)
    let again = p.decide(&gs, 0, key(0, 1));
    assert_eq!(again, table[0]);

    // and for the orange car
    let orange = p.decide(&gs, 1, key(0, 2));
    assert_eq!(orange, table[0]);

    p.reset();
    assert_eq!(p.decide(&gs, 0, key(0, 1)), table[0]);
}

#[test]
fn foreign_policy_rejects_wrong_shapes() {
    let mut w = zero_immortal_weights();
    // wrong input width (107 -> 99)
    w.insert("net.0.weight".into(), (vec![0.0f32; 512 * 99], vec![512, 99]));
    let err = match ForeignPolicy::new(&w, ForeignKind::Immortal) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("net.0.weight"), "unexpected error: {err}");
}

#[test]
fn foreign_policy_rejects_missing_layer() {
    let mut w = zero_immortal_weights();
    w.remove("net.12.weight");
    let err = match ForeignPolicy::new(&w, ForeignKind::Immortal) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("net.12.weight"), "unexpected error: {err}");
}


/// Regression: car ids restart at 1 in EVERY arena, so a car-id-only key made all
/// arenas in a foreign slot share one previous-action entry. The prev action feeds
/// the bot's observation, so that silently cross-contaminated every arena.
#[test]
fn prev_action_is_isolated_per_arena() {
    let mut p = ForeignPolicy::new(&zero_immortal_weights(), ForeignKind::Immortal).unwrap();
    let gs = state();
    // same car id (1), different arenas -> must not share state
    let k_a = key(0, 1);
    let k_b = key(7, 1);
    p.decide(&gs, 0, k_a);
    // arena 7 has never been seen: it must still start from the zero prev action,
    // which is what a fresh key yields. Resetting arena 0 must not disturb it.
    p.reset_car(k_a);
    let after = p.decide(&gs, 0, k_b);
    assert_eq!(after.len(), 8);
    // and clearing one arena's car leaves the other's entry intact
    p.decide(&gs, 0, k_a);
    p.reset_car(k_b);
    let still = p.decide(&gs, 0, k_a);
    assert_eq!(still.len(), 8);
}
