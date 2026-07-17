//! Context-gated projection of an 8-dim replay control vector onto the
//! nearest row of a discrete action table (Task B2 of the BC-pretrain plan,
//! `docs/superpowers/plans/2026-07-17-bc-pretrain.md`).
//!
//! Ports the LOGIC of rlgym-tools' `pick_action` (grounded actions are
//! scored on throttle/steer/handbrake; airborne actions are scored on
//! pitch/yaw/roll plus dodge-direction alignment for flip inputs) to our
//! 92-row `construct_engine::actions::make_lookup_table_v1()` table, rather
//! than reusing rlgym-tools' code directly (different table layout/size).
//!
//! ## Table shape recap (see `construct_engine::actions`)
//! - Rows 0..24 ("ground rows"): `[throttle, steer, 0, steer, 0, 0, boost,
//!   handbrake]` — pitch=roll=jump=0 always. This `pitch==0 && roll==0 &&
//!   jump==0` triple is unique to these 24 rows (the aerial-row loop
//!   explicitly skips that combination to avoid duplicating them), so it's
//!   a reliable ground/aerial row-shape discriminator that needs no
//!   knowledge of row index (`row_is_ground_shaped` below).
//! - Rows 24..90 ("aerial rows"): `[boost, yaw, pitch, yaw, roll, jump,
//!   boost, handbrake]` — the throttle/steer slots both carry `boost`/`yaw`
//!   duplicates, not real throttle. Two of these (`pitch=yaw=roll=0,
//!   jump=1`) are "pure jump" rows: a grounded jump-off (or an in-air
//!   second jump with no dodge direction), as opposed to a flip.
//! - Rows 90..92 ("stall rows"): `jump=1`, `yaw=-roll` so `dodgeDir =
//!   (-pitch, yaw+roll) == (0,0)` — the SAME zero magnitude as the
//!   pure-jump rows above, despite representing a different input (opposing
//!   diagonal stick vs. no stick input at all). See `RAW_TIEBREAK_W`.
//!
//! ## Cost model
//! For each table row, compute a mismatch cost over only the channels that
//! matter in the caller's context (grounded vs. airborne — see
//! `project_action`'s doc), then take the argmin. Ties resolve to the
//! lowest row index (rows are scanned ascending, `<` keeps the first-seen
//! winner) — in practice this never actually matters for a genuine
//! self-match (see `identity_every_row_projects_to_itself` in
//! `replay/tests/project_test.rs`): the raw-channel tiebreak below makes
//! self always the unique minimum.

/// Binary-channel mismatch weight — dominates continuous (`|a-b|`) costs so
/// a table row's on/off channels (handbrake, boost, jump-gated) are matched
/// before finer directional alignment is considered. Per the plan spec.
const BIN_W: f32 = 2.0;

/// Small tiebreak (well below any real directional cost gap — the table
/// only has 8 discrete non-zero dodge directions, at least 45 degrees
/// apart, i.e. dot-product cost gaps of >= ~0.29) added in the
/// airborne+jump branch to disambiguate rows whose `dodgeDir` alignment
/// cost is otherwise tied. This specifically resolves the "pure jump"
/// (pitch=yaw=roll=0) vs. "stall" (yaw=-roll, canceling) row collision: a
/// pure dot-product formula alone cannot tell "no directional input" from
/// "opposing diagonal input that cancels" since both have a zero-magnitude
/// `dodgeDir` — this term matches raw pitch/yaw/roll on top of the
/// dodge-alignment score to break that tie in favor of the closer raw
/// input.
const RAW_TIEBREAK_W: f32 = 0.01;

/// Tinier tiebreak preferring `row.boost==0` when `boost_amount<=0` makes
/// the boost channel itself unscored, so two rows that are otherwise
/// identical except `boost` resolve deterministically. Must stay well
/// below `RAW_TIEBREAK_W` so it never overrides a real directional
/// distinction.
const BOOST_TIEBREAK_W: f32 = 0.001;

fn pressed(v: f32) -> bool {
    v > 0.5
}

/// A table row's `pitch==0 && roll==0 && jump==0` triple is unique to the
/// 24 "ground rows" (see module doc) — every aerial row with `jump==0` has
/// at least one of pitch/roll nonzero by the table's own construction, so
/// this predicate needs no knowledge of row index/table layout beyond the
/// 8 stored floats.
fn row_is_ground_shaped(row: &[f32; 8]) -> bool {
    row[2] == 0.0 && row[4] == 0.0 && !pressed(row[5])
}

/// `BIN_W` if `r_boost != a_boost` and boost is scored; `BOOST_TIEBREAK_W`
/// if boost is unscored and `r_boost` is true (prefer `row.boost==0`); 0
/// otherwise.
fn boost_cost(boost_scored: bool, a_boost: bool, r_boost: bool) -> f32 {
    if boost_scored {
        if r_boost != a_boost {
            BIN_W
        } else {
            0.0
        }
    } else if r_boost {
        BOOST_TIEBREAK_W
    } else {
        0.0
    }
}

/// Projects an 8-dim replay control vector (`[throttle, steer, pitch, yaw,
/// roll, jump, boost, handbrake]`, matching `reconstruct::Tick::actions`'
/// layout) onto the index of the nearest row of `table` (typically
/// `construct_engine::actions::make_lookup_table_v1()`, 92 rows), using
/// `on_ground`/`has_flip`/`boost_amount` to decide which channels matter —
/// porting the logic (not the code) of rlgym-tools' `pick_action`.
///
/// - **Grounded** (`on_ground`): scores `throttle` and `steer` (matched
///   directly against the row's steer slot — ground rows already store
///   `yaw == steer`, so there's no separate yaw target to reconcile) plus
///   `handbrake`. `pitch`/`yaw`/`roll` aren't independently scored (they
///   don't affect a grounded car). If `jump` is pressed, ground rows are
///   never valid candidates (they're all `jump==0` by construction) —
///   candidates narrow to the table's two "pure jump" rows (`pitch=yaw=
///   roll=0, jump=1`), matched on `boost` only (a grounded jump-off has no
///   dodge direction to register).
/// - **Airborne** (`!on_ground`): if `jump` is pressed AND `has_flip`,
///   candidates narrow to `row.jump==1` rows, scored by dodge-direction
///   alignment: `dodgeDir = (-pitch, yaw + roll)`, cost `1 -
///   cos(angle(action, row))` (zero-magnitude `dodgeDir`s — pure-jump and
///   stall rows alike — are treated as a tie, broken by `RAW_TIEBREAK_W`).
///   Otherwise (jump not pressed, or pressed with no flip available, which
///   does nothing in-sim) candidates narrow to `row.jump==0` (excluding
///   dodge rows AND both stall rows), scored on `|a-b|` over pitch/yaw/roll.
/// - **Boost**: scored (`BIN_W` on mismatch) only when `boost_amount > 0`
///   — at 0 boost, holding boost does nothing in-sim, so it must not
///   distinguish otherwise-identical rows (`boost_cost`'s tiny tiebreak
///   still prefers `row.boost==0` so the choice is deterministic).
/// - **Handbrake**: scored only when grounded — airborne it's overloaded by
///   the table's auto-handbrake-on-flip-rows construction, not an
///   independent signal.
///
/// Ties (equal cost) resolve to the lowest row index.
pub fn project_action(
    action: &[f32; 8],
    on_ground: bool,
    has_flip: bool,
    boost_amount: f32,
    table: &[[f32; 8]],
) -> usize {
    let a_throttle = action[0];
    let a_steer = action[1];
    let a_pitch = action[2];
    let a_yaw = action[3];
    let a_roll = action[4];
    let a_jump = pressed(action[5]);
    let a_boost = pressed(action[6]);
    let a_handbrake = pressed(action[7]);

    let boost_scored = boost_amount > 0.0;

    let mut best_idx = 0usize;
    let mut best_cost = f32::INFINITY;

    for (idx, row) in table.iter().enumerate() {
        let r_pitch = row[2];
        let r_yaw = row[3];
        let r_roll = row[4];
        let r_jump = pressed(row[5]);
        let r_boost = pressed(row[6]);
        let r_handbrake = pressed(row[7]);
        let ground_shaped = row_is_ground_shaped(row);

        let cost: f32;
        if on_ground {
            if ground_shaped {
                if a_jump {
                    // Ground rows are all jump==0; a pressed jump can never
                    // land here (see the pure-jump branch below instead).
                    continue;
                }
                let mut c = (a_throttle - row[0]).abs() + (a_steer - row[1]).abs();
                if r_handbrake != a_handbrake {
                    c += BIN_W;
                }
                c += boost_cost(boost_scored, a_boost, r_boost);
                cost = c;
            } else if a_jump && r_jump && r_pitch == 0.0 && r_roll == 0.0 {
                // "Pure jump" row: a grounded jump-off has no dodge
                // direction to register, so only boost distinguishes the
                // two candidate rows.
                cost = boost_cost(boost_scored, a_boost, r_boost);
            } else {
                continue;
            }
        } else {
            if ground_shaped {
                // Ground rows never represent a valid airborne action.
                continue;
            }
            if a_jump && has_flip {
                if !r_jump {
                    continue;
                }
                let (adx, ady) = (-a_pitch, a_yaw + a_roll);
                let (rdx, rdy) = (-r_pitch, r_yaw + r_roll);
                let amag = (adx * adx + ady * ady).sqrt();
                let rmag = (rdx * rdx + rdy * rdy).sqrt();
                let dodge_cost = if amag < 1e-4 && rmag < 1e-4 {
                    0.0
                } else if amag < 1e-4 || rmag < 1e-4 {
                    1.0
                } else {
                    let dot = (adx * rdx + ady * rdy) / (amag * rmag);
                    1.0 - dot.clamp(-1.0, 1.0)
                };
                let raw_tiebreak = RAW_TIEBREAK_W
                    * ((a_pitch - r_pitch).abs() + (a_yaw - r_yaw).abs() + (a_roll - r_roll).abs());
                cost = dodge_cost + raw_tiebreak + boost_cost(boost_scored, a_boost, r_boost);
            } else {
                if r_jump {
                    // Excludes dodge rows AND both stall rows: jump isn't
                    // meaningfully pressed (either not pressed, or pressed
                    // with no flip available, which does nothing in-sim).
                    continue;
                }
                let mut c = (a_pitch - r_pitch).abs() + (a_yaw - r_yaw).abs() + (a_roll - r_roll).abs();
                c += boost_cost(boost_scored, a_boost, r_boost);
                cost = c;
            }
        }

        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use construct_engine::actions::make_lookup_table_v1;

    #[test]
    fn every_row_has_a_home_context() {
        // Smoke test complementing the integration tests in
        // `replay/tests/project_test.rs`: every table row is reachable as
        // SOME context's argmin (a much weaker property than exact
        // self-identity, just guards against an accidental dead row).
        let table = make_lookup_table_v1();
        let mut reached = vec![false; table.len()];
        for (i, row) in table.iter().enumerate() {
            let on_ground = i < 24;
            let got = project_action(row, on_ground, true, 1.0, &table);
            reached[got] = true;
        }
        assert!(reached.iter().all(|&r| r), "every table row must be reachable via self-projection");
    }
}
