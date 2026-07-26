use rocketsim_rs::sim::CarControls;

pub const TABLE_SIZE: usize = 90;
pub const TABLE_SIZE_V1: usize = 92;
pub const TABLE_SIZE_V1_AIR: usize = 104;

/// Row layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
pub fn make_lookup_table() -> Vec<[f32; 8]> {
    let mut actions: Vec<[f32; 8]> = Vec::with_capacity(TABLE_SIZE);
    // Ground
    for throttle in [-1.0f32, 0.0, 1.0] {
        for steer in [-1.0f32, 0.0, 1.0] {
            for boost in [0.0f32, 1.0] {
                for handbrake in [0.0f32, 1.0] {
                    if boost == 1.0 && throttle != 1.0 {
                        continue;
                    }
                    // Python `throttle or boost`: throttle if nonzero else boost
                    let t = if throttle != 0.0 { throttle } else { boost };
                    actions.push([t, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                }
            }
        }
    }
    // Aerial
    for pitch in [-1.0f32, 0.0, 1.0] {
        for yaw in [-1.0f32, 0.0, 1.0] {
            for roll in [-1.0f32, 0.0, 1.0] {
                for jump in [0.0f32, 1.0] {
                    for boost in [0.0f32, 1.0] {
                        if jump == 1.0 && yaw != 0.0 {
                            continue; // Only need roll for sideflip
                        }
                        if pitch == 0.0 && roll == 0.0 && jump == 0.0 {
                            continue; // Duplicate with ground
                        }
                        // Enable handbrake for potential wavedashes
                        let handbrake =
                            (jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0)) as u8 as f32;
                        actions.push([boost, yaw, pitch, yaw, roll, jump, boost, handbrake]);
                    }
                }
            }
        }
    }
    debug_assert_eq!(actions.len(), TABLE_SIZE);
    actions
}

/// Action table v1.1: the existing 90 rows APPENDED with 2 stall rows
/// (rlgym-tools-verified stall inputs; dodgeDir = (-pitch, yaw+roll) => yaw=-roll
/// => zero impulse). Append-only: indices 0-89 keep their v0 meaning.
pub fn make_lookup_table_v1() -> Vec<[f32; 8]> {
    let mut actions = make_lookup_table();
    actions.push([0., 0., 0., 1., -1., 1., 0., 1.]);
    actions.push([0., 0., 0., -1., 1., 1., 0., 1.]);
    debug_assert_eq!(actions.len(), TABLE_SIZE_V1);
    actions
}

/// Action table "v1-air": v1.1's 92 rows APPENDED with a 12-row clean-air block.
/// Append-only -- indices 0..92 keep their v1.1 meaning byte-for-byte, so this
/// table is a strict superset and every v1.1 row index still decodes the same.
///
/// WHY. v1.1 cannot express "jump without air roll". The handbrake in the aerial
/// loop above is DERIVED from rotation (`hb = jump && (pitch||yaw||roll)`, see
/// `make_lookup_table`), and the loop also skips `jump && yaw != 0`. Together
/// those two lines mean the generator structurally never emits a jump row that
/// pitches without also setting handbrake -- airborne, handbrake IS air roll. So
/// the control "climb while pitching, not spinning" is not rare in v1.1, it is
/// ABSENT at every index. Measured on the 92-row table:
///   * pure jump (jump, no pitch/yaw/roll, no hb):        2 rows  =  2.17%  (56, 57)
///   * jump+boost with no yaw/roll AND no air roll:       1 row   =  1.09%  (57)
///   * of the 20 jump rows, 18 force handbrake=1          -> P(clean jump | jump) = 10%
/// and the aerial primitive must be HELD for several consecutive decisions to
/// clear a hop. At tick_skip=8 (15 decisions/s) a uniform policy holds a clean
/// jump for 3 straight decisions (~0.2 s, the full-height takeoff that
/// `configs/reward_v9_aerial.toml`'s aerial_z_lo = 400 requires -- that config
/// measures hop 231.9 / double jump 393.6 / hop+boost 612.6 / full aerial 1238.1,
/// i.e. everything it pays for is above a hop) with probability 1.03e-5: once per
/// 97,336 decisions, ~0.96 times per 93,440-row iteration, one per 108 minutes of
/// car time. No reward can shape an action sequence that is never sampled. This
/// is the "bots naturally forget jumping" failure ZealanL (the RocketSim author)
/// names in the RLGym-PPO-Guide, whose prescription is air rewards AND
/// "doubling jump actions in discrete action parsers" -- v9 had the reward half
/// and not the action half.
///
/// WHAT. The 12 appended rows are the clean-air grid (pitch in {-1,0,+1} x boost
/// in {0,1}, jump=1, steer=yaw=roll=handbrake=0) emitted TWICE -- the "doubling"
/// above. `throttle = boost` and `steer = yaw = 0` follow the existing aerial-loop
/// convention, which makes 94/95 and 100/101 byte-identical duplicates of 56/57
/// rather than near-misses. Duplication is a free prior here and not a wasted
/// slot: `model_v1.py` computes logits as `policy_dot(pooled) @ act_embed(table).T`,
/// a function of the 8-float ROW rather than a per-slot parameter, so k copies of
/// a row multiply that control's sampled mass and gradient throughput by exactly
/// k (verified: logit[56] == logit[94] to 0.0, P({56,94,100}) = 3 x P(56)) while
/// staying impossible to tell apart, and `p_single` still decays to zero if the
/// control turns out to be bad. Under a plain `nn.Linear(d, N)` head this trick
/// would instead burn N parameters that drift apart.
///
/// EFFECT (uniform policy, the state a fresh from-scratch net starts in):
///   clean jump (jump, yaw=roll=0, hb=0)   2/92 = 2.17%  ->  14/104 = 13.46%
///   jump+boost, no yaw/roll, no air roll  1/92 = 1.09%  ->   7/104 =  6.73%
///   P(clean jump | jump)                        10.0%   ->          43.8%
///   P(hold 3, the full-height takeoff)    1.03e-5       ->  2.44e-3   (237x)
///   -> per 93,440-row iteration: 0.96 -> 228 takeoffs; per car, one per
///      108 minutes -> one per 27 seconds.
/// Dodge-capable share does rise 19.57% -> 25.00%, which is honest but reads the
/// physics wrong: a dodge needs a jump RISING EDGE while airborne with a flip
/// available, whereas rows 92..104 held from the ground are a takeoff WITH pitch
/// authority -- the input a human uses. The thing v1.1 genuinely cannot turn off,
/// forced air roll on 90% of jump rows, drops to 56%.
///
/// NOT DONE, deliberately: no yaw rows (`jump + yaw` is dodgeDir = (-pitch,
/// yaw+roll) != 0, a side dodge, not an aerial -- the v0 generator's
/// `jump && yaw != 0 -> continue` was right); no new sustain rows ("boost, no
/// jump" is already 28.85%, the sustain phase was never the bottleneck); and
/// NOTHING above is a mutation of rows 0..92 -- clearing handbrake on the
/// existing rows 32..83 would re-mean live indices for every v1.1 checkpoint.
/// Sized at +12 because +6 (grid once, no doubling) buys only 53x on the
/// load-bearing P(hold 3) where +12 buys 237x, and +18 buys a further 2.5x while
/// making one control class 18% of the action space.
pub fn make_lookup_table_v1_air() -> Vec<[f32; 8]> {
    let mut actions = make_lookup_table_v1();
    // Doubled per ZealanL's "doubling jump actions in discrete action parsers".
    for _ in 0..2 {
        for pitch in [-1.0f32, 0.0, 1.0] {
            for boost in [0.0f32, 1.0] {
                //            throttle steer  pitch  yaw  roll  jump  boost  handbrake
                actions.push([boost, 0.0, pitch, 0.0, 0.0, 1.0, boost, 0.0]);
            }
        }
    }
    debug_assert_eq!(actions.len(), TABLE_SIZE_V1_AIR);
    actions
}

/// Immortal's (RLMarlbot) 126-row action table. Distinct from `make_lookup_table`
/// (our 90-row v0 table): different layout quirks -- `throttle or boost` uses
/// Python truthy-or semantics (0 -> boost, else throttle), `steer` is written into
/// BOTH the steer(1) and yaw(3) slots on ground, aerial rows force handbrake=1.
/// Reproduces rlmarlbot/immortal/action/actionparser.py::ImmortalAction byte-exact.
/// Row layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
pub fn make_immortal_table() -> Vec<[f32; 8]> {
    let mut actions: Vec<[f32; 8]> = Vec::with_capacity(126);
    // Ground (36)
    for throttle in [-1.0f32, 0.0, 1.0] {
        for steer in [-1.0f32, 0.0, 1.0] {
            for boost in [0.0f32, 1.0] {
                for handbrake in [0.0f32, 1.0] {
                    if boost == 1.0 && throttle != 1.0 {
                        continue;
                    }
                    // Python `throttle or boost`: throttle if nonzero else boost
                    let t = if throttle != 0.0 { throttle } else { boost };
                    actions.push([t, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                }
            }
        }
    }
    // Aerial (90)
    for pitch in [-1.0f32, 0.0, 1.0] {
        for yaw in [-1.0f32, 0.0, 1.0] {
            for roll in [-1.0f32, 0.0, 1.0] {
                for jump in [0.0f32, 1.0] {
                    for boost in [0.0f32, 1.0] {
                        if pitch == 0.0 && roll == 0.0 && jump == 0.0 {
                            continue;
                        }
                        actions.push([boost, yaw, pitch, yaw, roll, jump, boost, 1.0]);
                    }
                }
            }
        }
    }
    debug_assert_eq!(actions.len(), 126);
    actions
}

pub fn to_controls(row: &[f32; 8]) -> CarControls {
    CarControls {
        throttle: row[0],
        steer: row[1],
        pitch: row[2],
        yaw: row[3],
        roll: row[4],
        jump: row[5] != 0.0,
        boost: row[6] != 0.0,
        handbrake: row[7] != 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_90_rows() {
        assert_eq!(make_lookup_table().len(), TABLE_SIZE);
    }

    #[test]
    fn first_ground_row_matches_rlgym_reference() {
        // throttle=-1, steer=-1, boost=0, handbrake=0 -> [-1,-1,0,-1,0,0,0,0]
        assert_eq!(make_lookup_table()[0], [-1., -1., 0., -1., 0., 0., 0., 0.]);
    }

    #[test]
    fn ground_rows_count_24() {
        // rows with pitch==roll==jump==0 produced by the ground loop
        let n = make_lookup_table().iter().take(24).count();
        assert_eq!(n, 24);
        // 25th row is the first aerial row
        let t = make_lookup_table();
        assert!(t[24][5] != 0.0 || t[24][2] != 0.0 || t[24][4] != 0.0);
    }

    #[test]
    fn to_controls_maps_booleans() {
        let c = to_controls(&[1., 0., 0., 0., 0., 1., 1., 0.]);
        assert_eq!(c.throttle, 1.0);
        assert!(c.jump && c.boost && !c.handbrake);
    }

    #[test]
    fn v1_table_appends_stalls_only() {
        let v0 = make_lookup_table();
        let v1 = make_lookup_table_v1();
        assert_eq!(v1.len(), TABLE_SIZE_V1);
        assert_eq!(TABLE_SIZE_V1, 92);
        assert_eq!(&v1[..90], &v0[..]);
        assert_eq!(v1[90], [0., 0., 0., 1., -1., 1., 0., 1.]);
        assert_eq!(v1[91], [0., 0., 0., -1., 1., 1., 0., 1.]);
    }

    #[test]
    fn v1_air_appends_12_and_prefix_is_v1() {
        // Append-only is the whole safety argument: every v1.1 index keeps its
        // meaning, so a v1.1 checkpoint's logit i still decodes to control i.
        let v1 = make_lookup_table_v1();
        let air = make_lookup_table_v1_air();
        assert_eq!(air.len(), TABLE_SIZE_V1_AIR);
        assert_eq!(TABLE_SIZE_V1_AIR, 104);
        assert_eq!(&air[..TABLE_SIZE_V1], &v1[..]);
    }

    #[test]
    fn v1_air_row_values() {
        let air = make_lookup_table_v1_air();
        // pitch in {-1,0,+1} x boost in {0,1}, jump=1, steer=yaw=roll=hb=0,
        // throttle = boost (the aerial-loop convention), emitted twice.
        for base in [92usize, 98] {
            assert_eq!(air[base], [0., 0., -1., 0., 0., 1., 0., 0.]); // nose down
            assert_eq!(air[base + 1], [1., 0., -1., 0., 0., 1., 1., 0.]); // nose down + boost
            assert_eq!(air[base + 2], [0., 0., 0., 0., 0., 1., 0., 0.]); // pure jump
            assert_eq!(air[base + 3], [1., 0., 0., 0., 0., 1., 1., 0.]); // pure jump + boost
            assert_eq!(air[base + 4], [0., 0., 1., 0., 0., 1., 0., 0.]); // nose up
            assert_eq!(air[base + 5], [1., 0., 1., 0., 0., 1., 1., 0.]); // THE aerial primitive
        }
        // Byte-identical to v1.1's two clean-jump rows, not near-misses.
        assert_eq!(air[94], air[56]);
        assert_eq!(air[95], air[57]);
        assert_eq!(air[100], air[56]);
        assert_eq!(air[101], air[57]);
        // No row in the appended block sets air roll or yaw.
        for r in &air[TABLE_SIZE_V1..] {
            assert_eq!(r[3], 0.0, "yaw must be 0: a jump with yaw is a side dodge");
            assert_eq!(r[4], 0.0, "roll must be 0");
            assert_eq!(r[7], 0.0, "handbrake airborne IS air roll -- must be 0");
            assert_eq!(r[5], 1.0, "every appended row is a jump row");
        }
    }

    #[test]
    fn clean_jump_share_is_14_of_104() {
        // The number the change exists to move. "Clean jump" = jump with no yaw,
        // no roll and no handbrake, i.e. a takeoff that is not also an air roll
        // and not a side dodge. 2/92 = 2.17% -> 14/104 = 13.46%; the probability
        // of HOLDING it for 3 consecutive decisions (the full-height takeoff)
        // goes 1.03e-5 -> 2.44e-3, a 237x change in how often a fresh policy
        // ever samples the sequence the aerial reward pays for.
        let clean = |t: &Vec<[f32; 8]>| {
            t.iter()
                .filter(|r| r[5] != 0.0 && r[3] == 0.0 && r[4] == 0.0 && r[7] == 0.0)
                .count()
        };
        let v1 = make_lookup_table_v1();
        let air = make_lookup_table_v1_air();
        assert_eq!(clean(&v1), 2);
        assert_eq!(clean(&air), 14);
        // P(clean jump | jump): 2/20 = 10% -> 14/32 = 43.75%.
        let jumps = |t: &Vec<[f32; 8]>| t.iter().filter(|r| r[5] != 0.0).count();
        assert_eq!(jumps(&v1), 20);
        assert_eq!(jumps(&air), 32);
        // Forced-air-roll jump rows are unchanged in COUNT (append-only never
        // mutates them); only their share of the table falls, 19.57% -> 17.31%.
        let rollers = |t: &Vec<[f32; 8]>| {
            t.iter().filter(|r| r[5] != 0.0 && r[7] != 0.0).count()
        };
        assert_eq!(rollers(&v1), 18);
        assert_eq!(rollers(&air), 18);
    }

    #[test]
    fn duplicate_multiplicities() {
        // Duplication is the mechanism, so pin it. Logits are a function of the
        // ROW (act_embed(table)), so k copies = exactly k x the sampled mass and
        // k x the gradient throughput for that control, with no way for the
        // policy to waste capacity distinguishing them.
        let air = make_lookup_table_v1_air();
        let count = |row: [f32; 8]| air.iter().filter(|r| **r == row).count();
        // The two whose rising edge is a DOUBLE JUMP (not a dodge) get the most
        // mass: 56/57 already existed, plus one copy each from the two grids.
        // Double jump reaches z~525 vs a hop's z~232 -- that is the height ceiling.
        assert_eq!(count([0., 0., 0., 0., 0., 1., 0., 0.]), 3); // pure jump: 56, 94, 100
        assert_eq!(count([1., 0., 0., 0., 0., 1., 1., 0.]), 3); // + boost:   57, 95, 101
        // The four pitch=+-1 controls are new, so exactly the 2 copies we pushed.
        assert_eq!(count([0., 0., -1., 0., 0., 1., 0., 0.]), 2);
        assert_eq!(count([1., 0., -1., 0., 0., 1., 1., 0.]), 2);
        assert_eq!(count([0., 0., 1., 0., 0., 1., 0., 0.]), 2);
        assert_eq!(count([1., 0., 1., 0., 0., 1., 1., 0.]), 2);
    }
}
