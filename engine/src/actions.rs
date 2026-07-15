use rocketsim_rs::sim::CarControls;

pub const TABLE_SIZE: usize = 90;
pub const TABLE_SIZE_V1: usize = 92;

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
}
