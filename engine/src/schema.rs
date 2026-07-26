use crate::actions::{TABLE_SIZE_V1, TABLE_SIZE_V1_AIR};
use crate::episode::ActionTableKind;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub pos_norm: f64,
    pub vel_norm: f64,
    pub ang_vel_norm: f64,
}

/// Obs v1 entity-layout metadata (see docs/superpowers/plans/2026-07-16-entity-transformer-obs-v1.md,
/// "Obs v1 layout"). Constants duplicated here (not imported from an `obs_v1`
/// module, which doesn't exist yet as of T2) are reconciled when T3 lands.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ObsV1Meta {
    pub max_ent: usize,
    pub ent_feat: usize,
    pub q_feat: usize,
    pub prev_actions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Schema {
    pub version: u32,
    pub obs_size: usize,
    pub action_table: String,
    pub action_count: usize,
    pub tick_skip: u32,
    pub normalization: Normalization,
    #[serde(default)]
    pub obs_v1: Option<ObsV1Meta>,
}

impl Schema {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let schema: Schema = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        schema.validate()?;
        Ok(schema)
    }

    fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            // v0 (and any future non-1 version) validation is unchanged: no
            // compiled-constant checks beyond successful deserialization.
            return Ok(());
        }
        const EXPECTED: ObsV1Meta = ObsV1Meta {
            max_ent: 17,
            ent_feat: 26,
            q_feat: 64,
            prev_actions: 5,
        };
        match &self.obs_v1 {
            None => Err("schema version 1 requires an [obs_v1] section".to_string()),
            Some(meta) if *meta != EXPECTED => Err(format!(
                "schema version 1 obs_v1 mismatch: expected {EXPECTED:?}, got {meta:?}"
            )),
            // Two action tables are legal under version 1: the frozen 92-row
            // v1.1 table and the 104-row v1-air table (v1.1 + a doubled
            // clean-air block, append-only). The obs contract is identical
            // between them -- only the policy's output dimension differs --
            // so the table is a second axis on the schema rather than a
            // version bump. Anything else is a stale/mismatched schema.
            Some(_) if self.action_count != TABLE_SIZE_V1 && self.action_count != TABLE_SIZE_V1_AIR => {
                Err(format!(
                    "schema version 1 requires action_count = {TABLE_SIZE_V1} or {TABLE_SIZE_V1_AIR}, got {}",
                    self.action_count
                ))
            }
            Some(_) => Ok(()),
        }
    }

    /// Which V1 action table this schema selects. Dispatches on `action_count`
    /// (the field `validate` already constrains) rather than the
    /// `action_table` name string, so a typo in the name cannot silently
    /// select the wrong table width.
    pub fn action_table_kind(&self) -> ActionTableKind {
        if self.action_count == TABLE_SIZE_V1_AIR {
            ActionTableKind::V1Air
        } else {
            ActionTableKind::V1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_v0() {
        let s = Schema::load("../schema/v0.toml").unwrap();
        assert_eq!(s.version, 0);
        assert_eq!(s.obs_size, 94);
        assert_eq!(s.action_count, 90);
        assert_eq!(s.tick_skip, 8);
        assert!((s.normalization.pos_norm - 1.0 / 2300.0).abs() < 1e-12);
        assert_eq!(s.obs_v1, None);
    }

    #[test]
    fn loads_v1() {
        let s = Schema::load("../schema/v1.toml").unwrap();
        assert_eq!(s.version, 1);
        assert_eq!(s.action_count, 92);
        assert_eq!(s.tick_skip, 8);
        assert!((s.normalization.pos_norm - 1.0 / 2300.0).abs() < 1e-12);
        assert!((s.normalization.vel_norm - 1.0 / 2300.0).abs() < 1e-12);
        assert!((s.normalization.ang_vel_norm - 1.0 / 5.5).abs() < 1e-12);
        assert_eq!(
            s.obs_v1,
            Some(ObsV1Meta {
                max_ent: 17,
                ent_feat: 26,
                q_feat: 64,
                prev_actions: 5,
            })
        );
        // v1.1 stays on the 92-row table -- this is the frozen lineage every
        // existing checkpoint and gate result was scored on.
        assert_eq!(s.action_table_kind(), ActionTableKind::V1);
    }

    #[test]
    fn loads_v1_air() {
        let s = Schema::load("../schema/v1_air.toml").unwrap();
        // version stays 1: the OBS contract is byte-identical to v1.toml, only
        // the action table (the policy's output dimension) differs.
        assert_eq!(s.version, 1);
        assert_eq!(s.action_count, TABLE_SIZE_V1_AIR);
        assert_eq!(s.action_table, "construct_104_v1air");
        assert_eq!(s.tick_skip, 8);
        assert_eq!(s.action_table_kind(), ActionTableKind::V1Air);
        let v1 = Schema::load("../schema/v1.toml").unwrap();
        assert_eq!(s.obs_v1, v1.obs_v1, "obs contract must be identical to v1");
        assert_eq!(s.obs_size, v1.obs_size);
        assert_eq!(s.tick_skip, v1.tick_skip);
    }

    #[test]
    fn rejects_v1_with_an_unknown_action_count() {
        let dir = std::env::temp_dir().join("construct_schema_bad_action_count");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.toml");
        std::fs::write(
            &path,
            "version = 1\nobs_size = 0\naction_table = \"nope\"\naction_count = 100\n\
             tick_skip = 8\n\n[obs_v1]\nmax_ent = 17\nent_feat = 26\nq_feat = 64\n\
             prev_actions = 5\n\n[normalization]\npos_norm = 1.0\nvel_norm = 1.0\n\
             ang_vel_norm = 1.0\n",
        )
        .unwrap();
        let err = Schema::load(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("92"), "error should name both legal widths: {err}");
        assert!(err.contains("104"), "error should name both legal widths: {err}");
    }
}
