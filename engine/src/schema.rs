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
            Some(_) if self.action_count != 92 => Err(format!(
                "schema version 1 requires action_count = 92, got {}",
                self.action_count
            )),
            Some(_) => Ok(()),
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
    }
}
