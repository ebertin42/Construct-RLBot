use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub pos_norm: f64,
    pub vel_norm: f64,
    pub ang_vel_norm: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Schema {
    pub version: u32,
    pub obs_size: usize,
    pub action_table: String,
    pub action_count: usize,
    pub tick_skip: u32,
    pub normalization: Normalization,
}

impl Schema {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        toml::from_str(&text).map_err(|e| format!("{path}: {e}"))
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
    }
}
