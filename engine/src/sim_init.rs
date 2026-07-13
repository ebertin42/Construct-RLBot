use std::sync::Once;

static INIT: Once = Once::new();

/// Idempotent RocketSim global init. Default path works from repo root and from engine/.
pub fn ensure_init(meshes_path: Option<&str>) {
    INIT.call_once(|| {
        let path = meshes_path.map(String::from).unwrap_or_else(|| {
            for candidate in ["assets/collision_meshes", "../assets/collision_meshes"] {
                if std::path::Path::new(candidate).join("soccar").exists() {
                    return candidate.to_string();
                }
            }
            "assets/collision_meshes".to_string()
        });
        rocketsim_rs::init(Some(&path), true);
    });
}
