//! Idempotent RocketSim global init.
//!
//! `rocketsim_rs::sim::Arena` construction segfaults with a fatal-error
//! message ("RocketSim has not been initialized, call RocketSim::Init()
//! first") unless `rocketsim_rs::init(..)` has already loaded the collision
//! meshes once per process. Mirrors `engine/src/sim_init.rs`'s asset layout
//! (`assets/collision_meshes/soccar`) and candidate search — this crate's
//! tests run with cwd = `replay/` (candidate 2: `../assets/collision_meshes`
//! resolves to the shared workspace-root `assets/` dir also used by
//! `engine/`), while a release binary invoked from the workspace root hits
//! candidate 1 directly.

use std::sync::Once;

static INIT: Once = Once::new();

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
