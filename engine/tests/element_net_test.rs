//! Golden test for the candle port of Element's actor, against the REAL Element
//! Agent (see scripts/gen_element_fixture.py). Compares the decoded controls-8
//! action, which is what actually drives the car -- a wrong head split would give
//! plausible logits and the wrong action.
//!
//! SKIPS when the (unlicensed, uncommitted) weights are absent.
use std::collections::HashMap;
use std::fs::File;

use construct_engine::element::ElementNet;
use ndarray::IxDyn;
use ndarray_npy::NpzReader;

fn weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::Path::new(&home).join(".cache/construct/element_weights.npz")
}

#[test]
fn element_net_matches_python_agent() {
    if !weights_path().exists() {
        eprintln!("SKIP: {} absent", weights_path().display());
        return;
    }
    let mut npz = NpzReader::new(File::open(weights_path()).unwrap()).unwrap();
    let mut map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for n in npz.names().unwrap() {
        let a: ndarray::Array<f32, IxDyn> = npz.by_name(&n).unwrap();
        map.insert(n.trim_end_matches(".npy").to_string(),
                   (a.iter().copied().collect(), a.shape().to_vec()));
    }

    let raw = std::fs::read_to_string("tests/fixtures/element_net.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let obs_size = v["obs_size"].as_u64().unwrap() as usize;
    let net = ElementNet::new(&map, obs_size).expect("build ElementNet");

    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let obs: Vec<f32> = case["obs"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() as f32).collect();
        let expect: Vec<f32> = case["action"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() as f32).collect();
        let got = net.decide(&obs).expect("decide");
        for k in 0..8 {
            assert!((got[k] - expect[k]).abs() < 1e-6,
                    "case {ci} action[{k}]: rust {} py {}", got[k], expect[k]);
        }
    }
}

#[test]
fn element_rejects_wrong_obs_size() {
    if !weights_path().exists() { return; }
    let mut npz = NpzReader::new(File::open(weights_path()).unwrap()).unwrap();
    let mut map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for n in npz.names().unwrap() {
        let a: ndarray::Array<f32, IxDyn> = npz.by_name(&n).unwrap();
        map.insert(n.trim_end_matches(".npy").to_string(),
                   (a.iter().copied().collect(), a.shape().to_vec()));
    }
    assert!(ElementNet::new(&map, 99).is_err(), "should reject a mismatched obs width");
}
