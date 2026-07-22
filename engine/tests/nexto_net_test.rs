//! Golden test for the candle port of Nexto's EARLPerceiver + ControlsPredictorDot.
//!
//! Compares against logits produced by the REAL TorchScript model on the same
//! (q, kv, mask) inputs (see scripts/gen_nexto_obs_fixture.py).
//!
//! SKIPS when the extracted weights are absent -- they are not committed (binary,
//! and CC BY-NC-SA carries obligations). Produce them with:
//!     scripts/extract_nexto_weights.py deploy/external/nexto/nexto-model.pt
use std::collections::HashMap;
use std::fs::File;

use construct_engine::actions::make_lookup_table;
use construct_engine::nexto::{NextoNet, NEXTO_ACTIONS};
use ndarray::IxDyn;
use ndarray_npy::NpzReader;

fn weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::Path::new(&home).join(".cache/construct/nexto_weights.npz")
}

fn load_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut npz = NpzReader::new(File::open(weights_path()).unwrap()).unwrap();
    let names = npz.names().unwrap();
    let mut map = HashMap::new();
    for n in names {
        let a: ndarray::Array<f32, IxDyn> = npz.by_name(&n).unwrap();
        let shape = a.shape().to_vec();
        let data: Vec<f32> = a.iter().copied().collect();
        // npz entries are named "<param>.npy"
        let key = n.trim_end_matches(".npy").to_string();
        map.insert(key, (data, shape));
    }
    map
}

#[test]
fn nexto_net_matches_torch_logits() {
    if !weights_path().exists() {
        eprintln!("SKIP: {} absent (weights are not committed)", weights_path().display());
        return;
    }
    let map = load_weights();
    let table = make_lookup_table();
    let net = NextoNet::new(&map, &table).expect("build NextoNet");

    let raw = std::fs::read_to_string("tests/fixtures/nexto_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let mut worst = 0.0f32;
    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let q: Vec<f32> = case["q"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() as f32).collect();
        let kv: Vec<f32> = case["kv"].as_array().unwrap().iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
            .collect();
        let mask: Vec<bool> = case["mask"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() != 0.0).collect();
        let expect: Vec<f32> = case["logits"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() as f32).collect();
        assert_eq!(expect.len(), NEXTO_ACTIONS);

        let got = net.forward(&q, &kv, &mask).expect("forward");
        assert_eq!(got.len(), NEXTO_ACTIONS, "case {ci} logit count");
        for k in 0..NEXTO_ACTIONS {
            let d = (got[k] - expect[k]).abs();
            if d > worst { worst = d; }
            assert!(d < 2e-3, "case {ci} logit {k}: rust {} torch {}", got[k], expect[k]);
        }
        // the argmax is what actually drives the car -- pin it exactly
        let am = |v: &[f32]| v.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
        assert_eq!(am(&got), am(&expect), "case {ci} argmax differs");
    }
    eprintln!("max abs logit diff: {worst:e}");
}

fn necto_weights_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::Path::new(&home).join(".cache/construct/necto_weights.npz")
}

/// Necto shares the obs contract and EARL trunk with Nexto but has 1 block, NO
/// LayerNorms, and a multi-discrete head split [3,3,2,2,2].
#[test]
fn necto_net_matches_torch_heads() {
    if !necto_weights_path().exists() {
        eprintln!("SKIP: {} absent", necto_weights_path().display());
        return;
    }
    let mut npz = NpzReader::new(File::open(necto_weights_path()).unwrap()).unwrap();
    let mut map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for n in npz.names().unwrap() {
        let a: ndarray::Array<f32, IxDyn> = npz.by_name(&n).unwrap();
        map.insert(n.trim_end_matches(".npy").to_string(),
                   (a.iter().copied().collect(), a.shape().to_vec()));
    }
    let table = make_lookup_table();
    let net = NextoNet::new(&map, &table).expect("build Necto net");

    let raw = std::fs::read_to_string("tests/fixtures/nexto_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let q: Vec<f32> = case["q"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() as f32).collect();
        let kv: Vec<f32> = case["kv"].as_array().unwrap().iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
            .collect();
        let mask: Vec<bool> = case["mask"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap() != 0.0).collect();
        let expect = case["necto_heads"].as_array().unwrap();

        match net.forward_head(&q, &kv, &mask).expect("forward") {
            construct_engine::nexto::HeadOut::MultiDiscrete(heads) => {
                assert_eq!(heads.len(), expect.len(), "case {ci} head count");
                for (h, eh) in heads.iter().zip(expect) {
                    let ev = eh.as_array().unwrap();
                    assert_eq!(h.len(), ev.len(), "case {ci} head width");
                    for k in 0..h.len() {
                        let e = ev[k].as_f64().unwrap() as f32;
                        assert!((h[k] - e).abs() < 2e-3,
                                "case {ci} head logit {k}: rust {} torch {}", h[k], e);
                    }
                }
                // The decoded controls are what actually drive the car -- check
                // them against the REFERENCE decode (agent.py), not just for
                // finiteness. A wrong head order yields correct logits and a
                // bot that drives into a wall.
                let c = construct_engine::nexto::decode_multi_discrete(&heads);
                let ec = case["necto_controls"].as_array().unwrap();
                for k in 0..8 {
                    let e = ec[k].as_f64().unwrap() as f32;
                    assert!((c[k] - e).abs() < 1e-6,
                            "case {ci} control {k}: rust {} py {}", c[k], e);
                }
            }
            _ => panic!("case {ci}: expected a multi-discrete head"),
        }
    }
}
