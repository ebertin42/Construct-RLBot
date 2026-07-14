use construct_engine::policy::{LayerWeights, MlpPolicy, PolicyWeights};

fn fixture() -> serde_json::Value {
    let text = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/policy_fixture.json"),
    )
    .expect("run scripts/gen_policy_fixture.py first");
    serde_json::from_str(&text).unwrap()
}

fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn layer(fx: &serde_json::Value, prefix: &str) -> LayerWeights {
    let sd = &fx["state_dict"];
    let shape = fx["shapes"][format!("{prefix}.weight")].as_array().unwrap();
    LayerWeights {
        w: f32s(&sd[format!("{prefix}.weight")]),
        b: f32s(&sd[format!("{prefix}.bias")]),
        out_dim: shape[0].as_u64().unwrap() as usize,
        in_dim: shape[1].as_u64().unwrap() as usize,
    }
}

#[test]
fn candle_forward_matches_pytorch_golden() {
    let fx = fixture();
    let weights = PolicyWeights {
        trunk: vec![layer(&fx, "trunk.0"), layer(&fx, "trunk.2")],
        policy: layer(&fx, "policy_head"),
        value: layer(&fx, "value_head"),
    };
    let net = MlpPolicy::new(&weights).unwrap();
    let obs = f32s(&fx["obs"]);
    let batch = fx["batch"].as_u64().unwrap() as usize;
    let (logits, values) = net.forward(&obs, batch, 94).unwrap();

    let exp_logits = f32s(&fx["expected_logits"]);
    let exp_values = f32s(&fx["expected_values"]);
    assert_eq!(logits.len(), exp_logits.len());
    let max_l = logits.iter().zip(&exp_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let max_v = values.iter().zip(&exp_values).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max_l < 1e-4, "logits max diff {max_l}");
    assert!(max_v < 1e-4, "values max diff {max_v}");
}
