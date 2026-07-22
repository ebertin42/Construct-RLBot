use construct_engine::foreign::ForeignMlp;
use std::collections::HashMap;

#[test]
fn foreign_mlp_forward_matches_torch() {
    let raw = std::fs::read_to_string("tests/fixtures/foreign_mlp.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let in_dim = v["in_dim"].as_u64().unwrap() as usize;
    let hidden = v["hidden"].as_u64().unwrap() as usize;
    let out_dim = v["out_dim"].as_u64().unwrap() as usize;
    let n_hidden = v["n_hidden"].as_u64().unwrap() as usize;

    let mut w: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (k, val) in v["weights"].as_object().unwrap() {
        let arr = val.as_array().unwrap();
        if arr[0].is_array() {
            let rows = arr.len();
            let cols = arr[0].as_array().unwrap().len();
            let mut flat = Vec::with_capacity(rows * cols);
            for r in arr {
                for c in r.as_array().unwrap() {
                    flat.push(c.as_f64().unwrap() as f32);
                }
            }
            w.insert(k.clone(), (flat, vec![rows, cols]));
        } else {
            let flat: Vec<f32> = arr.iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let n = flat.len();
            w.insert(k.clone(), (flat, vec![n]));
        }
    }
    let mlp = ForeignMlp::from_named(&w, in_dim, hidden, out_dim, n_hidden).unwrap();

    let inputs = v["inputs"].as_array().unwrap();
    let expect = v["logits"].as_array().unwrap();
    for (bi, row) in inputs.iter().enumerate() {
        let obs: Vec<f32> = row
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let got = mlp.forward(&obs, 1, in_dim).unwrap();
        let exp = expect[bi].as_array().unwrap();
        for j in 0..out_dim {
            let e = exp[j].as_f64().unwrap() as f32;
            assert!((got[j] - e).abs() < 1e-5, "b{bi} j{j}: {} vs {}", got[j], e);
        }
    }
}
