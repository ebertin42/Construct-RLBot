use construct_engine::actions::make_immortal_table;

#[test]
fn immortal_table_matches_python_fixture() {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_action.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let rows = v["table"].as_array().unwrap();
    let table = make_immortal_table();
    assert_eq!(table.len(), 126, "row count");
    assert_eq!(rows.len(), 126, "fixture row count");
    for (i, row) in rows.iter().enumerate() {
        for j in 0..8 {
            let expect = row[j].as_f64().unwrap() as f32;
            assert_eq!(table[i][j], expect, "row {i} col {j}");
        }
    }
}
