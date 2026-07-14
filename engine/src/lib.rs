use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod actions;
pub mod obs;
pub mod schema;
pub mod sim_init;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn schema_dict<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let s = crate::schema::Schema::load(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("version", s.version)?;
    d.set_item("obs_size", s.obs_size)?;
    d.set_item("action_table", s.action_table)?;
    d.set_item("action_count", s.action_count)?;
    d.set_item("tick_skip", s.tick_skip)?;
    d.set_item("pos_norm", s.normalization.pos_norm)?;
    d.set_item("vel_norm", s.normalization.vel_norm)?;
    d.set_item("ang_vel_norm", s.normalization.ang_vel_norm)?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_dict, m)?)?;
    Ok(())
}
