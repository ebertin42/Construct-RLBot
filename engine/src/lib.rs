use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

pub mod actions;
pub mod curriculum;
pub mod engine;
pub mod episode;
pub mod obs;
pub mod policy;
pub mod reward;
pub mod sampler;
pub mod schema;
pub mod sim_init;
pub mod viser;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn action_table<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
    let t = actions::make_lookup_table();
    let flat: Vec<f32> = t.iter().flatten().copied().collect();
    numpy::ndarray::Array2::from_shape_vec((t.len(), 8), flat).unwrap().into_pyarray(py)
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

// `unsendable`: MultiEngine holds `std::sync::mpsc::Receiver<WorkerOut>` internally
// (per worker), which is Send but not Sync, so pyo3's default Send+Sync pyclass bound
// fails to compile. The actor model itself (worker threads own their arenas; the main
// thread only touches Sender/Receiver ends) is exactly per the brief and is unaffected —
// `unsendable` only pins the *Python-facing* Engine wrapper to the single OS thread that
// constructed it (i.e. the thread holding the GIL when Engine() is called), which matches
// how the trainer contract uses this class (single Python thread drives reset()/step()).
#[pyclass(unsendable)]
struct Engine {
    inner: engine::MultiEngine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (num_arenas=32, blue=1, orange=1, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", meshes_path=None,
                        seed=0, num_threads=0, team_size_weights=None, curriculum_config_path=None))]
    fn new(
        num_arenas: usize,
        blue: usize,
        orange: usize,
        schema_path: &str,
        reward_config_path: &str,
        meshes_path: Option<&str>,
        seed: u32,
        num_threads: usize,
        team_size_weights: Option<Vec<f64>>,
        curriculum_config_path: Option<&str>,
    ) -> PyResult<Self> {
        sim_init::ensure_init(meshes_path);
        let sch = schema::Schema::load(schema_path).map_err(PyValueError::new_err)?;
        if sch.obs_size != obs::OBS_SIZE || sch.action_count != actions::TABLE_SIZE || sch.action_table != "rlgym_lookup_90" {
            return Err(PyValueError::new_err(format!(
                "schema {} disagrees with compiled engine (obs {} vs {}, actions {} vs {}, table {:?})",
                schema_path, sch.obs_size, obs::OBS_SIZE, sch.action_count, actions::TABLE_SIZE, sch.action_table
            )));
        }
        let cfg = reward::RewardConfig::load(reward_config_path).map_err(PyValueError::new_err)?;
        // `sizes[arena] = (blue, orange)`. `None` -> legacy uniform behavior (keeps the
        // asymmetric blue != orange case, e.g. tests' 1v0-style configs, working exactly
        // as before). `Some(w)` -> mixed 1v1/2v2/3v3 arenas via the largest-remainder
        // allocator; blue/orange args are ignored in that case (each mixed arena is size
        // vs size, i.e. (s, s)).
        let sizes: Vec<(usize, usize)> = match team_size_weights {
            Some(w) => {
                if w.len() != 3 || w.iter().any(|x| *x < 0.0) || w.iter().sum::<f64>() <= 0.0 {
                    return Err(PyValueError::new_err(
                        "team_size_weights must be 3 nonnegative floats summing > 0",
                    ));
                }
                engine::allocate_team_sizes(num_arenas, [w[0], w[1], w[2]])
                    .into_iter()
                    .map(|s| (s, s))
                    .collect()
            }
            None => vec![(blue, orange); num_arenas],
        };
        let curriculum = match curriculum_config_path {
            Some(p) => Some(crate::curriculum::CurriculumConfig::load(p).map_err(PyValueError::new_err)?),
            None => None,
        };
        Ok(Engine { inner: engine::MultiEngine::new(sizes, sch, cfg, seed, num_threads, curriculum) })
    }

    #[getter]
    fn num_agents(&self) -> usize { self.inner.num_agents }
    #[getter]
    fn obs_size(&self) -> usize { self.inner.obs_size }
    #[getter]
    fn action_count(&self) -> usize { self.inner.action_count }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let (n, d) = (self.inner.num_agents, self.inner.obs_size);
        let mut obs = vec![0.0f32; n * d];
        py.detach(|| self.inner.reset_into(&mut obs));
        numpy::ndarray::Array2::from_shape_vec((n, d), obs).unwrap().into_pyarray(py)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let acts = actions.as_slice()?.to_vec();
        let (n, d) = (self.inner.num_agents, self.inner.obs_size);
        let (mut obs, mut fin) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let mut rew = vec![0.0f32; n];
        let (mut term, mut trunc) = (vec![false; n], vec![false; n]);
        py.detach(|| {
            self.inner.step_into(&acts, &mut obs, &mut rew, &mut term, &mut trunc, &mut fin)
        })
        .map_err(PyValueError::new_err)?;
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, d), obs).unwrap().into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, d), fin).unwrap().into_pyarray(py),
        ))
    }

    /// Loads a PyTorch `state_dict` (as `dict[str, np.float32 ndarray]`) into every
    /// worker's `MlpPolicy` plus the debug-forward copy held on `MultiEngine`. Keys
    /// follow `construct.learn.model.PolicyValueNet`'s naming: `trunk.{0,2,...}.weight`
    /// / `.bias` (nn.Sequential Linear/ReLU pairs), `policy_head.{weight,bias}`,
    /// `value_head.{weight,bias}`.
    fn set_weights(&mut self, weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>) -> PyResult<()> {
        let arrays: HashMap<String, (Vec<f32>, Vec<usize>)> = weights.into_iter()
            .map(|(k, v)| {
                let shape = v.shape().to_vec();
                (k, (v.as_array().iter().copied().collect(), shape))
            })
            .collect();
        let pw = engine::parse_state_dict(arrays).map_err(PyValueError::new_err)?;
        self.inner.set_weights(pw).map_err(PyValueError::new_err)
    }

    /// Runs `steps` rounds of on-worker rollout (policy-driven actions, sampled with
    /// each arena's own deterministic Pcg32 — see `engine::MultiEngine::collect`) and
    /// returns a dict of numpy arrays for the trainer: `obs (T,N,94) f32`, `actions
    /// (T,N) i64`, `logprobs`/`values`/`rewards`/`final_values (T,N) f32`,
    /// `terminated`/`truncated (T,N) bool`, `last_values (N,) f32`. Requires
    /// `set_weights` to have been called first (else `ValueError`). The gather blocks
    /// for multiple seconds, so it runs under `py.detach` to free the GIL.
    fn collect<'py>(&mut self, py: Python<'py>, steps: usize) -> PyResult<Bound<'py, PyDict>> {
        let out = py.detach(|| self.inner.collect(steps)).map_err(PyValueError::new_err)?;
        let (t, n, d) = (steps, self.inner.num_agents, self.inner.obs_size);
        let obs: Bound<'py, PyArray3<f32>> =
            numpy::ndarray::Array3::from_shape_vec((t, n, d), out.obs).unwrap().into_pyarray(py);
        let actions: Bound<'py, PyArray2<i64>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.actions).unwrap().into_pyarray(py);
        let logprobs: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.logprobs).unwrap().into_pyarray(py);
        let values: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.values).unwrap().into_pyarray(py);
        let rewards: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.rewards).unwrap().into_pyarray(py);
        let terminated: Bound<'py, PyArray2<bool>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.terminated).unwrap().into_pyarray(py);
        let truncated: Bound<'py, PyArray2<bool>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.truncated).unwrap().into_pyarray(py);
        let final_values: Bound<'py, PyArray2<f32>> =
            numpy::ndarray::Array2::from_shape_vec((t, n), out.final_values).unwrap().into_pyarray(py);
        let last_values: Bound<'py, PyArray1<f32>> = out.last_values.into_pyarray(py);

        let dict = PyDict::new(py);
        dict.set_item("obs", obs)?;
        dict.set_item("actions", actions)?;
        dict.set_item("logprobs", logprobs)?;
        dict.set_item("values", values)?;
        dict.set_item("rewards", rewards)?;
        dict.set_item("terminated", terminated)?;
        dict.set_item("truncated", truncated)?;
        dict.set_item("final_values", final_values)?;
        dict.set_item("last_values", last_values)?;
        Ok(dict)
    }

    /// Runs `obs` (np f32, shape (B, obs_size)) through the policy loaded by the most
    /// recent `set_weights` call. Returns `(logits (B, action_count), values (B,))`.
    /// Used for parity testing against the PyTorch reference net — not part of the
    /// training rollout path (see Task 4's Cmd::Collect for that).
    fn debug_policy_forward<'py>(
        &self,
        py: Python<'py>,
        obs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
        let shape = obs.shape();
        let (batch, obs_dim) = (shape[0], shape[1]);
        let flat: Vec<f32> = obs.as_array().iter().copied().collect();
        let (logits, values) = self
            .inner
            .debug_policy_forward(&flat, batch, obs_dim)
            .map_err(PyValueError::new_err)?;
        let action_count = logits.len() / batch.max(1);
        Ok((
            numpy::ndarray::Array2::from_shape_vec((batch, action_count), logits)
                .unwrap()
                .into_pyarray(py),
            values.into_pyarray(py),
        ))
    }

    /// JSON dump of arena state + the engine-built obs for all agents of that arena.
    /// Used by tests/python/test_parity.py to check deploy/obs.py against the Rust
    /// obs builder (see Task 12 interfaces for the JSON schema).
    fn debug_state_and_obs<'py>(
        &mut self,
        py: Python<'py>,
        arena_idx: usize,
    ) -> PyResult<(String, Bound<'py, PyArray2<f32>>)> {
        let (json, obs, agents) = self
            .inner
            .debug_arena(arena_idx)
            .map_err(PyValueError::new_err)?;
        Ok((
            json,
            numpy::ndarray::Array2::from_shape_vec((agents, self.inner.obs_size), obs)
                .unwrap()
                .into_pyarray(py),
        ))
    }
}

// `unsendable`: `EpisodeArena` holds a `cxx::UniquePtr<Arena>`, which contains raw
// pointers (`cxx::private::Opaque`) that aren't `Sync`. Same rationale as `Engine`
// above: this only pins the Python-facing wrapper to the constructing thread, which
// matches how `RenderSession` is used (single Python thread drives reset()/step()).
#[pyclass(unsendable)]
struct RenderSession {
    arena: episode::EpisodeArena,
    stream: viser::ViserStream,
    pacer: viser::Pacer,
    num_agents: usize,
}

#[pymethods]
impl RenderSession {
    #[new]
    #[pyo3(signature = (blue=1, orange=1, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", meshes_path=None, seed=0))]
    fn new(blue: usize, orange: usize, schema_path: &str, reward_config_path: &str,
           meshes_path: Option<&str>, seed: u32) -> PyResult<Self> {
        sim_init::ensure_init(meshes_path);
        let sch = schema::Schema::load(schema_path).map_err(PyValueError::new_err)?;
        if sch.obs_size != obs::OBS_SIZE || sch.action_count != actions::TABLE_SIZE || sch.action_table != "rlgym_lookup_90" {
            return Err(PyValueError::new_err(format!(
                "schema {} disagrees with compiled engine (obs {} vs {}, actions {} vs {}, table {:?})",
                schema_path, sch.obs_size, obs::OBS_SIZE, sch.action_count, actions::TABLE_SIZE, sch.action_table
            )));
        }
        let cfg = reward::RewardConfig::load(reward_config_path).map_err(PyValueError::new_err)?;
        let tick_skip = sch.tick_skip;
        Ok(RenderSession {
            arena: episode::EpisodeArena::new(blue, orange, tick_skip, cfg, sch.normalization, seed),
            stream: viser::ViserStream::new().map_err(|e| PyValueError::new_err(e.to_string()))?,
            pacer: viser::Pacer::new(tick_skip),
            num_agents: blue + orange,
        })
    }

    #[getter]
    fn obs_size(&self) -> usize { obs::OBS_SIZE }
    #[getter]
    fn action_count(&self) -> usize { actions::TABLE_SIZE }
    #[getter]
    fn num_agents(&self) -> usize { self.num_agents }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let (n, d) = (self.num_agents, obs::OBS_SIZE);
        let mut o = vec![0.0f32; n * d];
        self.arena.write_obs(&mut o);
        numpy::ndarray::Array2::from_shape_vec((n, d), o).unwrap().into_pyarray(py)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_in: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let acts = actions_in.as_slice()?.to_vec();
        let (n, d) = (self.num_agents, obs::OBS_SIZE);
        let (mut o, mut fin) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let mut rew = vec![0.0f32; n];
        let mut flags = vec![episode::StepFlags::default(); n];
        self.arena.step(&acts, &mut rew, &mut flags, &mut fin);
        self.arena.write_obs(&mut o);
        let gs = self.arena.game_state();
        let _ = self.stream.send_state(gs);
        self.pacer.pace();
        let term: Vec<bool> = flags.iter().map(|f| f.terminated).collect();
        let trunc: Vec<bool> = flags.iter().map(|f| f.truncated).collect();
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, d), o).unwrap().into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, d), fin).unwrap().into_pyarray(py),
        ))
    }

    fn close(&mut self) {
        self.stream.quit();
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_dict, m)?)?;
    m.add_function(wrap_pyfunction!(action_table, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<RenderSession>()?;
    Ok(())
}
